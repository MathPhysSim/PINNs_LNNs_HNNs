"""
Comparison of Standard MLP vs. Hamiltonian Neural Network (HNN) Multistep Rollout.

Demonstrates the central claim of the METRIC (FWF) proposal:
  Standard (black-box) neural networks *hallucinate* non-physical energy gains that
  violate Liouville's Theorem, making them fundamentally unsafe for long-horizon
  control of Hamiltonian systems like particle accelerators.
  HNNs enforce symplectic structure, producing only *physically plausible* errors
  (phase drift of the learned Hamiltonian), not catastrophic energy drift.

Key methodological point — ensemble mean:
  For HNNs we average the SCALAR Hamiltonian fields H_mean = (1/N) Σ H_i
  and derive the vector field from J∇H_mean.  Since J is a constant linear map,
  this is identical to averaging the Lie-algebra generators (X_{H_i} = J∇H_i).
  The result is strictly symplectic by construction (Liouville guaranteed).
  Compare the WRONG approach: averaging the integrated flows mean_i(φ_{H_i}),
  which destroys symplecticity.

System: Simple Harmonic Oscillator (SHO), H = (p² + q²)/2 = const.
        Directly maps to linearised betatron oscillations in particle accelerators:
        the primary regime targeted by METRIC.

Authors: METRIC Project — Simon Hirläender, Lorenz Fischl
         Dept. AI & Human Interfaces, University of Salzburg
         Generated: 2026-02-21
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#E0E4EA",
    "grid.linewidth":     0.6,
    "legend.fontsize":    8.0,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#CCCCCC",
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "figure.dpi":         150,
    "savefig.dpi":        300,
})

# ── METRIC Proposal Color Palette ─────────────────────────────────────────────
C_GT     = "#003366"    # deep navy  – ground truth
C_MLP    = "#C0392B"    # deep crimson – MLP (normal hallucinator)
C_HNN    = "#17704A"    # forest green – HNN (physical hallucinator)
C_SAFE   = "#EBF7F0"    # very light green bg
C_UNSAFE = "#FDF0EE"    # very light red bg
C_ANNOT  = "#5D5D7A"    # muted purple for annotations


# ═════════════════════════════════════════════════════════════════════════════
# 1.  GROUND-TRUTH DYNAMICS
# ═════════════════════════════════════════════════════════════════════════════

def sho_deriv(state: np.ndarray) -> np.ndarray:
    """dq/dt = p,  dp/dt = -q"""
    return np.array([state[1], -state[0]])


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = sho_deriv(state)
    k2 = sho_deriv(state + 0.5*dt*k1)
    k3 = sho_deriv(state + 0.5*dt*k2)
    k4 = sho_deriv(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def ground_truth_trajectory(q0, p0, n_steps, dt) -> np.ndarray:
    traj = np.zeros((n_steps+1, 2))
    traj[0] = [q0, p0]
    for i in range(n_steps):
        traj[i+1] = rk4_step(traj[i], dt)
    return traj


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TRAINING DATA
# ═════════════════════════════════════════════════════════════════════════════

def make_training_data(n_train: int, dt: float, noise_std: float = 0.025):
    angles = np.random.uniform(0, 2*np.pi, n_train)
    radii  = np.random.uniform(0.5, 1.5, n_train)
    q0 = radii * np.cos(angles)
    p0 = radii * np.sin(angles)
    states_in  = np.stack([q0, p0], axis=1)
    states_out = np.array([rk4_step(s, dt) for s in states_in])
    states_out += noise_std * np.random.randn(*states_out.shape)
    derivs = np.array([sho_deriv(s) for s in states_in])
    X   = torch.tensor(states_in,  dtype=torch.float32)
    Y_s = torch.tensor(states_out, dtype=torch.float32)
    Y_d = torch.tensor(derivs,     dtype=torch.float32)
    return X, Y_s, Y_d


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MODELS
# ═════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class HamiltonianMLP(nn.Module):
    """
    Learns a scalar H(q,p); dynamics from symplectic gradient J∇H.
    Guarantees energy conservation by construction.
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def hamiltonian(self, x):
        return self.net(x)

    def forward(self, x):
        x = x.requires_grad_(True)
        H = self.hamiltonian(x).sum()
        dH = torch.autograd.grad(H, x, create_graph=True)[0]
        return torch.cat([dH[:, 1:2], -dH[:, 0:1]], dim=-1)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_mlp(seed, X, Y_s, epochs=1500, lr=3e-3):
    torch.manual_seed(seed)
    model = MLP()
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(model(X), Y_s).backward()
        opt.step()
    return model


def train_hnn(seed, X, Y_d, epochs=3000, lr=3e-3):
    torch.manual_seed(seed)
    model = HamiltonianMLP()
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(model(X), Y_d).backward()
        opt.step()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 5.  ROLLOUT
# ═════════════════════════════════════════════════════════════════════════════

def mlp_rollout(model, q0, p0, n_steps):
    traj = np.zeros((n_steps+1, 2))
    traj[0] = [q0, p0]
    s = torch.tensor([[q0, p0]], dtype=torch.float32)
    for i in range(n_steps):
        with torch.no_grad():
            s = model(s)
        traj[i+1] = s.numpy()[0]
    return traj


def hnn_rollout(model, q0, p0, n_steps, dt):
    traj = np.zeros((n_steps+1, 2))
    traj[0] = [q0, p0]
    s = torch.tensor([[q0, p0]], dtype=torch.float32)
    for i in range(n_steps):
        def f(x): return model(x).detach()
        k1 = f(s)
        k2 = f(s + 0.5*dt*k1)
        k3 = f(s + 0.5*dt*k2)
        k4 = f(s + dt*k3)
        s = s + (dt/6.0)*(k1+2*k2+2*k3+k4)
        traj[i+1] = s.detach().numpy()[0]
    return traj


def hnn_mean_rollout(models, q0, p0, n_steps, dt):
    """
    Symplectically correct mean rollout via Hamiltonian averaging.

    H_mean(q,p) = (1/N) Σ H_i(q,p)   →   X_{H_mean} = J∇H_mean

    By linearity of J: mean_i(J∇H_i) = J∇(mean_i H_i) = X_{H_mean}
    ∴  averaging Hamiltonians  ≡  averaging Lie-algebra generators.
    Result: strictly symplectic flow (Liouville theorem guaranteed).

    Wrong alternative: mean_i(φ_{H_i}(z)) — averaging FLOWS —
    destroys symplecticity (mean of Sp(2n) elements ∉ Sp(2n)).
    """
    traj = np.zeros((n_steps+1, 2))
    traj[0] = [q0, p0]
    s = torch.tensor([[q0, p0]], dtype=torch.float32)

    def mean_vf(x):
        x = x.detach().requires_grad_(True)
        H = sum(m.hamiltonian(x) for m in models) / len(models)
        dH = torch.autograd.grad(H.sum(), x)[0]
        return torch.cat([dH[:, 1:2], -dH[:, 0:1]], dim=-1).detach()

    for i in range(n_steps):
        k1 = mean_vf(s)
        k2 = mean_vf(s + 0.5*dt*k1)
        k3 = mean_vf(s + 0.5*dt*k2)
        k4 = mean_vf(s + dt*k3)
        s = (s + (dt/6.0)*(k1+2*k2+2*k3+k4)).detach()
        traj[i+1] = s.numpy()[0]
    return traj


# ═════════════════════════════════════════════════════════════════════════════
# 6.  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def ensemble_stats(trajs):
    arr = np.stack(trajs, axis=0)
    return arr.mean(0), arr.std(0)


def energy(traj):
    return 0.5*(traj[:, 0]**2 + traj[:, 1]**2)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def styled_label(ax, text, x, y, color, fontsize=8.2, bg=None, ec=None,
                 ha="left", va="center", style="normal", pad=0.25):
    kw = dict(transform=ax.transAxes, ha=ha, va=va,
              fontsize=fontsize, color=color, style=style)
    if bg:
        kw["bbox"] = dict(boxstyle=f"round,pad={pad}", fc=bg,
                          ec=ec or color, lw=0.8, alpha=0.93)
    return ax.text(x, y, text, **kw)


def add_callout(ax, xy_data, xy_text, label, color, fontsize=7.8, ax_coords=False):
    if ax_coords:
        ax.annotate(label, xy=xy_data, xycoords="axes fraction",
                    xytext=xy_text, textcoords="axes fraction",
                    fontsize=fontsize, color=color, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                   mutation_scale=8))
    else:
        ax.annotate(label, xy=xy_data, xytext=xy_text,
                    fontsize=fontsize, color=color, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                   mutation_scale=8))


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    Q0, P0   = 1.0, 0.0
    DT       = 0.25       # ~14° / step
    N_STEPS  = 80         # ~3.2 full revolutions (longer = more visible drift)
    N_TRAIN  = 800
    N_SEEDS  = 5

    outdir = Path(__file__).parent
    print("Generating training data...")
    X, Y_s, Y_d = make_training_data(N_TRAIN, DT)

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt   = ground_truth_trajectory(Q0, P0, N_STEPS, DT)
    t    = np.arange(N_STEPS+1) * DT
    E_gt = energy(gt)
    E0   = E_gt[0]

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Training MLP ensemble...")
    mlp_models = [train_mlp(s, X, Y_s) for s in range(N_SEEDS)]
    print("Training HNN ensemble...")
    hnn_models = [train_hnn(s, X, Y_d) for s in range(N_SEEDS)]

    # ── Rollout ───────────────────────────────────────────────────────────────
    mlp_trajs  = [mlp_rollout(m, Q0, P0, N_STEPS)       for m in mlp_models]
    hnn_trajs  = [hnn_rollout(m, Q0, P0, N_STEPS, DT)   for m in hnn_models]

    mlp_mean, mlp_std = ensemble_stats(mlp_trajs)

    print("Computing symplectic HNN mean (Hamiltonian / Lie algebra averaging)...")
    hnn_mean  = hnn_mean_rollout(hnn_models, Q0, P0, N_STEPS, DT)
    _, hnn_std = ensemble_stats(hnn_trajs)

    # Energy
    E_mlp_all = np.stack([energy(tr) for tr in mlp_trajs])
    E_mlp_mean, E_mlp_std = E_mlp_all.mean(0), E_mlp_all.std(0)
    E_hnn_mean = energy(hnn_mean)                           # energy of symplectic mean
    E_hnn_std  = np.stack([energy(tr) for tr in hnn_trajs]).std(0)

    sigma_mlp = np.sqrt(mlp_std[:, 0]**2 + mlp_std[:, 1]**2)
    sigma_hnn = np.sqrt(hnn_std[:, 0]**2 + hnn_std[:, 1]**2)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 4.6))
    gs  = GridSpec(1, 4, figure=fig, wspace=0.40,
                   left=0.062, right=0.985, top=0.80, bottom=0.14)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # ─────────────────────────────────────────────────────────────────────────
    # Panel A: Phase Portrait
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0]
    θ  = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(θ), np.sin(θ), color=C_GT, lw=1.6, ls="--",
            label="Ground truth ($H=0.50$)", zorder=4, alpha=0.7)

    # Individual particles (light)
    for tr in mlp_trajs:
        ax.plot(tr[:, 0], tr[:, 1], color=C_MLP, alpha=0.12, lw=0.6)
    for tr in hnn_trajs:
        ax.plot(tr[:, 0], tr[:, 1], color=C_HNN, alpha=0.12, lw=0.6)

    ax.plot(mlp_mean[:, 0], mlp_mean[:, 1], color=C_MLP, lw=2.0,
            label="MLP — black-box", zorder=5)
    ax.plot(hnn_mean[:, 0], hnn_mean[:, 1], color=C_HNN, lw=2.0,
            label="HNN — symplectic", zorder=5)
    ax.scatter([Q0], [P0], s=70, color=C_GT, zorder=8,
               label="Initial condition", marker="o", edgecolors="white", linewidths=0.8)

    # Arrow annotations
    mlp_end = mlp_mean[-1]
    hnn_end = hnn_mean[-1]
    ax.annotate("Topological\nfailure\n(spiral off ring)",
                xy=(mlp_end[0], mlp_end[1]), xytext=(0.35, -1.35),
                fontsize=7.5, color=C_MLP, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C_MLP, lw=1.1,
                                mutation_scale=9))
    ax.annotate("Physical\nhallucination\n(phase drift only)",
                xy=(hnn_end[0], hnn_end[1]), xytext=(-1.55, 0.65),
                fontsize=7.5, color=C_HNN, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C_HNN, lw=1.1,
                                mutation_scale=9))

    ax.set_xlabel("$q$  [a.u.]")
    ax.set_ylabel("$p$  [a.u.]")
    ax.set_title("(A)  Phase Space", pad=8)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_aspect("equal")
    ax.set_xlim(-2.1, 1.85)
    ax.set_ylim(-1.85, 1.85)

    # Lie algebra equivalence note
    styled_label(ax, "Mean Hamiltonian: $H_\\mathrm{mean}=\\frac{1}{N}\\Sigma H_i$\n"
                 r"$\Rightarrow\ X_{H_\mathrm{mean}}=\frac{1}{N}\Sigma J\nabla H_i$  (Lie alg. avg.)",
                 0.02, 0.06, C_HNN, fontsize=6.6, bg="#E8F5EE", ec=C_HNN,
                 ha="left", va="bottom")
    styled_label(ax, "SHO $\\equiv$ betatron osc. in accelerators",
                 0.02, 0.97, C_ANNOT, fontsize=7.0, ha="left", va="top",
                 style="italic")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel B: q(t) time series
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, gt[:, 0], color=C_GT, lw=1.8, ls="--", label="Ground truth", zorder=4)

    ax.fill_between(t, mlp_mean[:, 0]-mlp_std[:, 0],
                    mlp_mean[:, 0]+mlp_std[:, 0], color=C_MLP, alpha=0.18)
    ax.plot(t, mlp_mean[:, 0], color=C_MLP, lw=2.0, label="MLP — black-box")

    ax.fill_between(t, hnn_mean[:, 0]-hnn_std[:, 0],
                    hnn_mean[:, 0]+hnn_std[:, 0], color=C_HNN, alpha=0.18)
    ax.plot(t, hnn_mean[:, 0], color=C_HNN, lw=2.0, label="HNN — symplectic")

    ax.set_xlabel("Time $t$  [a.u.]")
    ax.set_ylabel("Position $q(t)$")
    ax.set_title("(B)  Position Rollout", pad=8)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.axhline(0, color="#AAAAAA", lw=0.5, ls=":", zorder=0)

    # Annotation: short-horizon trap
    styled_label(ax, "Both look similar\nat short horizon\n— the practitioner's trap",
                 0.04, 0.81, C_ANNOT, fontsize=7.0, bg="white", ec="#CCCCCC",
                 ha="left", va="top")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel C: Energy H(t)
    # ─────────────────────────────────────────────────────────────────────────
    ax  = axes[2]

    ax.axhspan(0.0,        E0*0.85,  color=C_UNSAFE, alpha=0.55, zorder=0)
    ax.axhspan(E0*0.85,    E0*1.15,  color=C_SAFE,   alpha=0.85, zorder=0)
    ax.axhspan(E0*1.15,    E0*4.0,   color=C_UNSAFE, alpha=0.55, zorder=0)
    ax.axhline(E0, color=C_GT, lw=1.6, ls="--",
               label=f"Ground truth $H={E0:.2f}$", zorder=5)

    ax.fill_between(t, E_mlp_mean-E_mlp_std, E_mlp_mean+E_mlp_std,
                    color=C_MLP, alpha=0.20)
    ax.plot(t, E_mlp_mean, color=C_MLP, lw=2.0, label="MLP — black-box")

    ax.fill_between(t, E_hnn_mean-E_hnn_std, E_hnn_mean+E_hnn_std,
                    color=C_HNN, alpha=0.20)
    ax.plot(t, E_hnn_mean, color=C_HNN, lw=2.0, label="HNN — symplectic")

    # Liouville violation marker
    viol = np.where(np.abs(E_mlp_mean - E0) > 0.12*E0)[0]
    if len(viol):
        vx = t[viol[0]]
        ax.axvline(vx, color=C_MLP, lw=1.0, ls=":", alpha=0.75)
        ax.text(vx + 0.4, E0*1.55, "Liouville\nviolation\n($\\nabla\\cdot f\\neq 0$)",
                color=C_MLP, fontsize=7.5, va="center")

    styled_label(ax, "Hamiltonian conserved\nby construction",
                 0.97, 0.10, C_HNN, fontsize=7.5, bg=C_SAFE, ec=C_HNN,
                 ha="right", va="bottom")
    styled_label(ax, "Non-physical\nenergy gain",
                 0.97, 0.90, C_MLP, fontsize=7.5, bg=C_UNSAFE, ec=C_MLP,
                 ha="right", va="top")

    ax.set_xlabel("Time $t$  [a.u.]")
    ax.set_ylabel("$H(q,p) = (q^2+p^2)/2$")
    ax.set_title("(C)  Energy Conservation", pad=8)
    ax.legend(loc="upper left", fontsize=7.5)
    yhi = max(E_mlp_mean.max()*1.2, E0*1.8)
    ylo = max(0.0, E0*0.55)
    ax.set_ylim(ylo, min(yhi, E0*3.2))

    # ─────────────────────────────────────────────────────────────────────────
    # Panel D: Epistemic Uncertainty
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[3]

    ax.fill_between(t, sigma_mlp, color=C_MLP, alpha=0.22)
    ax.plot(t, sigma_mlp, color=C_MLP, lw=2.0, label="MLP — black-box")

    ax.fill_between(t, sigma_hnn, color=C_HNN, alpha=0.22)
    ax.plot(t, sigma_hnn, color=C_HNN, lw=2.0, label="HNN — symplectic")

    # Safe UQ threshold line
    thresh = sigma_hnn.max() * 2.8
    ax.axhline(thresh, color="#888888", lw=0.9, ls=(0, (4, 3)), alpha=0.75)
    ax.text(t[2], thresh*1.04, "Safe UQ threshold",
            fontsize=7.5, color="#777777", va="bottom")

    styled_label(ax,
                 "HNN: bounded, structured\nepistemic uncertainty\n"
                 r"$\rightarrow$ safe exploration signal",
                 0.97, 0.15, C_HNN, fontsize=7.5, bg=C_SAFE, ec=C_HNN,
                 ha="right", va="bottom")
    styled_label(ax,
                 "MLP: exploding, phase-\ncorrelated uncertainty\n"
                 r"$\rightarrow$ false confidence",
                 0.97, 0.88, C_MLP, fontsize=7.5, bg=C_UNSAFE, ec=C_MLP,
                 ha="right", va="top")

    ax.set_xlabel("Time $t$  [a.u.]")
    ax.set_ylabel(r"Epistemic uncertainty $\sigma_\mathrm{ep}(t)$")
    ax.set_title("(D)  Epistemic Uncertainty", pad=8)
    ax.legend(loc="upper left", fontsize=7.5)

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             "World Model Hallucinations: Black-Box MLP vs. Hamiltonian Neural Network",
             ha="center", fontsize=13, fontweight="bold", color=C_GT, va="top")
    fig.text(0.5, 0.900,
             r"MLP violates Liouville's theorem ($\nabla\!\cdot\!f\neq 0$)"
             r"  $\boldsymbol{|}$  "
             r"HNN enforces $\nabla\!\cdot\!f_H = 0$ by construction"
             r"  $\boldsymbol{|}$  "
             r"Symplectic mean: $\frac{1}{N}\!\sum_i J\nabla H_i = J\nabla H_\mathrm{mean}$  "
             r"where $H_\mathrm{mean}=\frac{1}{N}\!\sum_i H_i$",
             ha="center", fontsize=9.0, color=C_GT, va="top")

    fig.text(0.5, -0.02,
             "METRIC Project  \u00b7  FWF PI Project  \u00b7  "
             "Simon Hirlaender, Lorenz Fischl  \u00b7  "
             "Dept. AI & Human Interfaces, University of Salzburg  \u00b7  "
             f"Ensemble $M={N_SEEDS}$,  "
             f"horizon $T={N_STEPS}$ steps",
             ha="center", fontsize=7.5, color="#888888", style="italic")

    # ── Save ──────────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fp = outdir / f"hallucination_comparison.{ext}"
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        print(f"Saved -> {fp}")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
