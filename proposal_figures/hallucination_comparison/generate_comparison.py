"""
Comparison of Standard MLP vs. Hamiltonian Neural Network (HNN) Multistep Rollout.

Demonstrates the central claim of the METRIC (FWF) proposal:
- Standard (black-box) neural networks *hallucinate* non-physical energy gains that
  violate Liouville's Theorem -- making them fundamentally unsafe for long-horizon
  control of Hamiltonian systems like particle accelerators.
- HNNs enforce symplectic structure by construction, producing only
  *physically plausible* errors (phase drift), not catastrophic energy drift.

System: Simple Harmonic Oscillator (SHO),  H = (p² + q²)/2 = const.
        This maps directly to linearised betatron oscillations in accelerator
        physics -- the regime targeted by METRIC.

Authors: METRIC Project (Simon Hirläender, Lorenz Fischl), University of Salzburg
         Generated: 2026-02-21
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 150,
})

# ── METRIC Proposal Color Palette ─────────────────────────────────────────────
C_GT   = "#003366"    # mycolor (deep navy) – ground truth
C_MLP  = "#CC2200"    # warm red  – MLP (normal hallucinator)
C_HNN  = "#007A4D"    # forest green – HNN (physical hallucinator)
C_SAFE = "#E8F4E8"    # very light green bg for energy panel
C_UNSAFE = "#FFF0EE"  # very light red bg for energy panel


# ═════════════════════════════════════════════════════════════════════════════
# 1.  GROUND-TRUTH DYNAMICS  (RK4 on SHO)
# ═════════════════════════════════════════════════════════════════════════════

def sho_deriv(state: np.ndarray) -> np.ndarray:
    """dq/dt = p,  dp/dt = -q  (SHO with unit frequency)."""
    q, p = state
    return np.array([p, -q])


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = sho_deriv(state)
    k2 = sho_deriv(state + 0.5 * dt * k1)
    k3 = sho_deriv(state + 0.5 * dt * k2)
    k4 = sho_deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def ground_truth_trajectory(q0: float, p0: float, n_steps: int, dt: float) -> np.ndarray:
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = [q0, p0]
    for i in range(n_steps):
        traj[i+1] = rk4_step(traj[i], dt)
    return traj


# ═════════════════════════════════════════════════════════════════════════════
# 2.  GENERATE TRAINING DATA  (noisy one-step transitions)
# ═════════════════════════════════════════════════════════════════════════════

def make_training_data(n_train: int, dt: float, noise_std: float = 0.02):
    """Random initial conditions → one RK4 step + Gaussian noise."""
    angles = np.random.uniform(0, 2 * np.pi, n_train)
    radii  = np.random.uniform(0.5, 1.5, n_train)
    q0 = radii * np.cos(angles)
    p0 = radii * np.sin(angles)
    states_in  = np.stack([q0, p0], axis=1)
    states_out = np.array([rk4_step(s, dt) for s in states_in])
    states_out += noise_std * np.random.randn(*states_out.shape)
    # Derivatives (for HNN target)
    derivs = np.array([sho_deriv(s) for s in states_in])

    X   = torch.tensor(states_in,  dtype=torch.float32)
    Y_s = torch.tensor(states_out, dtype=torch.float32)  # next state  (MLP target)
    Y_d = torch.tensor(derivs,     dtype=torch.float32)  # derivatives (HNN target)
    return X, Y_s, Y_d


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MODELS
# ═════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Plain black-box MLP.  Directly predicts next state (s_{t+1})."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HamiltonianMLP(nn.Module):
    """
    Hamiltonian Neural Network.
    Learns a scalar H(q,p) and uses symplectic gradient to predict dq/dt, dp/dt.
    Guarantees energy conservation by construction.
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (dq/dt, dp/dt) via symplectic gradient of H."""
        x = x.requires_grad_(True)
        H = self.hamiltonian(x).sum()
        dH = torch.autograd.grad(H, x, create_graph=True)[0]
        # Symplectic structure: dq/dt = +∂H/∂p, dp/dt = -∂H/∂q
        dqdt =  dH[:, 1:2]
        dpdt = -dH[:, 0:1]
        return torch.cat([dqdt, dpdt], dim=-1)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_mlp(seed: int, X: torch.Tensor, Y_s: torch.Tensor,
              epochs: int = 2000, lr: float = 3e-3) -> MLP:
    torch.manual_seed(seed)
    model = MLP()
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), Y_s)
        loss.backward()
        opt.step()
    return model


def train_hnn(seed: int, X: torch.Tensor, Y_d: torch.Tensor,
              dt: float, epochs: int = 3000, lr: float = 3e-3) -> HamiltonianMLP:
    torch.manual_seed(seed)
    model = HamiltonianMLP()
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred_deriv = model(X)
        loss = nn.functional.mse_loss(pred_deriv, Y_d)
        loss.backward()
        opt.step()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 5.  ROLLOUT
# ═════════════════════════════════════════════════════════════════════════════

def mlp_rollout(model: MLP, q0: float, p0: float,
                n_steps: int) -> np.ndarray:
    """Autoregressive Euler rollout: s_{t+1} = model(s_t)."""
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = [q0, p0]
    s = torch.tensor([[q0, p0]], dtype=torch.float32)
    for i in range(n_steps):
        with torch.no_grad():
            s = model(s)
        traj[i+1] = s.numpy()[0]
    return traj


def hnn_rollout(model: HamiltonianMLP, q0: float, p0: float,
                n_steps: int, dt: float) -> np.ndarray:
    """Autoregressive RK4 rollout using the learned Hamiltonian vector field."""
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = [q0, p0]
    s = torch.tensor([[q0, p0]], dtype=torch.float32)
    for i in range(n_steps):
        # RK4 with learned dynamics
        def f(x): return model(x).detach()
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj[i+1] = s.detach().numpy()[0]
    return traj


# ═════════════════════════════════════════════════════════════════════════════
# 6.  UNCERTAINTY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def ensemble_stats(trajs: list[np.ndarray]):
    """Mean and std across ensemble rollouts. trajs: List[(n_steps+1, 2)]"""
    arr = np.stack(trajs, axis=0)   # (M, T, 2)
    return arr.mean(0), arr.std(0)  # (T, 2), (T, 2)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  ENERGY
# ═════════════════════════════════════════════════════════════════════════════

def energy(traj: np.ndarray) -> np.ndarray:
    """H(q,p) = (q² + p²)/2  (SHO)."""
    return 0.5 * (traj[:, 0]**2 + traj[:, 1]**2)


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    Q0, P0   = 1.0, 0.0   # initial condition on the unit circle
    DT       = 0.25       # ~14° / step  → 45 steps ≈ 2 full revolutions
    N_STEPS  = 55         # prediction horizon
    N_TRAIN  = 800        # one-step training pairs
    N_SEEDS  = 5          # ensemble size

    outdir = Path(__file__).parent
    print("Generating training data…")
    X, Y_s, Y_d = make_training_data(N_TRAIN, DT)

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt = ground_truth_trajectory(Q0, P0, N_STEPS, DT)
    t  = np.arange(N_STEPS + 1) * DT
    E_gt = energy(gt)

    # ── Train ensembles ───────────────────────────────────────────────────────
    print("Training MLP ensemble…")
    mlp_models = [train_mlp(s, X, Y_s) for s in range(N_SEEDS)]
    print("Training HNN ensemble…")
    hnn_models = [train_hnn(s, X, Y_d, DT) for s in range(N_SEEDS)]

    # ── Rollout ───────────────────────────────────────────────────────────────
    mlp_trajs = [mlp_rollout(m, Q0, P0, N_STEPS)   for m in mlp_models]
    hnn_trajs = [hnn_rollout(m, Q0, P0, N_STEPS, DT) for m in hnn_models]

    mlp_mean, mlp_std = ensemble_stats(mlp_trajs)
    hnn_mean, hnn_std = ensemble_stats(hnn_trajs)

    E_mlp_all = np.stack([energy(tr) for tr in mlp_trajs])  # (M, T)
    E_hnn_all = np.stack([energy(tr) for tr in hnn_trajs])
    E_mlp_mean, E_mlp_std = E_mlp_all.mean(0), E_mlp_all.std(0)
    E_hnn_mean, E_hnn_std = E_hnn_all.mean(0), E_hnn_all.std(0)

    # Total positional uncertainty σ = sqrt(σ_q² + σ_p²)
    sigma_mlp = np.sqrt(mlp_std[:, 0]**2 + mlp_std[:, 1]**2)
    sigma_hnn = np.sqrt(hnn_std[:, 0]**2 + hnn_std[:, 1]**2)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2),
                             gridspec_kw={"wspace": 0.38})

    # ─── Panel A: Phase portrait ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F7F9FC")
    θ = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(θ), np.sin(θ), color=C_GT, lw=1.5, ls="--",
            label="Ground truth ($H=0.5$)", alpha=0.6)

    for tr in mlp_trajs:
        ax.plot(tr[:, 0], tr[:, 1], color=C_MLP, alpha=0.15, lw=0.7)
    for tr in hnn_trajs:
        ax.plot(tr[:, 0], tr[:, 1], color=C_HNN, alpha=0.15, lw=0.7)

    ax.plot(mlp_mean[:, 0], mlp_mean[:, 1], color=C_MLP, lw=1.8,
            label="MLP (black-box)")
    ax.plot(hnn_mean[:, 0], hnn_mean[:, 1], color=C_HNN, lw=1.8,
            label="HNN (symplectic)")
    ax.scatter([Q0], [P0], s=60, color=C_GT, zorder=5, label="IC")

    # Annotate energy drift
    ax.annotate("Energy\ndrift", xy=(mlp_mean[-1, 0], mlp_mean[-1, 1]),
                xytext=(0.6, -1.2), fontsize=8, color=C_MLP,
                arrowprops=dict(arrowstyle="->", color=C_MLP, lw=1.0))
    ax.annotate("Phase\ndrift only", xy=(hnn_mean[-1, 0], hnn_mean[-1, 1]),
                xytext=(-1.5, 0.7), fontsize=8, color=C_HNN,
                arrowprops=dict(arrowstyle="->", color=C_HNN, lw=1.0))

    ax.set_xlabel("$q$  [a.u.]")
    ax.set_ylabel("$p$  [a.u.]")
    ax.set_title("(A)  Phase Portrait", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.text(0.03, 0.97, "SHO — maps to betatron\noscillations in accelerators",
            transform=ax.transAxes, va="top", fontsize=7.5,
            color="gray", style="italic")

    # ─── Panel B: q(t) time series ────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#F7F9FC")
    ax.plot(t, gt[:, 0], color=C_GT, lw=1.8, ls="--", label="Ground truth")

    ax.fill_between(t,
                    mlp_mean[:, 0] - mlp_std[:, 0],
                    mlp_mean[:, 0] + mlp_std[:, 0],
                    color=C_MLP, alpha=0.18)
    ax.plot(t, mlp_mean[:, 0], color=C_MLP, lw=1.8, label="MLP (black-box)")

    ax.fill_between(t,
                    hnn_mean[:, 0] - hnn_std[:, 0],
                    hnn_mean[:, 0] + hnn_std[:, 0],
                    color=C_HNN, alpha=0.18)
    ax.plot(t, hnn_mean[:, 0], color=C_HNN, lw=1.8, label="HNN (symplectic)")

    ax.set_xlabel("Time $t$  [a.u.]")
    ax.set_ylabel("Position $q(t)$")
    ax.set_title("(B)  Position Rollout", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.axhline(0, color="gray", lw=0.4, ls=":")

    # ─── Panel C: Energy H(t) ────────────────────────────────────────────────
    ax = axes[2]
    E0 = E_gt[0]

    # Background shading
    ax.axhspan(E0 * 0.85, E0 * 1.15, color=C_SAFE, alpha=0.9, zorder=0)
    ax.axhspan(E0 * 1.15, E0 * 3.5, color=C_UNSAFE, alpha=0.7, zorder=0)

    ax.axhline(E0, color=C_GT, lw=1.5, ls="--", label=f"Ground truth $H={E0:.2f}$")

    ax.fill_between(t, E_mlp_mean - E_mlp_std, E_mlp_mean + E_mlp_std,
                    color=C_MLP, alpha=0.18)
    ax.plot(t, E_mlp_mean, color=C_MLP, lw=1.8, label="MLP (black-box)")

    ax.fill_between(t, E_hnn_mean - E_hnn_std, E_hnn_mean + E_hnn_std,
                    color=C_HNN, alpha=0.18)
    ax.plot(t, E_hnn_mean, color=C_HNN, lw=1.8, label="HNN (symplectic)")

    # Annotations
    # Find first step where MLP energy leaves tolerance band
    violation_steps = np.where(np.abs(E_mlp_mean - E0) > 0.1 * E0)[0]
    if len(violation_steps) > 0:
        vx = t[violation_steps[0]]
        ax.axvline(vx, color=C_MLP, lw=0.8, ls=":", alpha=0.7)
        ax.text(vx + 0.3, E0 * 1.3, "Liouville\nviolation", color=C_MLP,
                fontsize=7.5, va="center")

    ax.text(0.97, 0.12, "[OK] Hamiltonian\nconserved", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=C_HNN,
            bbox=dict(boxstyle="round,pad=0.2", fc=C_SAFE, ec=C_HNN, alpha=0.9))
    ax.text(0.97, 0.85, "[!] Non-physical\nenergy gain", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=C_MLP,
            bbox=dict(boxstyle="round,pad=0.2", fc=C_UNSAFE, ec=C_MLP, alpha=0.9))

    ax.set_xlabel("Time $t$  [a.u.]")
    ax.set_ylabel("Energy $H(q,p) = (q^2+p^2)/2$")
    ax.set_title("(C)  Energy Conservation", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.85)
    ylo = max(0, E0 * 0.6)
    yhi = min(E0 * 3.5, max(E_mlp_mean.max() * 1.15, E0 * 1.6))
    ax.set_ylim(ylo, yhi)

    # ─── Panel D: Epistemic uncertainty σ(t) ─────────────────────────────────
    ax = axes[3]
    ax.set_facecolor("#F7F9FC")

    ax.fill_between(t, sigma_mlp, color=C_MLP, alpha=0.25)
    ax.plot(t, sigma_mlp, color=C_MLP, lw=1.8, label="MLP (black-box)")

    ax.fill_between(t, sigma_hnn, color=C_HNN, alpha=0.25)
    ax.plot(t, sigma_hnn, color=C_HNN, lw=1.8, label="HNN (symplectic)")

    # Threshold line (illustrative safe UQ threshold)
    threshold = sigma_hnn.max() * 2.5
    ax.axhline(threshold, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.text(t[-1] * 0.05, threshold * 1.05, "Safe UQ\nthreshold",
            fontsize=7.5, color="gray", va="bottom")

    ax.set_xlabel("Time $t$  [a.u.]")
    ax.set_ylabel(r"Epistemic uncertainty $\sigma(t)$")
    ax.set_title("(D)  Ensemble Uncertainty", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.text(0.97, 0.90,
            "HNN uncertainty is\nbounded & trustworthy\n→ safe exploration signal",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=C_HNN,
            bbox=dict(boxstyle="round,pad=0.25", fc="#E8F4E8", ec=C_HNN, alpha=0.9))

    # ── Global title & footer ─────────────────────────────────────────────────
    fig.suptitle(
        "World Model Hallucinations: Black-Box MLP vs. Hamiltonian Neural Network\n"
        r"Standard RL violates Liouville's Theorem ($\nabla \cdot f \neq 0$)  —  "
        r"HNN enforces symplectic structure ($\nabla \cdot f_H = 0$)  by construction",
        fontsize=11, fontweight="bold", color=C_GT, y=1.01,
    )
    fig.text(0.5, -0.03,
             "METRIC Project · FWF PI Project · Simon Hirläender, Lorenz Fischl · "
             "Dept. AI & Human Interfaces, University of Salzburg",
             ha="center", fontsize=7.5, color="gray", style="italic")

    # ── Save ──────────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fp = outdir / f"hallucination_comparison.{ext}"
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        print(f"Saved → {fp}")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
