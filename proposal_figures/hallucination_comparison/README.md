# World Model Hallucinations: MLP vs. HNN

> **Proposal figure for the METRIC FWF PI Project**
> Authors: Simon Hirläender, Lorenz Fischl · University of Salzburg

## What this figure shows

This folder contains a **self-contained, reproducible Python script** that generates a 4-panel comparison figure demonstrating the core scientific claim of the METRIC proposal:

> *"Without structural constraints, learned models essentially 'hallucinate' non-physical energy gains, leading to policies that are mathematically unstable and operationally dangerous."*
> — METRIC Proposal, §1 Scientific Aspects

### The two failure modes

| Model | Failure mode | Technical cause |
|-------|-------------|----------------|
| **Standard MLP** (black-box RL world model) | **Normal hallucination**: energy grows/decays wildly → trajectory spirals, policy destabilises | No symplectic inductive bias; MSE loss ignores conservation laws |
| **HNN** (Hamiltonian Neural Network) | **Physical hallucination**: energy conserved but phase may drift | Learns an *imperfect* Hamiltonian — physically plausible errors only |

### Figure panels

| Panel | Content | Claim supported |
|-------|---------|----------------|
| **(A) Phase portrait** | MLP spiral vs HNN ring vs GT circle | Liouville: MLP violates phase-space area preservation |
| **(B) Position q(t)** | MLP diverges; HNN oscillates with phase drift | Long-horizon predictability fails without structure |
| **(C) Energy H(t)** | MLP energy explodes; HNN stays at H=0.5 | Symplectic structure guarantees energy conservation |
| **(D) Uncertainty σ(t)** | MLP uncertainty blows up; HNN bounded | HNN epistemic UQ is trustworthy → safe exploration signal |

### System

**Simple Harmonic Oscillator** `(q̈ = -q)`, `H = (p² + q²)/2 = const`.
Directly maps to linearised **betatron oscillations** in particle accelerators — the regime METRIC targets.

## Reproducing the figure

```bash
# From the repository root (requires Python 3.10+, torch, numpy, matplotlib)
pip install torch numpy matplotlib
cd proposal_figures/hallucination_comparison
python generate_comparison.py
```

Outputs: `hallucination_comparison.pdf` and `hallucination_comparison.png` (300 dpi).
Runtime: ~60 s on a modern laptop (CPU only).

## How to cite in the proposal

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{proposal_figures/hallucination_comparison/hallucination_comparison}
  \caption{%
    \textbf{Physical vs.\ Normal World Model Hallucinations.}
    Left to right: (\textbf{A}) Phase portrait, (\textbf{B}) position rollout,
    (\textbf{C}) energy conservation, (\textbf{D}) epistemic uncertainty.
    A standard MLP world model (red) violates Liouville's Theorem:
    energy drifts catastrophically under autoregressive rollout, making
    model-based policies operationally unsafe.
    The Hamiltonian Neural Network (green) enforces symplectic structure,
    conserving energy by construction---errors manifest only as
    physically plausible phase drift.
    System: Simple Harmonic Oscillator ($\ddot{q}=-q$), directly analogous
    to betatron oscillations in particle accelerators.
    Ensemble size $M=5$; shaded bands indicate $\pm 1\sigma$ epistemic uncertainty.
  }
  \label{fig:hallucination_comparison}
\end{figure}
```

## Dependencies

See `../../requirements.txt`. Minimum:

- `torch >= 2.0`
- `numpy >= 1.24`
- `matplotlib >= 3.7`
