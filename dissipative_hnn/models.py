"""Neural network models for Dissipative Hamiltonian Neural Networks.

Provides three model architectures:
    - MLP: A residual multi-layer perceptron baseline.
    - DHNN: Dissipative Hamiltonian Neural Network that decomposes vector fields
      into conservative (Hamiltonian) and dissipative (Rayleigh) components.
    - HNN: Standard Hamiltonian Neural Network (conservative only).

Original authors: Andrew Sosanya, Sam Greydanus (2020)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class MLP(nn.Module):
    """Residual Multi-Layer Perceptron with tanh activation.

    A 3-layer MLP with a residual connection on the second hidden layer.

    Args:
        input_dim: Dimension of input features.
        output_dim: Dimension of output.
        hidden_dim: Number of hidden units per layer.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super(MLP, self).__init__()
        self.lin_1 = nn.Linear(input_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional time concatenation.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.
            t: Optional time tensor to concatenate with input.

        Returns:
            Output tensor of shape ``(batch_size, output_dim)``.
        """
        inputs = torch.cat([x, t], dim=-1) if t is not None else x
        h = self.lin_1(inputs).tanh()
        h = h + self.lin_2(h).tanh()
        return self.lin_3(h)


class DHNN(nn.Module):
    """Dissipative Hamiltonian Neural Network.

    Decomposes a vector field into two components:
        - A **conservative** (rotational) component learned via a Hamiltonian ``H``,
          whose symplectic gradient preserves energy.
        - A **dissipative** (irrotational) component learned via a Rayleigh
          dissipation function ``D``, which models energy loss.

    Args:
        input_dim: Dimension of the phase-space input ``(q, p)``.
        hidden_dim: Number of hidden units in each internal MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(DHNN, self).__init__()
        self.mlp_h = MLP(input_dim, 1, hidden_dim)  # Conservative component (Hamiltonian)
        self.mlp_d = MLP(input_dim, 1, hidden_dim)  # Dissipative component (Rayleigh)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        as_separate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the predicted time-derivative of the state.

        Args:
            x: Phase-space state ``(q, p)`` of shape ``(batch_size, 2*dof)``.
                Must have ``requires_grad=True``.
            t: Optional time tensor.
            as_separate: If ``True``, return the dissipative and conservative
                components as a tuple instead of their sum.

        Returns:
            Predicted ``dx/dt``, or a tuple ``(irr_component, rot_component)``
            if ``as_separate=True``.
        """
        inputs = torch.cat([x, t], dim=-1) if t is not None else x
        D = self.mlp_d(inputs)
        H = self.mlp_h(inputs)

        # Gradient of dissipation function (irrotational field)
        irr_component = torch.autograd.grad(D.sum(), x, create_graph=True)[0]
        # Gradient of Hamiltonian (rotational / symplectic field)
        rot_component = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        # Apply symplectic structure: swap (dH/dq, dH/dp) → (dH/dp, -dH/dq)
        dHdq, dHdp = torch.split(rot_component, rot_component.shape[-1] // 2, dim=1)
        rot_component = torch.cat([dHdp, -dHdq], dim=-1)

        if as_separate:
            return irr_component, rot_component

        return irr_component + rot_component


class HNN(nn.Module):
    """Standard Hamiltonian Neural Network (conservative systems only).

    Learns a scalar Hamiltonian function ``H(q, p)`` and returns
    the symplectic gradient as the predicted dynamics.

    Args:
        input_dim: Dimension of the phase-space input ``(q, p)``.
        hidden_dim: Number of hidden units in the internal MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(HNN, self).__init__()
        self.mlp = MLP(input_dim, 1, hidden_dim)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the symplectic gradient of the learned Hamiltonian.

        Args:
            x: Phase-space state of shape ``(batch_size, 2*dof)``.
                Must have ``requires_grad=True``.
            t: Optional time tensor.

        Returns:
            Predicted ``dx/dt`` of shape ``(batch_size, 2*dof)``.
        """
        inputs = torch.cat([x, t], dim=-1) if t is not None else x
        output = self.mlp(inputs)

        H = output[..., 0]
        H_grads = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        # Symplectic gradient: (dH/dp, -dH/dq)
        dHdq, dHdp = torch.split(H_grads, H_grads.shape[-1] // 2, dim=1)
        return torch.cat([dHdp, -dHdq], dim=-1)