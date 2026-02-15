"""Hamiltonian Neural Network models.

Implements the core HNN architecture that learns a scalar Hamiltonian
and uses its symplectic gradient to predict dynamics. Also includes
a pixel-space variant with an autoencoder.

Original authors: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
"""

import torch
import numpy as np
from typing import Optional, List, Union, Tuple

from .nn_models import MLP
from .utils import rk4


class HNN(torch.nn.Module):
    """Hamiltonian Neural Network.

    Learns arbitrary vector fields that are sums of conservative
    and solenoidal (divergence-free) fields.

    Args:
        input_dim: Dimension of the phase-space state.
        differentiable_model: A differentiable model that outputs scalar fields.
        field_type: One of ``'solenoidal'``, ``'conservative'``, or ``'both'``.
        baseline: If ``True``, use the model as a plain neural ODE (no Hamiltonian structure).
        assume_canonical_coords: If ``True``, use the standard symplectic matrix.
    """

    def __init__(
        self,
        input_dim: int,
        differentiable_model: torch.nn.Module,
        field_type: str = "solenoidal",
        baseline: bool = False,
        assume_canonical_coords: bool = True,
    ):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self._permutation_tensor(input_dim)
        self.field_type = field_type

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass.

        Args:
            x: Input state of shape ``(batch_size, input_dim)``.

        Returns:
            If ``baseline``, returns model output directly.
            Otherwise, returns a tuple of two scalar fields ``(F1, F2)``.
        """
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, \
            "Output tensor should have shape [batch_size, 2]"
        return y.split(1, 1)

    def rk4_time_derivative(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute a single RK4 step of the time derivative."""
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(
        self, x: torch.Tensor, t=None, separate_fields: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute the learned vector field (time derivative).

        For ``baseline=True``, this is a plain neural ODE.
        Otherwise, it computes the Hamiltonian-style vector field
        using conservative and/or solenoidal components.

        Args:
            x: State tensor. Must have ``requires_grad=True``.
            t: Unused (for API compatibility).
            separate_fields: If ``True``, return components separately.

        Returns:
            Predicted ``dx/dt``, or a list ``[conservative, solenoidal]``.
        """
        if self.baseline:
            return self.differentiable_model(x)

        F1, F2 = self.forward(x)

        conservative_field = torch.zeros_like(x)
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != "solenoidal":
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != "conservative":
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def _permutation_tensor(self, n: int) -> torch.Tensor:
        """Construct the symplectic / Levi-Civita permutation tensor.

        For canonical coordinates, this is the standard symplectic matrix ``J``.
        """
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n // 2:], -M[:n // 2]])
        else:
            M = torch.ones(n, n)
            M *= 1 - torch.eye(n)
            M[::2] *= -1
            M[:, ::2] *= -1
            for i in range(n):
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M


class PixelHNN(torch.nn.Module):
    """Pixel-space Hamiltonian Neural Network.

    Combines an autoencoder with an HNN: encodes pixel observations
    into a latent space, applies Hamiltonian dynamics, and decodes back.

    Args:
        input_dim: Latent space dimension output by the encoder.
        hidden_dim: Hidden units in the HNN MLP.
        autoencoder: An encoder-decoder model with ``encode``/``decode`` methods.
        field_type: Type of vector field (``'solenoidal'``, ``'conservative'``, ``'both'``).
        nonlinearity: Activation function name.
        baseline: If ``True``, skip Hamiltonian structure.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        autoencoder: torch.nn.Module,
        field_type: str = "solenoidal",
        nonlinearity: str = "tanh",
        baseline: bool = False,
    ):
        super(PixelHNN, self).__init__()
        self.autoencoder = autoencoder
        self.baseline = baseline

        output_dim = input_dim if baseline else 2
        nn_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity)
        self.hnn = HNN(input_dim, differentiable_model=nn_model,
                       field_type=field_type, baseline=baseline)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel observations to latent space."""
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to pixel space."""
        return self.autoencoder.decode(z)

    def time_derivative(self, z: torch.Tensor, separate_fields: bool = False):
        """Compute time derivative in latent space."""
        return self.hnn.time_derivative(z, separate_fields)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, step forward in time, and decode."""
        z = self.encode(x)
        z_next = z + self.time_derivative(z)
        return self.decode(z_next)
