"""Neural network building blocks for Hamiltonian Neural Networks.

Provides a multi-layer perceptron (MLP) and an MLP-based autoencoder
with residual connections, used as the differentiable models inside HNNs.

Original authors: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
"""

import torch
import torch.nn as nn
from typing import Callable

from .utils import choose_nonlinearity


class MLP(nn.Module):
    """Three-layer MLP with configurable activation.

    Uses orthogonal weight initialization for improved training stability.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer width.
        output_dim: Output dimension.
        nonlinearity: Activation function name (``'tanh'``, ``'relu'``, etc.).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 nonlinearity: str = "tanh"):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=False)

        for layer in [self.linear1, self.linear2, self.linear3]:
            nn.init.orthogonal_(layer.weight)

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x: torch.Tensor, separate_fields: bool = False) -> torch.Tensor:
        """Forward pass through the MLP."""
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)


class MLPAutoencoder(nn.Module):
    """MLP Autoencoder with residual connections.

    An 8-layer autoencoder (4 encoder + 4 decoder) with skip connections
    in the hidden layers for improved gradient flow.

    Args:
        input_dim: Input/output dimension (e.g., number of pixels).
        hidden_dim: Hidden layer width.
        latent_dim: Bottleneck dimension.
        nonlinearity: Activation function name.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 nonlinearity: str = "tanh"):
        super(MLPAutoencoder, self).__init__()
        # Encoder
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.linear5 = nn.Linear(latent_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, input_dim)

        for layer in [self.linear1, self.linear2, self.linear3, self.linear4,
                      self.linear5, self.linear6, self.linear7, self.linear8]:
            nn.init.orthogonal_(layer.weight)

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        h = self.nonlinearity(self.linear1(x))
        h = h + self.nonlinearity(self.linear2(h))
        h = h + self.nonlinearity(self.linear3(h))
        return self.linear4(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to input space."""
        h = self.nonlinearity(self.linear5(z))
        h = h + self.nonlinearity(self.linear6(h))
        h = h + self.nonlinearity(self.linear7(h))
        return self.linear8(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        return self.decode(self.encode(x))