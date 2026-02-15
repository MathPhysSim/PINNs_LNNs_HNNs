"""Hamiltonian Neural Networks (HNNs).

A PyTorch implementation of Hamiltonian Neural Networks that learn
conservation laws from data by parameterizing the Hamiltonian function.

Based on:
    Greydanus, S., Dzamba, M. & Yosinski, J. (2019). Hamiltonian Neural Networks.
    NeurIPS 2019. https://arxiv.org/abs/1906.01563

Original authors: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
"""

from .hnn import HNN, PixelHNN
from .nn_models import MLP, MLPAutoencoder
from .utils import integrate_model, L2_loss, rk4, choose_nonlinearity
