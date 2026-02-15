"""Dissipative Hamiltonian Neural Networks (D-HNNs).

A PyTorch implementation of Dissipative Hamiltonian Neural Networks, which extend
standard HNNs with a Rayleigh dissipation function to model non-conservative systems.

Based on:
    Sosanya, A. & Greydanus, S. (2022). Dissipative Hamiltonian Neural Networks.
    arXiv:2201.10085. https://arxiv.org/abs/2201.10085

Original authors: Andrew Sosanya, Sam Greydanus (2020)
"""

from .data import get_spiral_data
from .utils import integrate_model, L2_loss, to_pickle, from_pickle
from .train import get_args, train
from .models import MLP, DHNN, HNN
from .numeric import approx_helmholtz_decomp, coords2fields