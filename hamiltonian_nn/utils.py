"""Utility functions for Hamiltonian Neural Networks.

Provides numerical integration, activation function selection,
serialization helpers, and the RK4 integrator.

Original authors: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
"""

import numpy as np
import os
import pickle
import torch
from typing import Optional, Callable


def integrate_model(
    model, t_span: tuple, y0: np.ndarray, fun: Optional[Callable] = None, **kwargs
):
    """Integrate a learned model forward in time using ``scipy.integrate.solve_ivp``.

    Args:
        model: A model with a ``time_derivative`` method.
        t_span: Integration interval ``(t0, tf)``.
        y0: Initial state vector.
        fun: Optional custom right-hand-side function.
        **kwargs: Additional arguments passed to ``solve_ivp``.

    Returns:
        ``scipy.integrate.OdeResult`` solution object.
    """
    from scipy.integrate import solve_ivp

    def default_fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
        x = x.view(1, np.size(np_x))
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    fun = default_fun if fun is None else fun
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def rk4(fun: Callable, y0: torch.Tensor, t: float, dt: float, *args, **kwargs) -> torch.Tensor:
    """Fourth-order Runge–Kutta integration step.

    Args:
        fun: Right-hand-side function ``f(y, t)``.
        y0: Initial state.
        t: Current time.
        dt: Time step.

    Returns:
        State increment ``dy``.
    """
    dt2 = dt / 2.0
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
    k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
    k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
    return dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def L2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the mean squared L2 loss between two tensors."""
    return (u - v).pow(2).mean()


def choose_nonlinearity(name: str) -> Callable:
    """Map an activation function name to its PyTorch implementation.

    Supported: ``'tanh'``, ``'relu'``, ``'sigmoid'``, ``'softplus'``,
    ``'selu'``, ``'elu'``, ``'swish'``.

    Args:
        name: Name of the activation function.

    Returns:
        The corresponding callable.

    Raises:
        ValueError: If the name is not recognized.
    """
    activations = {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "softplus": torch.nn.functional.softplus,
        "selu": torch.nn.functional.selu,
        "elu": torch.nn.functional.elu,
        "swish": lambda x: x * torch.sigmoid(x),
    }
    if name not in activations:
        raise ValueError(f"Nonlinearity '{name}' not recognized. Choose from: {list(activations.keys())}")
    return activations[name]


def to_pickle(thing, path: str) -> None:
    """Serialize an object to a pickle file."""
    with open(path, "wb") as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path: str):
    """Deserialize an object from a pickle file."""
    with open(path, "rb") as handle:
        return pickle.load(handle)