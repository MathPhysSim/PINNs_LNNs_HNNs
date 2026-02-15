"""Utility functions for model integration, serialization, and visualization.

Original authors: Andrew Sosanya, Sam Greydanus (2020)
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
        fun: Optional custom right-hand-side function ``f(t, x)``.
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


def L2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the mean squared L2 loss between two tensors."""
    return (u - v).pow(2).mean()


def to_pickle(thing, path: str) -> None:
    """Serialize an object to a pickle file."""
    with open(path, "wb") as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path: str):
    """Deserialize an object from a pickle file."""
    with open(path, "rb") as handle:
        return pickle.load(handle)