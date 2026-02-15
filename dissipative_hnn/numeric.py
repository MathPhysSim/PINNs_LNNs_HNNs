"""Numerical Helmholtz–Hodge decomposition utilities.

Provides functions to decompose a 2D vector field into its rotational
(divergence-free) and irrotational (curl-free) components using a
Gauss–Seidel Poisson solver.

Original authors: Andrew Sosanya, Sam Greydanus (2020)
"""

import numpy as np
from functools import partial
from typing import Optional, Tuple

RHO = 0.75
GRIDSIZE = 20


def get_interp_model(x: np.ndarray, dx: np.ndarray, method: str = "nearest"):
    """Create a partial interpolation function using ``scipy.interpolate.griddata``."""
    from scipy import interpolate
    return partial(interpolate.griddata, x, dx, method=method)


def coords2fields(
    x: np.ndarray,
    dx: np.ndarray,
    hw: Optional[Tuple[int, int]] = None,
    replace_nans: bool = True,
    method: str = "nearest",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate scattered coordinate data onto a regular grid.

    Args:
        x: Coordinates of shape ``(N, 2)``.
        dx: Vector field values of shape ``(N, 2)``.
        hw: Grid dimensions ``(height, width)``; defaults to ``(GRIDSIZE, GRIDSIZE)``.
        replace_nans: Replace NaN values with the field mean.
        method: Interpolation method (``'nearest'``, ``'linear'``, ``'cubic'``).
        verbose: Print grid size information.

    Returns:
        Tuple of ``(x_field, dx_field)`` on the regular grid.
    """
    if hw is None:
        h = w = GRIDSIZE
        if verbose:
            print(f"Using gridsize={GRIDSIZE}")
    else:
        h, w = hw

    xx = np.linspace(x[:, 0].min(), x[:, 1].max(), w)
    yy = np.linspace(x[:, 0].min(), x[:, 1].max(), h)
    x_field = np.stack(np.meshgrid(xx, yy), axis=-1)

    interp_model = get_interp_model(x, dx, method=method)
    dx_field = interp_model(x_field)

    if replace_nans:
        dx_field[np.where(np.isnan(dx_field))] = np.nanmean(dx_field)

    return x_field, dx_field


def project(vx: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project a velocity field to be approximately divergence-free.

    Uses 1000 iterations of Gauss–Seidel to solve the Poisson equation
    for the pressure field, then subtracts its gradient.

    Args:
        vx: x-component of velocity on a regular grid.
        vy: y-component of velocity on a regular grid.

    Returns:
        Projected ``(vx, vy)`` that is approximately divergence-free.
    """
    p = np.zeros(vx.shape)
    div = -0.5 * (
        np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)
        + np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)
    )

    for _ in range(1000):
        p = (
            div
            + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1)
            + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)
        ) / 4.0

    vx = vx - 0.5 * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))
    vy = vy - 0.5 * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))
    return vx, vy


def approx_helmholtz_decomp(x: np.ndarray, dx: np.ndarray, **kwargs):
    """Compute an approximate Helmholtz–Hodge decomposition of a 2D vector field.

    Args:
        x: Coordinates of shape ``(N, 2)``.
        dx: Vector field values of shape ``(N, 2)``.
        **kwargs: Passed to ``coords2fields``.

    Returns:
        Tuple of ``(x_field, dx_field, dx_rot, dx_irr)`` where
        ``dx_rot`` is the divergence-free component and ``dx_irr``
        is the curl-free component.
    """
    x_field, dx_field = coords2fields(x, dx, **kwargs)
    dx0, dx1 = dx_field[..., 0], dx_field[..., 1]
    dx_rot = np.stack(project(dx0, dx1), axis=-1)
    dx_irr = dx_field - dx_rot
    return x_field, dx_field, dx_rot, dx_irr