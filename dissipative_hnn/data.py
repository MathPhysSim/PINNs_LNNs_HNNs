"""Synthetic data generation for Dissipative HNN experiments.

Generates a 2D spiral vector field composed of rotational (conservative)
and irrotational (dissipative) components for training and evaluation.

Original authors: Andrew Sosanya, Sam Greydanus (2020)
"""

import numpy as np
from typing import Dict


def get_spiral_data(args) -> Dict[str, np.ndarray]:
    """Generate a synthetic spiral vector-field dataset.

    Creates a composite field from a rotational component ``(-y, x)``
    and an irrotational component ``(x, y)``, then splits into
    train/test sets.

    Args:
        args: Configuration object with attribute ``train_split`` (float).

    Returns:
        Dictionary with keys ``'x'``, ``'x_test'``, ``'y'``, ``'y_test'``,
        ``'y_rot'``, ``'y_rot_test'``, ``'y_irr'``, ``'y_irr_test'``.
    """
    x, y = np.meshgrid(np.arange(-2, 2, 0.2), np.arange(-2, 2, 0.25))

    # Rotational (conservative) and irrotational (dissipative) components
    u_rot, v_rot = -y, x
    u_irr, v_irr = x, y
    u = u_rot + u_irr
    v = v_rot + v_irr

    # Reshape to (N, 2) convention
    x = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    y_rot = np.concatenate([u_rot.reshape(-1, 1), v_rot.reshape(-1, 1)], axis=1)
    y_irr = np.concatenate([u_irr.reshape(-1, 1), v_irr.reshape(-1, 1)], axis=1)
    y = y_rot + y_irr

    # Shuffle
    shuffle_ixs = np.random.permutation(x.shape[0])
    x, y_rot, y_irr, y = x[shuffle_ixs], y_rot[shuffle_ixs], y_irr[shuffle_ixs], y[shuffle_ixs]

    # Train / test split
    split_ix = int(x.shape[0] * args.train_split)
    data = {
        "x": x[:split_ix], "x_test": x[split_ix:],
        "y_rot": y_rot[:split_ix], "y_rot_test": y_rot[split_ix:],
        "y_irr": y_irr[:split_ix], "y_irr_test": y_irr[split_ix:],
        "y": y[:split_ix], "y_test": y[split_ix:],
    }
    return data