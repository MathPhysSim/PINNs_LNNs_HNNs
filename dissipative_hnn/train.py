"""Training utilities for Dissipative Hamiltonian Neural Networks.

Provides hyperparameter configuration, batching helpers, and a general-purpose
training loop for DHNN / HNN / MLP models.

Original authors: Andrew Sosanya, Sam Greydanus (2020)
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Union


class ObjectView:
    """Lightweight wrapper that exposes a dictionary as object attributes."""

    def __init__(self, d: dict):
        self.__dict__ = d


def get_args(as_dict: bool = False) -> Union[Dict[str, Any], ObjectView]:
    """Return default hyperparameters for training.

    Args:
        as_dict: If ``True``, return a plain dictionary; otherwise an ``ObjectView``.

    Returns:
        Hyperparameter configuration.
    """
    arg_dict = {
        "input_dim": 3,
        "hidden_dim": 256,
        "output_dim": 2,
        "learning_rate": 1e-2,
        "test_every": 100,
        "print_every": 200,
        "batch_size": 128,
        "train_split": 0.80,
        "total_steps": 5000,
        "device": "cuda",
        "seed": 42,
        "as_separate": False,
        "decay": 0,
    }
    return arg_dict if as_dict else ObjectView(arg_dict)


def get_batch(
    v: np.ndarray, step: int, args: ObjectView
) -> torch.Tensor:
    """Extract a mini-batch from a dataset array and move it to the target device.

    Args:
        v: Full dataset array of shape ``(N, D)``.
        step: Current training step (used for cycling through data).
        args: Hyperparameters (must contain ``batch_size`` and ``device``).

    Returns:
        Batch tensor on the configured device with gradients enabled.
    """
    dataset_size, _ = v.shape
    bix = (step * args.batch_size) % dataset_size
    v_batch = v[bix : bix + args.batch_size, :]
    return torch.tensor(v_batch, requires_grad=True, dtype=torch.float32, device=args.device)


def train(
    model: torch.nn.Module, args: ObjectView, data: Dict[str, np.ndarray]
) -> Dict[str, List[float]]:
    """Train a model with L2 loss and Adam optimizer.

    Args:
        model: A DHNN, HNN, or MLP model.
        args: Hyperparameters.
        data: Dataset dictionary with keys ``'x'``, ``'t'``, ``'dx'``,
              ``'x_test'``, ``'t_test'``, ``'dx_test'``.

    Returns:
        Dictionary of training and test losses per step.
    """
    model = model.to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.decay
    )

    model.train()
    t0 = time.time()
    results: Dict[str, List[float]] = {
        "train_loss": [], "test_loss": [], "test_acc": [], "global_step": 0
    }

    for step in range(args.total_steps):
        x, t, dx = [get_batch(data[k], step, args) for k in ["x", "t", "dx"]]
        dx_hat = model(x, t=t)
        loss = (dx - dx_hat).pow(2).mean()

        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()

        results["train_loss"].append(loss.item())

        if step % args.test_every == 0:
            x_test, t_test, dx_test = [
                get_batch(data[k], step=0, args=args) for k in ["x_test", "t_test", "dx_test"]
            ]
            dx_hat_test = model(x_test, t=t_test)
            test_loss = (dx_test - dx_hat_test).pow(2).mean().item()
            results["test_loss"].append(test_loss)

        if step % args.print_every == 0:
            print(
                f"step {step}, dt {time.time() - t0:.3f}, "
                f"train_loss {loss.item():.2e}, test_loss {test_loss:.2e}"
            )
            t0 = time.time()

    model = model.cpu()
    return results