"""
Training loss.

    L = E[(α_t / β_t)² ‖x̂_0 - x_0‖²]

Since x̂_0 - x_0 = (β_t / α_t)(ε - ε̂), this equals the standard
noise-prediction loss E‖ε̂ - ε‖² (averaged over coordinates).
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax import Array

from ddpm.sde import ou_coefficients

__all__ = ["weighted_x0_loss"]


def weighted_x0_loss(params: Any, apply_fn: Callable, t: Array, x0: Array,
                     xt: Array) -> Array:
    """Mean of (α_t/β_t)² (x̂_0 - x_0)² over the batch (B,) and coordinates (d,)."""
    x0_hat = apply_fn(params, t, xt)
    alpha, beta = ou_coefficients(t)
    weight = (alpha / beta) ** 2
    return jnp.mean(weight[:, None] * (x0_hat - x0) ** 2)
