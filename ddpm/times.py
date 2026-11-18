"""
Time grids and training-time samplers.

Small times matter most (fine structure of the data lives there), so both the
sampling grid and the optional training-time sampler are biased toward t = 0
through a log(t + eps) change of variable. `eps` sets how strongly: smaller
`eps` packs more points near 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "exponential_linspace",
    "exponential_uniform",
    "sample_times_uniform",
    "sample_times_exponential",
]


def exponential_linspace(eps: float, T: float, size: int) -> Array:
    """Deterministic grid on [0, T], evenly spaced in log(t + eps), increasing.

    t_k = exp(linspace(log eps, log(T + eps), size)) - eps, so t_0 = 0 and
    t_{size-1} = T. Used to discretize the reverse SDE.
    """
    u = jnp.linspace(jnp.log(eps), jnp.log(T + eps), size)
    return jnp.exp(u) - eps


def exponential_uniform(key: Array, T: float, size: int, eps: float) -> Array:
    """Random times in [0, T], uniform in log(t + eps) (biased toward small t)."""
    u = jax.random.uniform(key, (size,), minval=jnp.log(eps), maxval=jnp.log(T + eps))
    return jnp.exp(u) - eps


def sample_times_uniform(key: Array, T: float, size: int, t_min: float) -> Array:
    """t ~ Uniform(t_min, T). `t_min > 0` keeps β_t > 0 (the loss divides by it)."""
    return jax.random.uniform(key, (size,), minval=t_min, maxval=T)


def sample_times_exponential(key: Array, T: float, size: int, t_min: float,
                             eps: float) -> Array:
    """`exponential_uniform`, clipped to [t_min, T]."""
    return jnp.clip(exponential_uniform(key, T, size, eps), t_min, T)
