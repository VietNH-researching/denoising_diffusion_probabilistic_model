"""
Training data: noisy points on a curve, and per-coordinate standardization.

`CurveData` draws (x, f(x) + noise) with x ~ Uniform(x_min, x_max) and
noise ~ N(0, noise_std²). The default curve is f = sin.

`Standardizer` maps data to mean 0, std 1 per coordinate and back. The identity
standardizer (`Standardizer.identity`) leaves data unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["CurveData", "Standardizer"]


@dataclass(frozen=True)
class CurveData:
    """Points (x, fn(x) + noise), x ~ Uniform(x_min, x_max), noise ~ N(0, noise_std²)."""
    fn: Callable[[Array], Array] = jnp.sin
    x_min: float = -10.0
    x_max: float = 10.0
    noise_std: float = 0.05

    def sample(self, key: Array, n: int) -> Array:
        """Draw `n` points, shape (n, 2)."""
        key_x, key_noise = jax.random.split(key)
        x = jax.random.uniform(key_x, (n,), minval=self.x_min, maxval=self.x_max)
        y = self.fn(x) + self.noise_std * jax.random.normal(key_noise, (n,))
        return jnp.c_[x, y]


@dataclass(frozen=True)
class Standardizer:
    """Per-coordinate affine map z = (x - mean) / std and its inverse."""
    mean: Array
    std: Array

    @classmethod
    def identity(cls, dim: int) -> "Standardizer":
        return cls(jnp.zeros(dim), jnp.ones(dim))

    @classmethod
    def fit(cls, samples: Array) -> "Standardizer":
        """Mean and std per coordinate of `samples`, shape (n, dim)."""
        return cls(samples.mean(axis=0), samples.std(axis=0))

    def normalize(self, x: Array) -> Array:
        return (x - self.mean) / self.std

    def unnormalize(self, z: Array) -> Array:
        return self.mean + self.std * z

    def score_to_data_space(self, score_z: Array) -> Array:
        """Chain rule: if x = mean + std · z, then ∇_x log p(x) = ∇_z log p(z) / std."""
        return score_z / self.std
