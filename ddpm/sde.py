"""
Ornstein–Uhlenbeck forward (noising) process.

The SDE is:
    dX_t = -½ X_t dt + dW_t,   X_0 ~ data

Closed-form marginal:
    X_t | X_0 ~ N(α_t X_0, β_t² I),   α_t = e^{-t/2},   β_t² = 1 - α_t² = 1 - e^{-t}

For large t the marginal is ≈ N(0, I) whatever the data distribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "ou_coefficients",
    "ou_forward_sample",
    "ou_forward_sample_batch",
    "score_from_x0_hat",
]


def ou_coefficients(t: Array) -> tuple[Array, Array]:
    """α_t = e^{-t/2} and β_t = sqrt(1 - α_t²).

    β_t uses `expm1`, which is the same formula but stays accurate for small t.
    """
    alpha = jnp.exp(-0.5 * t)
    beta = jnp.sqrt(-jnp.expm1(-t))
    return alpha, beta


def ou_forward_sample(key: Array, x0: Array, t: Array) -> Array:
    """Draw x_t ~ N(α_t x_0, β_t² I) for one point x0 of shape (d,)."""
    alpha, beta = ou_coefficients(t)
    return alpha * x0 + beta * jax.random.normal(key, x0.shape)


@jax.jit
def ou_forward_sample_batch(key: Array, x0: Array, t: Array) -> Array:
    """`ou_forward_sample` for a batch: x0 (B, d), t (B,); one key per row."""
    keys = jax.random.split(key, x0.shape[0])
    return jax.vmap(ou_forward_sample)(keys, x0, t)


def score_from_x0_hat(x: Array, t: Array, x0_hat: Array) -> Array:
    """Score of the noised marginal from a denoised estimate (Tweedie):

        ∇_x log p_t(x) ≈ -(x - α_t x̂_0) / β_t²
    """
    alpha, beta = ou_coefficients(t)
    return -(x - alpha * x0_hat) / beta**2
