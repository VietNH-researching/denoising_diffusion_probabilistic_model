"""
Reverse-SDE sampler.

The time reversal of the OU process dX = -½ X dt + dW is

    dY = [½ Y + ∇log p_t(Y)] dt + dW,   run from t = T down to 0, Y_T ~ N(0, I)

The score comes from the denoiser through `score_from_x0_hat`. Euler–Maruyama
on a grid t_0 = 0 < t_1 < … < t_N = T, stepping from t = t_n to s = t_{n-1}:

    y ← y + Δ (½ y + score(y, t)) + √Δ ε,   Δ = t - s > 0

The network is evaluated at t_N … t_1 only, never at t = 0 (β_0 = 0).
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from ddpm.sde import score_from_x0_hat

__all__ = ["reverse_sde"]


@partial(jax.jit, static_argnames="apply_fn")
def reverse_sde(key: Array, params: Any, apply_fn: Callable, x_T: Array,
                t_grid: Array) -> Array:
    """Integrate the reverse SDE from t_grid[-1] down to t_grid[0].

    Parameters
    ----------
    key : PRNG key; split once per step.
    params, apply_fn : the denoiser, called as apply_fn(params, t (B,), x (B, d)) → x̂_0.
    x_T : (B, d) starting points, usually N(0, I) draws.
    t_grid : (N + 1,) increasing times with t_grid[0] = 0.

    Returns
    -------
    (B, d) samples at t = t_grid[0].
    """
    t_rev = t_grid[::-1]

    def step(carry, ts):
        x, key = carry
        t, s = ts
        delta = t - s
        x0_hat = apply_fn(params, jnp.full(x.shape[0], t), x)
        drift = 0.5 * x + score_from_x0_hat(x, t, x0_hat)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, x.shape)
        x = x + delta * drift + jnp.sqrt(delta) * noise
        return (x, key), None

    (x, _), _ = jax.lax.scan(step, (x_T, key), (t_rev[:-1], t_rev[1:]))
    return x
