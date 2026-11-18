"""
Unit tests for ddpm.sampler.reverse_sde.

  - matches a plain Python-loop Euler–Maruyama (same keys, same steps)
  - with the exact denoiser for Gaussian data, recovers that Gaussian
"""

import jax
import jax.numpy as jnp
import pytest

from ddpm.sampler import reverse_sde
from ddpm.sde import ou_coefficients, score_from_x0_hat
from ddpm.times import exponential_linspace


def gaussian_denoiser(params, t, x):
    """Exact E[x_0 | x_t] for data x_0 ~ N(m, s² I); params = (m, s)."""
    m, s = params
    alpha, beta = ou_coefficients(t[:, None])
    gain = alpha * s**2 / (alpha**2 * s**2 + beta**2)
    return m + gain * (x - alpha * m)


def reverse_sde_loop(key, params, apply_fn, x, t_grid):
    """Reference implementation: the original notebook's Python loop."""
    t_rev = list(reversed(t_grid))
    for i in range(len(t_grid) - 1):
        t, s = t_rev[i], t_rev[i + 1]
        delta = t - s
        x0_hat = apply_fn(params, jnp.repeat(t, len(x)), x)
        drift = 0.5 * x + score_from_x0_hat(x, t, x0_hat)
        key, subkey = jax.random.split(key)
        x = x + delta * drift + jnp.sqrt(delta) * jax.random.normal(subkey, x.shape)
    return x


def test_matches_python_loop():
    params = (jnp.array([0.5, -1.0]), 0.7)
    grid = exponential_linspace(0.1, 4.0, 30)
    x_T = jax.random.normal(jax.random.key(0), (256, 2))
    key = jax.random.key(1)
    scan = reverse_sde(key, params, gaussian_denoiser, x_T, grid)
    loop = reverse_sde_loop(key, params, gaussian_denoiser, x_T, grid)
    assert jnp.allclose(scan, loop, atol=1e-4)


@pytest.mark.parametrize("m,s", [(0.0, 1.0), (0.5, 0.5)])
def test_recovers_gaussian(m, s):
    params = (jnp.array([m, -m]), s)
    grid = exponential_linspace(0.1, 10.0, 400)   # T = 10: start mismatch is negligible
    x_T = jax.random.normal(jax.random.key(2), (20_000, 2))
    x = reverse_sde(jax.random.key(3), params, gaussian_denoiser, x_T, grid)
    assert jnp.allclose(x.mean(axis=0), jnp.array([m, -m]), atol=0.03)
    assert jnp.allclose(x.std(axis=0), s, atol=0.03)
