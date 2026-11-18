"""Unit tests for ddpm.times (time grids and training-time samplers)."""

import jax
import jax.numpy as jnp
import pytest

from ddpm.times import (
    exponential_linspace,
    exponential_uniform,
    sample_times_exponential,
    sample_times_uniform,
)


class TestExponentialLinspace:
    def test_endpoints_and_size(self):
        grid = exponential_linspace(0.1, 4.0, 200)
        assert grid.shape == (200,)
        assert jnp.allclose(grid[0], 0.0, atol=1e-6)
        assert jnp.allclose(grid[-1], 4.0, atol=1e-5)

    def test_increasing_steps(self):
        """Strictly increasing, with steps growing geometrically away from t = 0."""
        steps = jnp.diff(exponential_linspace(0.1, 4.0, 200))
        assert jnp.all(steps > 0)
        assert jnp.all(jnp.diff(steps) > -1e-6)
        assert steps[-1] / steps[0] == pytest.approx(41.0, rel=0.02)   # (T + eps) / eps


class TestSamplers:
    def test_exponential_uniform_range_and_bias(self):
        t = exponential_uniform(jax.random.key(0), 4.0, 20_000, 0.1)
        assert jnp.all(t >= -1e-6) and jnp.all(t <= 4.0 + 1e-5)
        # P(t <= 0.1) = log(2) / log(41) ≈ 0.187; uniform would give 0.025
        assert float(jnp.mean(t <= 0.1)) == pytest.approx(0.187, abs=0.01)

    def test_uniform_respects_t_min(self):
        t = sample_times_uniform(jax.random.key(1), 4.0, 10_000, 1e-4)
        assert jnp.all(t >= 1e-4) and jnp.all(t <= 4.0)

    def test_exponential_respects_t_min(self):
        t = sample_times_exponential(jax.random.key(2), 4.0, 10_000, 1e-4, 0.1)
        assert jnp.all(t >= 1e-4) and jnp.all(t <= 4.0)
