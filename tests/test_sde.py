"""
Unit tests for ddpm.sde.

  - ou_coefficients: boundary values, α² + β² = 1, small-t accuracy
  - ou_forward_sample(_batch): shape, empirical moments, per-row keys
  - score_from_x0_hat: exact for Gaussian data
"""

import jax
import jax.numpy as jnp
import numpy as np

from ddpm.sde import (
    ou_coefficients,
    ou_forward_sample,
    ou_forward_sample_batch,
    score_from_x0_hat,
)


class TestOUCoefficients:
    def test_at_t0(self):
        alpha, beta = ou_coefficients(0.0)
        assert jnp.allclose(alpha, 1.0) and jnp.allclose(beta, 0.0)

    def test_large_t(self):
        alpha, beta = ou_coefficients(40.0)
        assert jnp.allclose(alpha, 0.0, atol=1e-6) and jnp.allclose(beta, 1.0)

    def test_variance_preserving(self):
        alpha, beta = ou_coefficients(jnp.linspace(0.0, 10.0, 50))
        assert jnp.allclose(alpha**2 + beta**2, 1.0, atol=1e-6)

    def test_beta_accurate_for_small_t(self):
        """float32 β must match the float64 value of sqrt(1 - e^{-t}) for tiny t."""
        t = np.array([1e-7, 1e-5, 1e-3], dtype=np.float32)
        _, beta = ou_coefficients(jnp.asarray(t))
        exact = np.sqrt(-np.expm1(-t.astype(np.float64)))
        np.testing.assert_allclose(np.asarray(beta), exact, rtol=1e-5)


class TestOUForwardSample:
    def test_shape(self):
        assert ou_forward_sample(jax.random.key(0), jnp.zeros(2), 1.0).shape == (2,)

    def test_empirical_moments(self):
        x0 = jnp.array([2.0, -1.0])
        t = 1.0
        n = 20_000
        xt = ou_forward_sample_batch(jax.random.key(1), jnp.tile(x0, (n, 1)), jnp.full(n, t))
        alpha, beta = ou_coefficients(t)
        assert jnp.allclose(xt.mean(axis=0), alpha * x0, atol=0.03)
        assert jnp.allclose(xt.std(axis=0), beta, atol=0.03)

    def test_batch_uses_one_key_per_row(self):
        key = jax.random.key(2)
        x0 = jnp.arange(6.0).reshape(3, 2)
        t = jnp.array([0.1, 0.5, 2.0])
        batch = ou_forward_sample_batch(key, x0, t)
        keys = jax.random.split(key, 3)
        rows = jnp.stack([ou_forward_sample(k, x, s) for k, x, s in zip(keys, x0, t)])
        assert jnp.allclose(batch, rows, atol=1e-6)


class TestScoreFromX0Hat:
    def test_standard_gaussian_data(self):
        """Data N(0, I) stays N(0, I) under the OU process: posterior mean α x, score -x."""
        x = jnp.array([[0.3, -1.2], [2.0, 0.5]])
        for t in (0.05, 1.0, 3.0):
            alpha, _ = ou_coefficients(t)
            assert jnp.allclose(score_from_x0_hat(x, t, alpha * x), -x, atol=1e-4)
