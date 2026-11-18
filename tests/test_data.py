"""Unit tests for ddpm.data (CurveData, Standardizer)."""

import jax
import jax.numpy as jnp

from ddpm.data import CurveData, Standardizer


class TestCurveData:
    def test_shape_and_range(self):
        data = CurveData()
        s = data.sample(jax.random.key(0), 5000)
        assert s.shape == (5000, 2)
        assert s[:, 0].min() >= data.x_min and s[:, 0].max() <= data.x_max

    def test_noise_level(self):
        data = CurveData(noise_std=0.1)
        s = data.sample(jax.random.key(1), 20_000)
        residual = s[:, 1] - jnp.sin(s[:, 0])
        assert abs(float(residual.std()) - 0.1) < 0.005

    def test_custom_curve(self):
        data = CurveData(fn=lambda x: x**2 - 1, noise_std=0.0)
        s = data.sample(jax.random.key(2), 100)
        assert jnp.allclose(s[:, 1], s[:, 0] ** 2 - 1)


class TestStandardizer:
    def test_fit_gives_unit_scale(self):
        s = CurveData().sample(jax.random.key(3), 50_000)
        z = Standardizer.fit(s).normalize(s)
        assert jnp.allclose(z.mean(axis=0), 0.0, atol=1e-4)
        assert jnp.allclose(z.std(axis=0), 1.0, atol=1e-4)

    def test_roundtrip(self):
        s = CurveData().sample(jax.random.key(4), 100)
        st = Standardizer.fit(s)
        assert jnp.allclose(st.unnormalize(st.normalize(s)), s, atol=1e-5)

    def test_identity(self):
        s = CurveData().sample(jax.random.key(5), 10)
        st = Standardizer.identity(2)
        assert jnp.array_equal(st.normalize(s), s)
        assert jnp.array_equal(st.score_to_data_space(s), s)
