"""Unit tests for ddpm.net (Denoiser) and ddpm.loss (weighted_x0_loss)."""

import jax
import jax.numpy as jnp
import optax
import pytest

from ddpm.loss import weighted_x0_loss
from ddpm.net import Denoiser
from ddpm.sde import ou_coefficients


@pytest.fixture
def net_and_params():
    net = Denoiser(hidden_dims=(32, 32))
    params = net.init(jax.random.key(0), jnp.zeros((4, 1)), jnp.zeros((4, 2)))
    return net, params


class TestDenoiser:
    def test_shape_and_t_layouts(self, net_and_params):
        net, params = net_and_params
        x = jax.random.normal(jax.random.key(1), (8, 2))
        t = jnp.linspace(0.1, 3.0, 8)
        out_flat = net.apply(params, t, x)
        out_col = net.apply(params, t[:, None], x)
        assert out_flat.shape == (8, 2)
        assert jnp.allclose(out_flat, out_col)

    def test_zero_noise_prediction(self, net_and_params):
        """With ε̂ = 0 (zeroed last layer), x̂_0 = x / α_t."""
        net, params = net_and_params
        params = jax.tree_util.tree_map(lambda p: p, params)
        last = params["params"]["last_layer"]
        params["params"]["last_layer"] = jax.tree_util.tree_map(jnp.zeros_like, last)
        x = jax.random.normal(jax.random.key(2), (5, 2))
        t = jnp.full(5, 1.3)
        alpha, _ = ou_coefficients(t[:, None])
        assert jnp.allclose(net.apply(params, t, x), x / alpha, atol=1e-6)


class TestWeightedX0Loss:
    def test_equals_noise_prediction_loss(self, net_and_params):
        """(α/β)² ‖x̂_0 - x_0‖² = ‖ε̂ - ε‖², with ε̂ recovered from x̂_0."""
        net, params = net_and_params
        k0, k1, k2 = jax.random.split(jax.random.key(3), 3)
        x0 = jax.random.normal(k0, (64, 2))
        t = jax.random.uniform(k1, (64,), minval=0.05, maxval=4.0)
        eps = jax.random.normal(k2, x0.shape)
        alpha, beta = ou_coefficients(t[:, None])
        xt = alpha * x0 + beta * eps

        x0_hat = net.apply(params, t, xt)
        eps_hat = (xt - alpha * x0_hat) / beta
        expected = jnp.mean((eps_hat - eps) ** 2)
        assert jnp.allclose(weighted_x0_loss(params, net.apply, t, x0, xt), expected, rtol=1e-4)

    def test_decreases_after_steps(self, net_and_params):
        net, params = net_and_params
        k0, k1, k2 = jax.random.split(jax.random.key(4), 3)
        x0 = jax.random.normal(k0, (256, 2)) * 0.3 + 1.0
        t = jax.random.uniform(k1, (256,), minval=0.05, maxval=4.0)
        alpha, beta = ou_coefficients(t[:, None])
        xt = alpha * x0 + beta * jax.random.normal(k2, x0.shape)

        opt = optax.adam(1e-2)
        state = opt.init(params)
        loss0 = weighted_x0_loss(params, net.apply, t, x0, xt)
        for _ in range(50):
            grads = jax.grad(weighted_x0_loss)(params, net.apply, t, x0, xt)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
        assert weighted_x0_loss(params, net.apply, t, x0, xt) < loss0
