"""
Denoising network.

An MLP on (t, x_t) predicts the noise ε̂ and returns the denoised estimate

    x̂_0 = (x_t - β_t ε̂) / α_t

Predicting the noise trains much better than predicting x_0 directly. The
network is batched: t has shape (B,) or (B, 1), x has shape (B, d).
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from ddpm.sde import ou_coefficients

__all__ = ["Denoiser"]


class Denoiser(nn.Module):
    hidden_dims: tuple[int, ...] = (256, 256)
    output_dim: int = 2

    def setup(self):
        self.layers = [nn.Dense(h) for h in self.hidden_dims]
        self.last_layer = nn.Dense(self.output_dim)

    def __call__(self, t: Array, x: Array) -> Array:
        t = t[:, None] if t.ndim == 1 else t   # (B, 1)
        h = jnp.concatenate([t, x], axis=1)    # (B, 1 + d)

        # estimate the noise
        for layer in self.layers:
            h = nn.elu(layer(h))
        noise_hat = self.last_layer(h)

        # convert to an estimate of x0
        alpha, beta = ou_coefficients(t)
        return (x - beta * noise_hat) / alpha
