# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: gpu_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Diffusion model on a noisy sine curve
#
# Learn to generate 2-D points $(x, \sin x + \text{noise})$ with a score-based
# diffusion model:
#
# 1. **Forward process**: an Ornstein–Uhlenbeck (OU) SDE
#    $dX_t = -\tfrac12 X_t\,dt + dW_t$, whose marginal is
#    $X_t = \alpha_t X_0 + \beta_t \varepsilon$ with $\alpha_t = e^{-t/2}$,
#    $\beta_t = \sqrt{1-\alpha_t^2}$, $\varepsilon \sim \mathcal N(0, I)$.
# 2. **Training**: a neural net predicts the noise $\hat\varepsilon$ and returns
#    the denoised estimate $\hat x_0$.
# 3. **Sampling**: start from $\mathcal N(0, I)$ and integrate the reverse SDE
#    with Euler–Maruyama.
#
# The math is in `docs/METHOD.md`; the choice of `T`, and why the data is not
# rescaled, are in `docs/EXPERIMENTS.md`.

# %% [markdown]
# ## Imports and seeds

# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Allocate GPU memory on demand.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Deterministic GPU ops, so a seed reproduces its run.
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from scipy.stats import shapiro
from tqdm.auto import tqdm

from ddpm.data import CurveData, Standardizer
from ddpm.loss import weighted_x0_loss
from ddpm.net import Denoiser
from ddpm.sampler import reverse_sde
from ddpm.sde import ou_coefficients, ou_forward_sample_batch, score_from_x0_hat
from ddpm.times import (
    exponential_linspace,
    exponential_uniform,
    sample_times_exponential,
    sample_times_uniform,
)
from ddpm.utils.metrics import curve_distance, x_coverage
from ddpm.utils.report import write_record

SEED = 0
# Seed sweeps: `RUN_SEED=N` runs seed N and writes under results/sine/seedN/.
# Unset, the run uses SEED above and writes under results/sine/.
if "RUN_SEED" in os.environ:
    SEED = int(os.environ["RUN_SEED"])
SEED_DIR = f"seed{SEED}" if "RUN_SEED" in os.environ else None
key = jax.random.key(SEED)

print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## 0. Configuration

# %%
@dataclass
class ModelConfig:
    """Denoiser architecture."""
    hidden_dims: tuple[int, ...] = (256, 256)


@dataclass
class TrainingConfig:
    """Forward process, training and sampling hyperparameters."""
    # Forward process
    T: float = 4.0              # final time: T < 3 under-covers x, T > 5 only blurs samples
    t_min: float = 1e-4         # smallest training time: avoids beta = 0 (division by zero)
    eps: float = 0.1            # offset of the exponential time grid / sampler
    normalize: bool = False     # standardize data before diffusion (blurs samples here)

    # Training
    n_iter: int = 1_000
    batch_size: int = 512
    learning_rate: float = 0.01
    time_sampler: str = "uniform"   # "uniform" or "exponential" (biased toward small t)

    # Sampling
    n_steps: int = 200          # time-grid points for the reverse SDE
    n_samples: int = 10_000


# Instantiate configs — edit these to tune the experiment
data = CurveData()              # sin(x) + N(0, 0.05²), x ~ U(-10, 10)
model_cfg = ModelConfig()
train_cfg = TrainingConfig()

N_PLOT = 1000                   # data points shown in the plots
y_min, y_max = -3, 3            # plotting range only (sin is in [-1, 1])

# Paths: outputs go to results/sine/[seedN/].
EXPERIMENT = "sine"
RESULTS_DIR = Path("../../results") / EXPERIMENT
if SEED_DIR is not None:
    RESULTS_DIR = RESULTS_DIR / SEED_DIR
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / "log.txt"


def save_and_show(fig, name):
    """Save to FIGURES_DIR/<name>.png, show inline, then close (scripts reuse open figures)."""
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150)
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 1. Data

# %%
key, subkey = jax.random.split(key)
data_plot = data.sample(subkey, N_PLOT)

fig, ax = plt.subplots()
ax.scatter(data_plot[:, 0], data_plot[:, 1], s=1, color="red")
ax.set_title("Data from the sine function")
ax.set_xlabel("x")
ax.set_ylabel("sin(x) + noise")
save_and_show(fig, "data")

# %% [markdown]
# ### Rescaling (off by default)
#
# With `normalize = True` the model works on standardized data
# $z = (x - \mu)/\sigma$, and samples are mapped back with $x = \mu + \sigma z$.
# It makes the diffused data match $\mathcal N(0, I)$ at $t = T$, but the
# samples come out blurrier: see `docs/EXPERIMENTS.md`.

# %%
if train_cfg.normalize:
    # mean / std per coordinate, estimated from a large sample
    key, subkey = jax.random.split(key)
    standardizer = Standardizer.fit(data.sample(subkey, 100_000))
else:
    standardizer = Standardizer.identity(2)
print("mean:", standardizer.mean, " std:", standardizer.std)

# %% [markdown]
# ## 2. Forward diffusion (OU process)
#
# **`jax.vmap`** vectorizes a function: instead of a Python loop over samples,
# `ou_forward_sample_batch` applies the single-point sampler to each row of `x`,
# each element of `t` and each random key independently, inside compiled XLA.
# **`jax.jit`** compiles it for speed.

# %%
# 25 snapshots of the diffusion at t = T/25, 2T/25, ..., T (in the space the model works in)
data_model_space = standardizer.normalize(data_plot)
fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True, sharey=True)
for plot_index, ax in enumerate(axes.flat):
    key, subkey = jax.random.split(key)
    t = train_cfg.T * (plot_index + 1) / 25.0
    res = ou_forward_sample_batch(subkey, data_model_space, jnp.full(N_PLOT, t))

    ax.scatter(res[:, 0], res[:, 1], s=1)
    ax.set_title(f"t = {t:.2f}", fontsize=9)
fig.tight_layout()
save_and_show(fig, "forward_diffusion")

# %% [markdown]
# At $t = T$, check how close the diffused data is to the $\mathcal N(0, I)$
# distribution sampling starts from. Shapiro–Wilk tests normality only, not
# mean 0 / variance 1, so the mean and std are printed too.

# %%
# `res` holds the last snapshot, t = T
for dim, name in enumerate(["x", "y"]):
    stat, p = shapiro(res[:, dim])
    print(f"{name}: mean = {res[:, dim].mean():+.3f}, std = {res[:, dim].std():.3f}, "
          f"Shapiro-Wilk p-value = {p:.3f}")

# %% [markdown]
# ## 3. Time discretization
#
# Small times matter most, so time points can be biased toward $t = 0$.

# %%
key, subkey = jax.random.split(key)
t_random = exponential_uniform(subkey, train_cfg.T, 100, train_cfg.eps)
t_grid = exponential_linspace(train_cfg.eps, train_cfg.T, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharex=True)
ax1.scatter(t_random, jnp.zeros_like(t_random), alpha=0.5)
ax1.set_title("Exponential uniform sampling")
ax2.scatter(t_grid, jnp.zeros_like(t_grid), alpha=0.5)
ax2.set_title("Exponential linear spacing")
for ax in (ax1, ax2):
    ax.set_yticks([])
    ax.set_xlabel("t")
save_and_show(fig, "time_grids")

# %% [markdown]
# ## 4. Denoiser network
#
# The network predicts the noise $\hat\varepsilon(t, x_t)$ and converts it to
# $\hat x_0 = (x_t - \beta_t \hat\varepsilon) / \alpha_t$. Learning the noise
# works much better than learning $x_0$ directly.

# %%
denoiser_net = Denoiser(hidden_dims=model_cfg.hidden_dims)
nn_apply = jax.jit(denoiser_net.apply)

key, subkey = jax.random.split(key)
B_init = 16
params = denoiser_net.init(subkey, jnp.zeros((B_init, 1)), jnp.zeros((B_init, 2)))

# sanity checks: output shape and parameter shapes
print("output shape:", nn_apply(params, jnp.ones((B_init, 1)), jnp.zeros((B_init, 2))).shape)
for layer_name, layer_params in params["params"].items():
    for param_name, value in layer_params.items():
        print(f"{layer_name:>10} / {param_name:<6} shape: {value.shape}")

# %% [markdown]
# ## 5. Training
#
# Loss: $\mathbb E\big[(\alpha_t/\beta_t)^2\,\|\hat x_0 - x_0\|^2\big]$, which
# equals the noise-prediction loss $\mathbb E\|\hat\varepsilon - \varepsilon\|^2$.
#
# `jax.value_and_grad(f)` returns both $f$ and its gradient with respect to its
# first argument (`params`).

# %%
optimizer = optax.adam(learning_rate=train_cfg.learning_rate)
opt_state = optimizer.init(params)


def sample_times(key, size):
    if train_cfg.time_sampler == "uniform":
        return sample_times_uniform(key, train_cfg.T, size, train_cfg.t_min)
    if train_cfg.time_sampler == "exponential":
        return sample_times_exponential(key, train_cfg.T, size, train_cfg.t_min, train_cfg.eps)
    raise ValueError(f"unknown time_sampler {train_cfg.time_sampler!r}")


def training_batch(key):
    """(t, x0, xt): times (B,), clean data (B, 2), diffused data (B, 2), in model space."""
    key_t, key_x0, key_noise = jax.random.split(key, 3)
    t = sample_times(key_t, train_cfg.batch_size)
    x0 = standardizer.normalize(data.sample(key_x0, train_cfg.batch_size))
    xt = ou_forward_sample_batch(key_noise, x0, t)
    return t, x0, xt


@jax.jit
def update(params, opt_state, t, x0, xt):
    """One step: compute the loss, differentiate it, update the parameters."""
    loss_value, grad = jax.value_and_grad(weighted_x0_loss)(params, nn_apply, t, x0, xt)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return loss_value, params, opt_state


loss_history = []
with open(LOG_PATH, "w") as log:
    for ii in tqdm(range(train_cfg.n_iter)):
        key, subkey = jax.random.split(key)
        t, x0, xt = training_batch(subkey)
        loss_value, params, opt_state = update(params, opt_state, t, x0, xt)
        loss_history.append(float(loss_value))
        if (ii + 1) % 100 == 0:
            log.write(f"iter {ii + 1:>6}  loss(mean of last 100) "
                      f"{sum(loss_history[-100:]) / 100:.4f}\n")

# %%
fig, ax = plt.subplots()
ax.semilogy(loss_history)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
ax.set_title("Training loss")
save_and_show(fig, "loss")

# %% [markdown]
# ## 6. Reverse diffusion (sampling)
#
# Score from the denoiser: $\nabla \log p_t(x) \approx -(x - \alpha_t \hat x_0)/\beta_t^2$.
# The reverse SDE $dX = \big[\tfrac12 X + \nabla\log p_t(X)\big]\,dt + dW$ is
# integrated backward from $T$ to $0$ with Euler–Maruyama on the exponential grid.

# %%
discretization_points = exponential_linspace(train_cfg.eps, train_cfg.T, train_cfg.n_steps)

key, subkey = jax.random.split(key)
xT = jax.random.normal(subkey, (train_cfg.n_samples, 2))

key, subkey = jax.random.split(key)
final_sample = standardizer.unnormalize(
    reverse_sde(subkey, params, denoiser_net.apply, xT, discretization_points))

# %% [markdown]
# ## 7. Evaluation

# %%
# Estimated score at a small time, on a grid
t_score = 0.01
N_disc = 30

grid_x, grid_y = jnp.meshgrid(jnp.linspace(data.x_min, data.x_max, N_disc),
                              jnp.linspace(y_min, y_max, N_disc))
xt_grid = jnp.c_[grid_x.flatten(), grid_y.flatten()]
zt_grid = standardizer.normalize(xt_grid)
x0_hat_grid = nn_apply(params, jnp.full(len(zt_grid), t_score), zt_grid)
score_hat = standardizer.score_to_data_space(
    score_from_x0_hat(zt_grid, t_score, x0_hat_grid)).reshape(N_disc, N_disc, 2)

# x-component scaled by 2 so the horizontal arrows are easier to see
fig, ax = plt.subplots()
ax.quiver(grid_x, grid_y, 2 * score_hat[:, :, 0], score_hat[:, :, 1])
ax.set_title(f"Estimated score at t = {t_score}")
save_and_show(fig, "score_field")

# %%
key, subkey = jax.random.split(key)
true_sample = data.sample(subkey, train_cfg.n_samples)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
ax1.scatter(true_sample[:, 0], true_sample[:, 1], s=1, color="red")
ax1.set_title("True data")
ax2.scatter(final_sample[:, 0], final_sample[:, 1], s=1, color="black")
ax2.set_title("Generated by reverse diffusion")
save_and_show(fig, "samples")

# %% [markdown]
# ### Reported metrics
#
# - **Sharpness**: median $|y - \sin x|$, overall and per $|x|$ band. Exact data
#   scores $\approx 0.674 \times$ `noise_std` $\approx 0.034$.
# - **Coverage**: Wasserstein-1 distance between generated and true $x$ values,
#   and the std of $x$ (true: $20/\sqrt{12} \approx 5.77$).
#
# Written to `results/sine/[seedN/]reverse_sde/metrics.json`.

# %%
sharpness = curve_distance(final_sample, data.fn)
sharpness_ref = curve_distance(true_sample, data.fn)
coverage = x_coverage(final_sample, true_sample, data.x_min, data.x_max)

print(f"{'median |y - sin(x)|':<22}" + "".join(f"{band:>10}" for band in sharpness))
print(f"{'  true data':<22}" + "".join(f"{v:>10.3f}" for v in sharpness_ref.values()))
print(f"{'  generated':<22}" + "".join(f"{v:>10.3f}" for v in sharpness.values()))
print(f"x coverage: W1 = {coverage['w1_x']:.3f}, std = {coverage['std_x']:.2f} "
      f"(true {coverage['std_x_ref']:.2f}), outside [{data.x_min:g}, {data.x_max:g}]: "
      f"{coverage['frac_outside']:.1%}")

metrics = {
    "curve_distance": sharpness,
    "curve_distance_ref": sharpness_ref,
    **coverage,
    "final_loss": sum(loss_history[-100:]) / 100,
    "n_samples": train_cfg.n_samples,
}
write_record(RESULTS_DIR, EXPERIMENT, "reverse_sde", metrics, seed=SEED,
             config={"data": data, "model": model_cfg, "training": train_cfg},
             notes="Reverse-SDE samples vs fresh data draws.")
print(f"Record written to {RESULTS_DIR / 'reverse_sde' / 'metrics.json'}")
