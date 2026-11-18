"""
Make the three README figures:

    assets/overview.png   the idea in one picture: data <-> noise, forward and backward
    assets/noising.png    data -> noise, snapshots of the forward OU process at t = 0 ... T
    assets/sampling.png   noise -> data, snapshots of the reverse SDE at t = T ... 0

Points are coloured by their x position on the curve (the clean point for the
noising figure, the final sample for the sampling figure), so you can follow
them through the process.

Usage (from the repo root):
    python scripts/make_readme_figures.py
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from ddpm.data import CurveData
from ddpm.loss import weighted_x0_loss
from ddpm.net import Denoiser
from ddpm.sampler import reverse_sde
from ddpm.sde import ou_forward_sample_batch
from ddpm.times import exponential_linspace, sample_times_exponential

# Best setting found in docs/EXPERIMENTS.md: 10k iterations, small-t-biased training times
SEED = 0
T = 4.0
T_MIN = 1e-4
EPS = 0.1
N_ITER = 10_000
BATCH = 512
LR = 0.01
N_STEPS = 10_000          # sampling grid points (N_STEPS - 1 Euler–Maruyama steps)
N_POINTS = 4000
SNAPSHOT_TIMES = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, T]   # noising order, read row by row

OUT_DIR = Path(__file__).resolve().parents[1] / "assets"
CMAP = "viridis"
XLIM, YLIM = (-11, 11), (-4, 4)


def train(key, data):
    net = Denoiser()
    key, subkey = jax.random.split(key)
    params = net.init(subkey, jnp.zeros((1, 1)), jnp.zeros((1, 2)))
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, key):
        key_t, key_x0, key_noise = jax.random.split(key, 3)
        t = sample_times_exponential(key_t, T, BATCH, T_MIN, EPS)
        x0 = data.sample(key_x0, BATCH)
        xt = ou_forward_sample_batch(key_noise, x0, t)
        loss, grad = jax.value_and_grad(weighted_x0_loss)(params, net.apply, t, x0, xt)
        updates, opt_state = optimizer.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    for i in range(N_ITER):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = update(params, opt_state, subkey)
        if (i + 1) % 2000 == 0:
            print(f"iter {i + 1:>6}  loss {float(loss):.4f}")
    return net, params


def sample_with_snapshots(key, net, params):
    """Run the reverse SDE in segments; return {grid time: samples} for each snapshot."""
    # sampling grid (N_STEPS points, N_STEPS - 1 steps); the grid point nearest each
    # snapshot time is moved onto it, so snapshots land exactly on the grid
    grid = exponential_linspace(EPS, T, N_STEPS)
    idx = sorted({int(jnp.argmin(jnp.abs(grid - s))) for s in SNAPSHOT_TIMES}, reverse=True)
    for i, s in zip(idx, sorted(SNAPSHOT_TIMES, reverse=True)):
        grid = grid.at[i].set(s)
    assert len(idx) == len(SNAPSHOT_TIMES) and bool(jnp.all(jnp.diff(grid) > 0))

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (N_POINTS, 2))
    snapshots = {float(grid[idx[0]]): x}
    for hi, lo in zip(idx[:-1], idx[1:]):
        key, subkey = jax.random.split(key)
        x = reverse_sde(subkey, params, net.apply, x, grid[lo:hi + 1])
        snapshots[float(grid[lo])] = x
    return snapshots


def strip(snapshots, colour_by, title, path):
    """A 3-column grid of scatter panels, read row by row, with equal x and y scales."""
    ncols = 3
    nrows = -(-len(snapshots) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.05 * nrows + 0.4),
                             sharex=True, sharey=True)
    for ax, (t, pts) in zip(axes.flat, snapshots):
        pts = np.asarray(pts)
        ax.scatter(pts[:, 0], pts[:, 1], c=colour_by, cmap=CMAP, vmin=-10, vmax=10,
                   s=1.5, alpha=0.8, linewidths=0, rasterized=True)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_title(f"t = {t:.2g}", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def overview(panels, colour_by, path):
    """Three snapshots (data, partly noisy, noise) with a forward arrow above, backward below."""
    fig = plt.figure(figsize=(12, 4.6))
    width, height, bottom = 0.27, 0.27, 0.36
    lefts = [0.045, 0.365, 0.685]
    for left, (label, pts) in zip(lefts, panels):
        ax = fig.add_axes([left, bottom, width, height])
        pts = np.asarray(pts)
        ax.scatter(pts[:, 0], pts[:, 1], c=colour_by, cmap=CMAP, vmin=-10, vmax=10,
                   s=1.5, alpha=0.8, linewidths=0, rasterized=True)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_title(label, fontsize=13)

    x_first, x_last = lefts[0] + width / 2, lefts[-1] + width / 2
    arrows = [  # (start, end, curvature, colour, text y, title, subtitle)
        ((x_first, 0.73), (x_last, 0.73), -0.12, "#c0392b", 0.93,
         "Forward process: add a little noise, step by step",
         "fixed recipe, nothing to learn"),
        ((x_last, 0.29), (x_first, 0.29), -0.12, "#2471a3", 0.08,
         "Backward process: remove a little noise, step by step",
         "a neural network learns how to do this"),
    ]
    for start, end, rad, colour, text_y, title, subtitle in arrows:
        fig.patches.append(matplotlib.patches.FancyArrowPatch(
            start, end, transform=fig.transFigure, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=10,head_width=6", linewidth=2.5, color=colour))
        fig.text(0.5, text_y, title, ha="center", va="center", fontsize=14,
                 fontweight="bold", color=colour)
        fig.text(0.5, text_y - 0.055, subtitle, ha="center", va="center", fontsize=11,
                 color="#555555", style="italic")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = CurveData()
    key = jax.random.key(SEED)

    # Noising: one set of clean points, diffused to each snapshot time
    key, subkey = jax.random.split(key)
    x0 = data.sample(subkey, N_POINTS)
    forward = []
    for t in SNAPSHOT_TIMES:
        key, subkey = jax.random.split(key)
        xt = x0 if t == 0 else ou_forward_sample_batch(subkey, x0, jnp.full(N_POINTS, t))
        forward.append((t, xt))
    strip(forward, np.asarray(x0[:, 0]), "Noising: data → noise",
          OUT_DIR / "noising.png")
    by_time = dict(forward)
    overview([("data", by_time[0.0]), ("partly noisy", by_time[0.25]),
              ("pure noise", by_time[T])], np.asarray(x0[:, 0]), OUT_DIR / "overview.png")

    # Sampling: train, then run the reverse SDE from N(0, I) back to the curve
    key, subkey = jax.random.split(key)
    net, params = train(subkey, data)
    key, subkey = jax.random.split(key)
    snaps = sample_with_snapshots(subkey, net, params)
    final = np.asarray(snaps[min(snaps)][:, 0])
    reverse = sorted(snaps.items(), key=lambda kv: -kv[0])
    strip(reverse, final, "Sampling: noise → data", OUT_DIR / "sampling.png")


if __name__ == "__main__":
    main()
