# ddpm

A small score-based diffusion model in JAX. It learns to generate 2-D points on a
noisy sine curve, $(x, \sin x + \varepsilon)$ with $x \sim \mathcal U(-10, 10)$:

1. **Forward process.** An Ornstein–Uhlenbeck SDE $dX = -\tfrac12 X\,dt + dW$
   turns the data into $\mathcal N(0, I)$ by time $T$.
2. **Training.** An MLP on $(t, x_t)$ predicts the noise and returns the
   denoised estimate $\hat x_0$; the loss is the noise-prediction loss.
3. **Sampling.** The reverse SDE, driven by the score from $\hat x_0$, runs from
   $\mathcal N(0, I)$ at $t = T$ back to $t = 0$ with Euler–Maruyama.

The math is in [docs/METHOD.md](docs/METHOD.md). How `T` was chosen, and why the
data is not rescaled, is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Installation

```bash
pip install -e ".[test,notebooks]"
```

Install the JAX build that matches your accelerator; see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Running

The experiment is `notebooks/sine/sine.py` (Jupytext percent format), paired
with `sine.ipynb`. Hyperparameters are in `ModelConfig` and `TrainingConfig`;
the data is `CurveData()`. Run the script from its own folder:

```bash
cd notebooks/sine
python sine.py                 # writes to results/sine/
RUN_SEED=1 python sine.py      # writes to results/sine/seed1/
```

`scripts/run_seeds.sh` runs seeds 0, 1 and 2 (about 20 s each on one GPU).

A run writes `log.txt`, `figures/` and `reverse_sde/metrics.json` with:

- **sharpness**: median $|y - \sin x|$ overall and per $|x|$ band
  (`curve_distance`; exact data scores ≈ 0.034);
- **coverage**: Wasserstein-1 distance between generated and true $x$ values
  (`w1_x`), the std of $x$, and the share of samples outside $[-10, 10]$.

The reported results are in `results/sine/seed{0,1,2}/`:

| seed | curve distance (all) | 0–3 | 3–6 | 6–10 | W1 x | std x (true ≈ 5.77) |
|---|---|---|---|---|---|---|
| 0 | 0.311 | 0.159 | 0.333 | 0.425 | 0.47 | 6.02 |
| 1 | 0.304 | 0.168 | 0.317 | 0.528 | 0.83 | 5.52 |
| 2 | 0.278 | 0.165 | 0.303 | 0.420 | 0.52 | 5.41 |

The notebook enables deterministic GPU ops, so a seed gives the same run on the
same GPU model and software versions.

## Tests

```bash
pytest
```

## Layout

```
ddpm/                  package: data, OU process, time grids, denoiser, loss, reverse-SDE sampler
notebooks/sine/        the experiment (sine.py + paired sine.ipynb)
docs/                  method summary and experiment notes
scripts/run_seeds.sh   seeds 0-2 of every experiment
results/               metrics.json records
tests/                 pytest suite
```

See [CODEMAP.md](CODEMAP.md) for the module map and
[CONTRIBUTING.md](CONTRIBUTING.md) for development notes.
