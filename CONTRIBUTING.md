# Contributing

## Setup and tests

```bash
pip install -e ".[test,notebooks]"
pytest                  # full suite (~30 s)
pytest -n auto          # in parallel
```

## Conventions

- Numerical code must trace under `jax.jit`. Use `jax.lax.scan` or
  `fori_loop` inside JIT regions, not Python loops.
- Callables passed to jitted functions (`apply_fn` in `reverse_sde`) are static
  arguments.
- `Denoiser` is batched: `t` has shape `(B,)` or `(B, 1)`, `x` has shape `(B, d)`.
- Training times must be ≥ `t_min > 0`: at t = 0, β = 0 and the loss weight
  (α/β)² is infinite.
- Random keys are split in a fixed order in the notebook. Reordering the splits
  changes the results for a given seed.

## Adding an experiment

Create `notebooks/<name>/<name>.py` in Jupytext percent format and pair it:

```bash
jupytext --set-formats ipynb,py:percent notebooks/<name>/<name>.py
```

Keep hyperparameters in `ModelConfig` and `TrainingConfig`, honour `RUN_SEED`,
and write outputs to `results/<name>/[seedN/]`. Notebooks import only from the
package. Commit `.ipynb` files without outputs; figures go to `results/`.

Under `results/`, only `metrics.json` files are tracked.
