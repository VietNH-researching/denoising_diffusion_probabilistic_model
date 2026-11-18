# Code map

## Package

| Module | Contents |
|---|---|
| `data.py` | `CurveData` (noisy points on a curve, default `sin`); `Standardizer` (per-coordinate rescaling) |
| `sde.py` | OU forward process: `ou_coefficients` (α_t, β_t), forward sampling, `score_from_x0_hat` |
| `times.py` | Exponential sampling grid; uniform and small-t-biased training-time samplers |
| `net.py` | `Denoiser`: MLP that predicts the noise and returns x̂₀ (batched) |
| `loss.py` | `weighted_x0_loss`: (α/β)²‖x̂₀ − x₀‖², equal to the noise-prediction loss |
| `sampler.py` | `reverse_sde`: Euler–Maruyama on the reverse SDE (`jax.lax.scan`, jitted) |
| `utils/metrics.py` | `curve_distance` (sharpness), `x_coverage` (spread along the curve) |
| `utils/report.py` | Writes `metrics.json` records |

## Dependencies

Arrows mean "imports from":

```
notebooks/*, tests/*
    └──► ddpm.<module>
            net      ──► sde
            loss     ──► sde
            sampler  ──► sde
```

`data.py`, `sde.py` and `times.py` import no other package module. `utils`
never imports from the core package.

## One run (`notebooks/sine/sine.py`)

```
CurveData.sample ──► [Standardizer.normalize] ──► ou_forward_sample_batch at t ~ sampler(times.py)
                                                        │
                       Denoiser ◄── adam on weighted_x0_loss (n_iter steps)
                          │
N(0, I) at t = T ──► reverse_sde on exponential_linspace ──► [Standardizer.unnormalize] ──► metrics.json
```
