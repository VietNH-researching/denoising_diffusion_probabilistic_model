# Experiment notes

All runs use the sine data (`CurveData()`: x ~ U(-10, 10), noise std 0.05), the
(256, 256) denoiser, Adam with learning rate 0.01, batch 512, 200 sampling steps
and 10,000 samples, unless stated otherwise.

**Metrics** (lower is better): *curve* is the median |y − sin x| of generated
samples (exact data: 0.034); *W1 x* is the Wasserstein-1 distance between
generated and true x values; *std x* is the std of generated x (true: 5.77).

These sweeps were run on the single-notebook version of the code, before it was
split into the `ddpm` package. The package reproduces that notebook's seed-42
run (same Shapiro p-values at t = T, curve distance 0.263, band values within
0.001; the sampler moved from a Python loop to `jax.lax.scan`).

## Rescaling the data does not help

The x coordinate has variance ≈ 33, so at T = 3 the diffused x still has std
1.60 instead of 1. Standardizing the data (`normalize = True`) fixes that (std
1.01), but samples get **blurrier**: rescaled, one sine period is only ≈ 1.1
units wide, so the curve is washed out by t ≈ 0.1. Nearly all the useful signal
then sits at tiny t, which uniform training-time sampling rarely visits.

Curve distance at T = 3, mean of 2 seeds:

| setting | 1k iterations | 10k iterations |
|---|---|---|
| raw scale, uniform t | 0.25 | 0.14 |
| raw scale, `time_sampler = "exponential"` | 0.20 | **0.10** |
| `normalize = True`, uniform t | 0.65 | 0.56 |
| `normalize = True`, `time_sampler = "exponential"` | 0.63 | 0.26 |
| raw scale, T = 6 | 0.36 | 0.21 |

## Choosing T

Mean ± standard error over 6 seeds:

| T | 1k iter, uniform t: curve / W1 x / std x | 10k iter, exponential t: curve / W1 x / std x |
|---|---|---|
| 2 | 0.201 ± 0.007 / 2.14 ± 0.10 / 3.58 | 0.092 ± 0.003 / 2.03 ± 0.14 / 3.71 |
| 3 | 0.245 ± 0.009 / 1.14 ± 0.19 / 4.71 | 0.105 ± 0.004 / 1.01 ± 0.20 / 5.00 |
| 4 | 0.283 ± 0.012 / 0.85 ± 0.13 / 5.33 | 0.115 ± 0.003 / 1.14 ± 0.30 / 5.51 |
| 5 | 0.308 ± 0.013 / 0.95 ± 0.11 / 5.43 | 0.118 ± 0.002 / 1.55 ± 0.33 / 5.64 |
| 6 | 0.318 ± 0.017 / 0.95 ± 0.13 / 5.44 | 0.120 ± 0.003 / 1.25 ± 0.42 / 5.69 |
| 8 | 0.337 ± 0.016 / 1.09 ± 0.11 / 5.32 | 0.128 ± 0.004 / 1.37 ± 0.54 / 5.69 |
| 10 | 0.359 ± 0.015 / 1.08 ± 0.14 / 5.45 | 0.128 ± 0.005 / 1.23 ± 0.36 / 5.84 |

- T = 2 is too small: sampling starts from N(0, 1) while the noised x still
  has std ≈ 2.3, so samples are squeezed toward x = 0.
- Coverage stops improving at T ≈ 4; beyond that W1 x stays ≈ 1 within seed
  noise, so the remaining error is the network's.
- Sharpness gets steadily worse as T grows, likely because the fixed training
  and sampling budgets are spread over a longer time range (not tested).

**Chosen: T = 4**, the best coverage in the default setup for a small loss in
sharpness. With 10k iterations and exponential training times, T = 3 and T = 4
are about even.

## Current results (T = 4, defaults)

`scripts/run_seeds.sh`, records in `results/sine/seed{0,1,2}/`:

| seed | curve | W1 x | std x |
|---|---|---|---|
| 0 | 0.311 | 0.47 | 6.02 |
| 1 | 0.304 | 0.83 | 5.52 |
| 2 | 0.278 | 0.52 | 5.41 |

The best setting found so far, 10k iterations with exponential training times
(curve ≈ 0.10–0.12), is still about 3× blurrier than the data.
