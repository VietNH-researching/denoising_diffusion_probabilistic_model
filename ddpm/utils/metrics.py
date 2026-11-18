"""
Sample-quality metrics for 2-D curve data (x, f(x) + noise).

- `curve_distance`: sharpness. Median |y - f(x)| overall and per |x| band.
  For exact data it is ≈ 0.674 · noise_std (median of a half-normal).
- `x_coverage`: spread along the curve. 1-D Wasserstein-1 distance between the
  generated and reference x values, the std of the generated x, and the share
  of samples outside the data's x range.

Lower is better for every distance. Never imports from the core package.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import wasserstein_distance

__all__ = ["DEFAULT_BANDS", "curve_distance", "x_coverage"]

#: |x| bands for `curve_distance`: centre, middle and edges of x ∈ [-10, 10].
DEFAULT_BANDS: tuple[tuple[float, float], ...] = ((0.0, 3.0), (3.0, 6.0), (6.0, 10.0))


def curve_distance(samples, fn: Callable, bands: Sequence[tuple[float, float]] = DEFAULT_BANDS
                   ) -> dict[str, float | None]:
    """Median |y - fn(x)| of `samples` (n, 2), overall (`all`) and per |x| band.

    A band with no samples gives `None`.
    """
    samples = np.asarray(samples)
    x, y = samples[:, 0], samples[:, 1]
    residual = np.abs(y - np.asarray(fn(x)))
    out: dict[str, float | None] = {"all": float(np.median(residual))}
    for lo, hi in bands:
        in_band = (np.abs(x) >= lo) & (np.abs(x) < hi)
        out[f"{lo:g}-{hi:g}"] = float(np.median(residual[in_band])) if in_band.any() else None
    return out


def x_coverage(samples, reference, x_min: float, x_max: float) -> dict[str, float]:
    """W1 between the x values of `samples` and `reference` (both (n, 2)), plus spread."""
    x = np.asarray(samples)[:, 0]
    x_ref = np.asarray(reference)[:, 0]
    return {
        "w1_x": float(wasserstein_distance(x, x_ref)),
        "std_x": float(x.std()),
        "std_x_ref": float(x_ref.std()),
        "frac_outside": float(np.mean((x < x_min) | (x > x_max))),
    }
