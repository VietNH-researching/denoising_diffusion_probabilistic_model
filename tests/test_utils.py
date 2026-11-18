"""Tests for ddpm.utils (metrics and metrics.json records)."""

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ddpm.data import CurveData
from ddpm.utils.metrics import curve_distance, x_coverage
from ddpm.utils.report import json_safe, write_record


class TestCurveDistance:
    def test_exact_curve_is_zero(self):
        x = np.linspace(-10, 10, 101)
        d = curve_distance(np.c_[x, np.sin(x)], np.sin)
        assert d["all"] == pytest.approx(0.0, abs=1e-12)
        assert set(d) == {"all", "0-3", "3-6", "6-10"}

    def test_noisy_data_is_half_normal_median(self):
        s = CurveData(noise_std=0.05).sample(jax.random.key(0), 50_000)
        assert curve_distance(s, jnp.sin)["all"] == pytest.approx(0.674 * 0.05, rel=0.03)

    def test_empty_band_is_none(self):
        x = np.linspace(-2, 2, 11)
        assert curve_distance(np.c_[x, np.sin(x)], np.sin)["6-10"] is None


class TestXCoverage:
    def test_identical_sets(self):
        s = np.random.default_rng(0).uniform(-10, 10, (1000, 2))
        c = x_coverage(s, s, -10, 10)
        assert c["w1_x"] == 0.0 and c["frac_outside"] == 0.0
        assert c["std_x"] == c["std_x_ref"]

    def test_narrow_samples_detected(self):
        rng = np.random.default_rng(1)
        ref = rng.uniform(-10, 10, (5000, 2))
        narrow = ref * 0.5
        assert x_coverage(narrow, ref, -10, 10)["w1_x"] > 2.0


@dataclass
class _Cfg:
    lr: float = 1e-3
    n: int = 4


def test_json_safe():
    out = json_safe({"f": np.float32(0.5), "j": jnp.asarray(0.25), "i": np.int64(3),
                     "nan": float("nan"), "arr": jnp.arange(3), "cfg": _Cfg(),
                     "fn": jnp.sin, "tuple": (1, 2.0)})
    assert out == {"f": 0.5, "j": 0.25, "i": 3, "nan": None, "arr": [0, 1, 2],
                   "cfg": {"lr": 1e-3, "n": 4}, "fn": "sin", "tuple": [1, 2.0]}
    json.dumps(out)


def test_write_record_layout(tmp_path):
    rec = write_record(tmp_path / "sine" / "seed1", "sine", "reverse_sde",
                       {"w1_x": np.float64(0.5)}, seed=1, config={"data": CurveData()})
    on_disk = json.loads((tmp_path / "sine" / "seed1" / "reverse_sde" / "metrics.json").read_text())
    assert on_disk == rec
    assert (on_disk["experiment"], on_disk["run_id"], on_disk["seed"]) == ("sine", "reverse_sde", 1)
    assert on_disk["config"]["data"]["fn"] == "sin"
    for key in ("schema_version", "method", "commit", "timestamp", "notes"):
        assert key in on_disk
