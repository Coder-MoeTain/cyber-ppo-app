"""Small dependency-light statistical utilities for paired seed comparisons."""
from __future__ import annotations
import numpy as np


def paired_bootstrap_ci(a, b, *, confidence=0.95, n_boot=10000, seed=1234):
    """Percentile bootstrap CI for mean paired difference (a - b)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("paired samples must be 1-D, equal length, and n >= 2")
    if not (0 < confidence < 1) or n_boot < 100:
        raise ValueError("confidence must be in (0,1) and n_boot >= 100")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("samples must be finite")
    diff = x - y
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return {"mean_difference": float(diff.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "confidence": float(confidence),
            "n": int(diff.size)}


def paired_permutation_test(a, b, *, n_permutations=20000, seed=1234):
    """Two-sided paired sign-flip randomization test; finite-sample corrected."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("paired samples must be 1-D, equal length, and n >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("samples must be finite")
    d = x - y
    observed = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, d.size))
    null = np.abs((signs * d).mean(axis=1))
    p = (1 + int(np.count_nonzero(null >= observed))) / (n_permutations + 1)
    return {"mean_difference": float(d.mean()), "p_value": float(p),
            "n": int(d.size), "permutations": int(n_permutations)}


def holm_bonferroni(p_values):
    """Return Holm-adjusted p-values in original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a 1-D array in [0,1]")
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
