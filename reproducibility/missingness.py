"""Seeded Jenga-style rank-block missingness with explicit protein-only inputs."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd


def combo_seed(seed, mechanism, fraction):
    token = f"{seed}|{mechanism}|{float(fraction):.6f}"
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)


def corrupt(x, mechanism, fraction, seed):
    """Exactly floor(n*f) omissions/column, preserving the original rank-block idea.

    Stable protein order replaces Jenga's unordered Python set. Rank blocks use
    the immutable complete reference to avoid sorting previously introduced NaNs.
    MAR is the Jenga label: its driver protein can itself become incomplete, so
    this is not a proof of Rubin MAR conditional on the final observed matrix.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Simulation requires a finite complete protein matrix.")
    if mechanism not in {"MCAR", "MAR", "MNAR"} or not 0 < fraction < 1:
        raise ValueError("Use MCAR/MAR/MNAR and a fraction strictly between 0 and 1.")
    n, p = a.shape
    k = int(n * fraction)
    if k < 1 or p < 2:
        raise ValueError("Too few rows/columns for this missingness panel.")
    rng = np.random.RandomState(seed)
    mask = np.zeros_like(a, dtype=bool)
    for j in range(p):
        if mechanism == "MCAR":
            rows = rng.permutation(n)[:k]
        else:
            start = rng.randint(0, n - k)
            driver = j if mechanism == "MNAR" else rng.choice([c for c in range(p) if c != j])
            rows = np.argsort(a[:, driver], kind="stable")[start:start + k]
        mask[rows, j] = True
    result = a.copy()
    result[mask] = np.nan
    return pd.DataFrame(result, index=x.index, columns=x.columns), mask


def validation_mask(incomplete, seed, rate=0.05):
    """Quantile-stratified holdouts drawn only from remaining observed cells."""
    a = np.asarray(incomplete, dtype=float)
    rng = np.random.default_rng(seed)
    result = np.zeros_like(a, dtype=bool)
    for j in range(a.shape[1]):
        rows = np.flatnonzero(np.isfinite(a[:, j]))
        bins = pd.qcut(a[rows, j], q=min(5, len(rows)), labels=False, duplicates="drop")
        if pd.isna(bins).all():
            bins = np.zeros(len(rows))
        for level in np.unique(np.asarray(bins)[np.isfinite(bins)]):
            group = rows[np.asarray(bins) == level]
            count = min(len(group), int(np.ceil(rate * len(group))))
            result[rng.choice(group, size=count, replace=False), j] = True
    if not result.any() or np.any(np.sum(np.isfinite(a) & ~result, axis=0) == 0):
        raise ValueError("Invalid validation mask: empty or a fully hidden protein.")
    return result


def masked_rmse(reference, completed, mask):
    a, b, m = np.asarray(reference, float), np.asarray(completed, float), np.asarray(mask, bool)
    if a.shape != b.shape or a.shape != m.shape or not m.any():
        raise ValueError("Scoring arrays must align and the evaluation mask must be nonempty.")
    if not np.isfinite(a[m]).all() or not np.isfinite(b[m]).all():
        raise ValueError("Non-finite predictions cannot be silently dropped from RMSE.")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
