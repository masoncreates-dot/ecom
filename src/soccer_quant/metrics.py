"""Forecast-quality and trading metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def outcome_index(home_goals, away_goals) -> np.ndarray:
    hg, ag = np.asarray(home_goals), np.asarray(away_goals)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-12, 1)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[outcomes]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score for ordered home/draw/away outcomes (lower is better)."""
    onehot = np.eye(probs.shape[1])[outcomes]
    cum = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum(cum ** 2, axis=1) / (probs.shape[1] - 1)))


def calibration(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> pd.DataFrame:
    flat_p = probs.ravel()
    flat_y = np.eye(probs.shape[1])[outcomes].ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(flat_p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                         "predicted": float(flat_p[mask].mean()), "observed": float(flat_y[mask].mean())})
    return pd.DataFrame(rows)


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))
