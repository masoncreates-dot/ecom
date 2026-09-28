"""Remove the bookmaker margin from a set of prices.

``shin`` (default) models the favourite-longshot bias via insider-trading
share z and is the most accurate of the three for 1X2 markets; ``power`` is
a good general alternative; ``proportional`` is the naive baseline.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied(prices) -> np.ndarray:
    return 1.0 / np.asarray(prices, dtype=float)


def overround(prices) -> float:
    return float(implied(prices).sum() - 1.0)


def devig(prices, method: str = "shin") -> np.ndarray:
    pi = implied(prices)
    if pi.sum() <= 1.0 + 1e-9:  # no margin (or exchange prices with negative overround)
        return pi / pi.sum()
    if method == "proportional":
        return pi / pi.sum()
    if method == "power":
        k = brentq(lambda k: np.sum(pi ** k) - 1.0, 1.0, 50.0)
        return pi ** k
    if method == "shin":
        total = pi.sum()

        def probs(z):
            return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / total) - z) / (2 * (1 - z))

        try:
            z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.4)
        except ValueError:  # margin too large to bracket; fall back
            return devig(prices, "power")
        p = probs(z)
        return p / p.sum()
    raise ValueError(f"unknown devig method {method}")
