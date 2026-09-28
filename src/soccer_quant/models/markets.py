"""Turn a score matrix into prices for every market we trade.

Every selection is described by two matrices over the score grid: the
fraction of the stake that wins and the fraction that loses (pushes and
quarter-line half results fall out naturally). From those:

    EV(odds)   = (odds - 1) * W - L        W = sum(P * win), L = sum(P * lose)
    fair odds  = 1 + L / W
    Kelly      = argmax_f  E[log(1 + f * R)]

AH lines are the handicap applied to the named side (home -0.75, away +0.75).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar

OU_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)


@dataclass(frozen=True)
class Selection:
    market: str  # "1x2", "dc", "ou", "btts", "ah"
    pick: str  # home/draw/away, 1x/x2/12, over/under, yes/no, home/away
    line: float = 0.0

    def label(self) -> str:
        if self.market == "ou":
            return f"{self.pick.title()} {self.line:g}"
        if self.market == "ah":
            return f"AH {self.pick} {self.line:+g}"
        if self.market == "btts":
            return f"BTTS {self.pick}"
        if self.market == "dc":
            return f"Double chance {self.pick.upper()}"
        return self.pick.title()


@lru_cache(maxsize=4096)
def fractions(sel: Selection, size: int) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.indices((size, size))
    if sel.market == "1x2":
        win = {"home": i > j, "draw": i == j, "away": i < j}[sel.pick].astype(float)
        return win, 1.0 - win
    if sel.market == "dc":
        win = {"1x": i >= j, "x2": i <= j, "12": i != j}[sel.pick].astype(float)
        return win, 1.0 - win
    if sel.market == "btts":
        both = (i > 0) & (j > 0)
        win = (both if sel.pick == "yes" else ~both).astype(float)
        return win, 1.0 - win
    if sel.market == "ou":
        base = (i + j) if sel.pick == "over" else -(i + j)
        offset = -sel.line if sel.pick == "over" else sel.line
        return _line_fractions(base, offset)
    if sel.market == "ah":
        base = (i - j) if sel.pick == "home" else (j - i)
        return _line_fractions(base, sel.line)
    raise ValueError(f"unknown market {sel.market}")


def _line_fractions(base: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    quarter = abs((offset * 4) % 2 - 1) < 1e-9  # x.25 / x.75
    parts = (offset - 0.25, offset + 0.25) if quarter else (offset,)
    win = np.zeros(base.shape)
    lose = np.zeros(base.shape)
    for part in parts:
        margin = np.round(base + part, 6)
        win += (margin > 0) / len(parts)
        lose += (margin < 0) / len(parts)
    return win, lose


def win_lose(matrix: np.ndarray, sel: Selection) -> tuple[float, float]:
    win, lose = fractions(sel, matrix.shape[0])
    return float((matrix * win).sum()), float((matrix * lose).sum())


def probability(matrix: np.ndarray, sel: Selection) -> float:
    """Win probability with pushes removed (what 'fair odds' are quoted from)."""
    w, l = win_lose(matrix, sel)
    return w / (w + l) if w + l > 0 else 0.0


def fair_odds(matrix: np.ndarray, sel: Selection) -> float:
    w, l = win_lose(matrix, sel)
    return 1.0 + l / w if w > 0 else float("inf")


def expected_value(matrix: np.ndarray, sel: Selection, odds: float, commission: float = 0.0) -> float:
    w, l = win_lose(matrix, sel)
    return (odds - 1.0) * (1.0 - commission) * w - l


def kelly(matrix: np.ndarray, sel: Selection, odds: float, commission: float = 0.0) -> float:
    """Full-Kelly stake fraction, exact for pushes and quarter lines."""
    if expected_value(matrix, sel, odds, commission) <= 0:
        return 0.0
    win, lose = fractions(sel, matrix.shape[0])
    ret = (odds - 1.0) * (1.0 - commission) * win - lose
    returns, probs = [], []
    for r in np.unique(np.round(ret, 9)):
        returns.append(r)
        probs.append(matrix[np.isclose(ret, r)].sum())
    returns, probs = np.array(returns), np.array(probs)
    if returns.min() >= 0:
        return 1.0
    upper = min(0.999, 0.999 / -returns.min())
    res = minimize_scalar(lambda f: -np.sum(probs * np.log1p(f * returns)), bounds=(0.0, upper), method="bounded")
    return float(res.x)


def settle(sel: Selection, home_goals: int, away_goals: int, odds: float, commission: float = 0.0) -> float:
    """Net return per unit stake for a settled bet."""
    size = max(home_goals, away_goals) + 1
    win, lose = fractions(sel, size)
    return float((odds - 1.0) * (1.0 - commission) * win[home_goals, away_goals] - lose[home_goals, away_goals])


def summarize(matrix: np.ndarray) -> dict:
    """Probabilities and fair odds for the standard markets."""
    i, j = np.indices(matrix.shape)
    p_home, p_draw, p_away = (float(matrix[i > j].sum()), float(np.trace(matrix)), float(matrix[i < j].sum()))
    out = {
        "home_xg": float((matrix.sum(axis=1) * np.arange(matrix.shape[0])).sum()),
        "away_xg": float((matrix.sum(axis=0) * np.arange(matrix.shape[1])).sum()),
        "1x2": {"home": p_home, "draw": p_draw, "away": p_away},
        "double_chance": {"1x": p_home + p_draw, "x2": p_draw + p_away, "12": p_home + p_away},
        "btts": {"yes": probability(matrix, Selection("btts", "yes"))},
        "over": {line: probability(matrix, Selection("ou", "over", line)) for line in OU_LINES},
        "home_clean_sheet": float(matrix[:, 0].sum()),
        "away_clean_sheet": float(matrix[0, :].sum()),
    }
    out["btts"]["no"] = 1 - out["btts"]["yes"]
    flat = np.argsort(matrix, axis=None)[::-1][:8]
    out["top_scores"] = [(int(k // matrix.shape[1]), int(k % matrix.shape[1]), float(matrix.flat[k])) for k in flat]
    out["ah_fair_line"] = fair_ah_line(matrix)
    out["total_fair_line"] = fair_total_line(matrix)
    return out


def fair_ah_line(matrix: np.ndarray) -> float:
    """Home handicap whose fair price is closest to evens."""
    lines = np.arange(-3.0, 3.01, 0.25)
    gaps = [abs(fair_odds(matrix, Selection("ah", "home", float(h))) - 2.0) for h in lines]
    return float(lines[int(np.argmin(gaps))])


def fair_total_line(matrix: np.ndarray) -> float:
    lines = np.arange(0.5, 6.01, 0.25)
    gaps = [abs(fair_odds(matrix, Selection("ou", "over", float(t))) - 2.0) for t in lines]
    return float(lines[int(np.argmin(gaps))])
