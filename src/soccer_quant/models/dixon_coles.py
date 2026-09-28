"""Time-weighted Dixon-Coles team ratings.

log E[home goals] = c + h + attack[home] - defence[away]
log E[away goals] = c     + attack[away] - defence[home]

plus the Dixon-Coles low-score correction (rho) and exponential time decay
(xi) so recent matches count more. Ratings are shrunk toward a prior with a
ridge penalty; promoted/new teams get a stronger, below-average prior.

The goal "signal" can be a blend of real goals and xG / shots-on-target
proxies, which carry less noise than goals alone. The Poisson term is then a
quasi-likelihood; the rho correction always uses the real score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class DixonColes:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_advantage: float
    intercept: float
    rho: float
    unknown_prior: float = 0.0
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {t: i for i, t in enumerate(self.teams)}

    def __contains__(self, team: str) -> bool:
        return team in self._index

    def rating(self, team: str) -> tuple[float, float]:
        i = self._index.get(team)
        if i is None:
            return self.unknown_prior, self.unknown_prior
        return float(self.attack[i]), float(self.defence[i])

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        att_h, def_h = self.rating(home)
        att_a, def_a = self.rating(away)
        adv = 0.0 if neutral else self.home_advantage
        return (float(np.exp(self.intercept + adv + att_h - def_a)),
                float(np.exp(self.intercept + att_a - def_h)))

    def table(self) -> list[tuple[str, float, float]]:
        """(team, attack, defence) sorted by overall strength."""
        rows = [(t, float(self.attack[i]), float(self.defence[i])) for t, i in self._index.items()]
        return sorted(rows, key=lambda r: -(r[1] + r[2]))


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """P(home = i, away = j) for i, j in 0..max_goals, normalised."""
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    m[0, 0] *= max(1.0 - lam * mu * rho, 0.0)
    m[0, 1] *= max(1.0 + lam * rho, 0.0)
    m[1, 0] *= max(1.0 + mu * rho, 0.0)
    m[1, 1] *= max(1.0 - rho, 0.0)
    return m / m.sum()


def fit(home: list[str] | np.ndarray, away: list[str] | np.ndarray,
        home_goals: np.ndarray, away_goals: np.ndarray, *,
        weights: np.ndarray | None = None,
        home_signal: np.ndarray | None = None, away_signal: np.ndarray | None = None,
        ridge: float = 1.0, priors: dict[str, float] | None = None,
        ridge_mult: dict[str, float] | None = None, unknown_prior: float = 0.0) -> DixonColes:
    home = np.asarray(home)
    away = np.asarray(away)
    teams = sorted(set(home) | set(away))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = np.array([index[t] for t in home])
    ai = np.array([index[t] for t in away])
    gx = np.asarray(home_goals, dtype=float)
    gy = np.asarray(away_goals, dtype=float)
    xs = gx if home_signal is None else np.asarray(home_signal, dtype=float)
    ys = gy if away_signal is None else np.asarray(away_signal, dtype=float)
    w = np.ones_like(gx) if weights is None else np.asarray(weights, dtype=float)
    prior = np.array([(priors or {}).get(t, 0.0) for t in teams])
    lam_r = ridge * np.array([(ridge_mult or {}).get(t, 1.0) for t in teams])

    masks = ((gx == 0) & (gy == 0), (gx == 0) & (gy == 1), (gx == 1) & (gy == 0), (gx == 1) & (gy == 1))
    data = (hi, ai, xs, ys, w, n, prior, lam_r, masks)

    mean_goals = max(np.average((xs + ys) / 2, weights=w), 0.1)
    x0 = np.concatenate([prior, prior, [0.25, np.log(mean_goals) - 0.12, -0.05]])
    bounds = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
    res = minimize(_objective, x0, args=data, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "gtol": 1e-7})
    x = res.x
    return DixonColes(teams=teams, attack=x[:n], defence=x[n:2 * n], home_advantage=float(x[2 * n]),
                      intercept=float(x[2 * n + 1]), rho=float(x[2 * n + 2]), unknown_prior=unknown_prior)


def _objective(x, hi, ai, xs, ys, w, n, prior, lam_r, masks):
    att, dfn = x[:n], x[n:2 * n]
    home_adv, c, rho = x[2 * n], x[2 * n + 1], x[2 * n + 2]
    eta_h = c + home_adv + att[hi] - dfn[ai]
    eta_a = c + att[ai] - dfn[hi]
    lam, mu = np.exp(eta_h), np.exp(eta_a)

    ll = w * (xs * eta_h - lam + ys * eta_a - mu)
    g_h = w * (xs - lam)
    g_a = w * (ys - mu)
    g_rho = 0.0

    m00, m01, m10, m11 = masks
    tau = np.ones_like(lam)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.maximum(tau, 1e-10)
    ll = ll + w * np.log(tau)

    wt = w / tau
    g_h[m00] += -wt[m00] * lam[m00] * mu[m00] * rho
    g_a[m00] += -wt[m00] * lam[m00] * mu[m00] * rho
    g_h[m01] += wt[m01] * lam[m01] * rho
    g_a[m10] += wt[m10] * mu[m10] * rho
    g_rho += np.sum(-wt[m00] * lam[m00] * mu[m00]) + np.sum(wt[m01] * lam[m01]) \
        + np.sum(wt[m10] * mu[m10]) - np.sum(wt[m11])

    d_att = att - prior
    d_def = dfn - prior
    f = -ll.sum() + np.sum(lam_r * (d_att ** 2 + d_def ** 2))

    grad = np.empty_like(x)
    grad[:n] = -(np.bincount(hi, g_h, n) + np.bincount(ai, g_a, n)) + 2 * lam_r * d_att
    grad[n:2 * n] = (np.bincount(ai, g_h, n) + np.bincount(hi, g_a, n)) + 2 * lam_r * d_def
    grad[2 * n] = -g_h.sum()
    grad[2 * n + 1] = -(g_h.sum() + g_a.sum())
    grad[2 * n + 2] = -g_rho
    return f, grad
