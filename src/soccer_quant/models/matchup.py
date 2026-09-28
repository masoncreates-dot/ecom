"""Stylistic matchups: does this attack's style exploit that defence's style
beyond what the two teams' overall strengths already say?

Team style profiles come from match stats (shot volume, set pieces via
corners, shot quality via on-target share). A heavily regularised ridge
regression learns how attack-style x defence-style interactions move goals
relative to the Dixon-Coles expectation. Effects are capped; if they don't
improve the backtest, set ``matchup_cap = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ModelParams
from .context import Adjustment
from .dixon_coles import DixonColes

FEATURES = ("shot volume", "set pieces", "shot quality")


@dataclass
class MatchupModel:
    profiles: pd.DataFrame  # z-scored style metrics indexed by team
    beta: np.ndarray
    center: np.ndarray
    cap: float

    @classmethod
    def fit(cls, matches: pd.DataFrame, dc: DixonColes, weights: np.ndarray, p: ModelParams) -> "MatchupModel | None":
        cols = ["home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners"]
        if p.matchup_cap <= 0 or not set(cols) <= set(matches.columns):
            return None
        mask = matches[cols].notna().all(axis=1).to_numpy()
        if mask.sum() < p.matchup_min_matches:
            return None
        m, w = matches[mask], weights[mask]
        profiles = _profiles(m, w)
        x_home = _features(profiles, m["home_key"], m["away_key"])
        x_away = _features(profiles, m["away_key"], m["home_key"])
        lam_mu = np.array([dc.expected_goals(h, a) for h, a in zip(m["home_key"], m["away_key"])])
        r_home = np.log((m["home_goals"].to_numpy() + 0.5) / (lam_mu[:, 0] + 0.5))
        r_away = np.log((m["away_goals"].to_numpy() + 0.5) / (lam_mu[:, 1] + 0.5))
        X = np.vstack([x_home, x_away])
        y = np.concatenate([r_home, r_away])
        ww = np.concatenate([w, w])
        center = np.average(X, axis=0, weights=ww)
        Xc = X - center
        yc = y - np.average(y, weights=ww)
        A = Xc.T @ (Xc * ww[:, None]) + p.matchup_ridge * np.eye(X.shape[1])
        beta = np.linalg.solve(A, Xc.T @ (yc * ww))
        return cls(profiles, beta, center, p.matchup_cap)

    def adjustment(self, home: str, away: str) -> Adjustment:
        fh = _features(self.profiles, [home], [away])[0] - self.center
        fa = _features(self.profiles, [away], [home])[0] - self.center
        eh = float(np.clip(fh @ self.beta, -self.cap, self.cap))
        ea = float(np.clip(fa @ self.beta, -self.cap, self.cap))
        top = FEATURES[int(np.argmax(np.abs(fh * self.beta) + np.abs(fa * self.beta)))]
        return Adjustment("matchup", float(np.exp(eh)), float(np.exp(ea)), f"largest effect: {top}")

    def style(self, team: str) -> dict[str, float]:
        if team not in self.profiles.index:
            return {}
        return self.profiles.loc[team].round(2).to_dict()


def _profiles(m: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    home = pd.DataFrame({"team": m["home_key"].to_numpy(), "w": w,
                         "shots_for": m["home_shots"].to_numpy(), "shots_against": m["away_shots"].to_numpy(),
                         "sot_for": m["home_sot"].to_numpy(), "sot_against": m["away_sot"].to_numpy(),
                         "corners_for": m["home_corners"].to_numpy(), "corners_against": m["away_corners"].to_numpy()})
    away = pd.DataFrame({"team": m["away_key"].to_numpy(), "w": w,
                         "shots_for": m["away_shots"].to_numpy(), "shots_against": m["home_shots"].to_numpy(),
                         "sot_for": m["away_sot"].to_numpy(), "sot_against": m["home_sot"].to_numpy(),
                         "corners_for": m["away_corners"].to_numpy(), "corners_against": m["home_corners"].to_numpy()})
    both = pd.concat([home, away])
    metrics = [c for c in both.columns if c not in ("team", "w")]
    agg = both.groupby("team").apply(
        lambda g: pd.Series({c: np.average(g[c], weights=g["w"]) for c in metrics}), include_groups=False)
    agg["quality_for"] = agg["sot_for"] / agg["shots_for"].clip(lower=1)
    agg["quality_against"] = agg["sot_against"] / agg["shots_against"].clip(lower=1)
    return (agg - agg.mean()) / agg.std(ddof=0).replace(0, 1)


def _features(profiles: pd.DataFrame, attackers, defenders) -> np.ndarray:
    def col(teams, name):
        return profiles[name].reindex(list(teams)).fillna(0.0).to_numpy()

    return np.column_stack([
        col(attackers, "shots_for") * col(defenders, "shots_against"),
        col(attackers, "corners_for") * col(defenders, "corners_against"),
        col(attackers, "quality_for") * col(defenders, "quality_against"),
    ])
