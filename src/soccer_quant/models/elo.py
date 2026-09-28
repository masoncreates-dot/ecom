"""Goal-margin Elo ratings (World Football Elo style).

Elo reacts to results in a different way from Dixon-Coles: it is sequential,
updates after every match and never forgets entirely. Its rating gap is
mapped to an expected goal supremacy with a linear fit on history, which the
predictor blends with the Dixon-Coles supremacy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Elo:
    def __init__(self, k: float = 20.0, home_advantage: float = 60.0, initial: float = 1500.0,
                 new_team: float = 1440.0, season_gap_days: float = 40.0, regress: float = 0.2,
                 burn_in_days: float = 180.0):
        self.k = k
        self.home_advantage = home_advantage
        self.initial = initial
        self.new_team = new_team
        self.season_gap_days = season_gap_days
        self.regress = regress
        self.burn_in_days = burn_in_days
        self.ratings: dict[str, float] = {}
        self.intercept = 0.0
        self.slope = 1.0 / 250.0  # goals per Elo point; replaced by the fit

    def fit(self, matches: pd.DataFrame) -> "Elo":
        """matches: kickoff, home_key, away_key, home_goals, away_goals (finished only)."""
        m = matches.sort_values("kickoff")
        if m.empty:
            return self
        first = m["kickoff"].iloc[0]
        last_played: dict[str, pd.Timestamp] = {}
        diffs, margins = [], []
        for kickoff, home, away, hg, ag in zip(m["kickoff"], m["home_key"], m["away_key"],
                                               m["home_goals"], m["away_goals"]):
            established = (kickoff - first).days > 60
            for team in (home, away):
                if team not in self.ratings:
                    self.ratings[team] = self.new_team if established else self.initial
                elif (kickoff - last_played[team]).days > self.season_gap_days:
                    self.ratings[team] = self.initial + (1 - self.regress) * (self.ratings[team] - self.initial)
                last_played[team] = kickoff
            dr = self.ratings[home] + self.home_advantage - self.ratings[away]
            if (kickoff - first).days > self.burn_in_days:
                diffs.append(dr)
                margins.append(hg - ag)
            expected = 1.0 / (1.0 + 10 ** (-dr / 400.0))
            gd = hg - ag
            result = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
            n = abs(gd)
            mult = 1.0 if n <= 1 else 1.5 if n == 2 else (11.0 + n) / 8.0
            delta = self.k * mult * (result - expected)
            self.ratings[home] += delta
            self.ratings[away] -= delta
        if len(diffs) >= 50:
            self.slope, self.intercept = (float(v) for v in np.polyfit(diffs, margins, 1))
        return self

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.new_team)

    def supremacy(self, home: str, away: str, neutral: bool = False) -> float:
        """Expected home goal difference implied by the rating gap."""
        dr = self.rating(home) - self.rating(away) + (0.0 if neutral else self.home_advantage)
        return self.intercept + self.slope * dr
