"""Fit every model on data available at ``as_of`` and price upcoming fixtures.

Pipeline per match:
  Dixon-Coles expected goals
  -> Elo supremacy blend
  -> player availability / lineup / transfer adjustment (both teams)
  -> new-manager effect
  -> weather, altitude, travel, rest
  -> head-to-head residual
  -> stylistic matchup
  = model expected goals -> score matrix -> all markets
  -> optional blend with the market's implied expected goals
  -> value signals against the best available prices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import ModelParams, TradingParams
from .dataset import Dataset
from .models import context, dixon_coles, markets
from .models.context import Adjustment
from .models.elo import Elo
from .models.matchup import MatchupModel
from .models.players import PlayerModel, TeamPlayerAdjustment
from .storage import SCHEDULED
from .trading.signals import Signal, find_signals, market_probabilities

log = logging.getLogger(__name__)


@dataclass
class MatchPrediction:
    match_id: str
    league: str
    kickoff: pd.Timestamp
    home: str
    away: str
    base_xg: tuple[float, float]
    model_xg: tuple[float, float]
    final_xg: tuple[float, float]
    rho: float
    adjustments: list[Adjustment]
    model_matrix: np.ndarray
    matrix: np.ndarray
    markets: dict
    model_markets: dict
    market_probs: dict
    players: dict[str, TeamPlayerAdjustment] = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)

    def as_row(self) -> dict:
        m, mm = self.markets, self.model_markets
        mkt = self.market_probs.get(("1x2", 0.0), {})
        return {
            "match_id": self.match_id, "kickoff": self.kickoff, "home": self.home, "away": self.away,
            "home_xg": round(self.final_xg[0], 2), "away_xg": round(self.final_xg[1], 2),
            "p_home": round(m["1x2"]["home"], 4), "p_draw": round(m["1x2"]["draw"], 4),
            "p_away": round(m["1x2"]["away"], 4),
            "model_p_home": round(mm["1x2"]["home"], 4), "model_p_draw": round(mm["1x2"]["draw"], 4),
            "model_p_away": round(mm["1x2"]["away"], 4),
            "market_p_home": mkt.get("home"), "market_p_draw": mkt.get("draw"), "market_p_away": mkt.get("away"),
            "p_over_2_5": round(m["over"][2.5], 4), "p_btts": round(m["btts"]["yes"], 4),
            "fair_ah_home": m["ah_fair_line"], "fair_total": m["total_fair_line"],
            "top_score": "{}-{}".format(*m["top_scores"][0][:2]),
        }


class Predictor:
    def __init__(self, data: Dataset, model: ModelParams, trading: TradingParams, *,
                 use_players: bool = True, use_weather: bool = True, use_matchup: bool = True,
                 use_h2h: bool = True, use_elo: bool = True, market_weight: float | None = None):
        self.data = data
        self.p = model
        self.t = trading
        self.use_players = use_players
        self.use_weather = use_weather
        self.use_matchup = use_matchup
        self.use_h2h = use_h2h
        self.use_elo = use_elo
        self.market_weight = model.market_weight if market_weight is None else market_weight

    # ------------------------------------------------------------------ fitting
    def fit(self, as_of: pd.Timestamp) -> "Predictor":
        self.as_of = pd.Timestamp(as_of)
        done = self.data.completed
        done = done[done["kickoff"] < self.as_of]
        hist = done[done["kickoff"] >= self.as_of - pd.Timedelta(days=self.p.max_history_days)]
        if len(hist) < 30:
            raise ValueError(f"only {len(hist)} finished {self.data.league.key} matches before {self.as_of:%Y-%m-%d}")
        age = (self.as_of - hist["kickoff"]).dt.total_seconds().to_numpy() / 86400.0
        weights = np.exp(-self.p.time_decay_xi * age) * context.coach_weights(hist, self.data.coaches, self.as_of, self.p)
        home_signal, away_signal = self._goal_signal(hist)

        counts = pd.concat([hist["home_key"], hist["away_key"]]).value_counts()
        new_teams = [t for t, c in counts.items() if c < self.p.new_team_min_matches]
        self.dc = dixon_coles.fit(
            hist["home_key"].to_numpy(), hist["away_key"].to_numpy(),
            hist["home_goals"].to_numpy(), hist["away_goals"].to_numpy(),
            weights=weights, home_signal=home_signal, away_signal=away_signal, ridge=self.p.ridge,
            priors={t: self.p.new_team_prior for t in new_teams},
            ridge_mult={t: self.p.new_team_ridge_mult for t in new_teams},
            unknown_prior=self.p.new_team_prior,
        )
        self.elo = Elo(k=self.p.elo_k, home_advantage=self.data.league.elo_home_advantage).fit(done) \
            if self.use_elo else None
        self.matchup = MatchupModel.fit(hist, self.dc, weights, self.p) if self.use_matchup else None
        season = self.data.league.season_for(self.as_of)
        self.players = PlayerModel(self.data.players, self.data.squads, self.data.teams, self.data.league,
                                   season, self.p) if self.use_players and not self.data.players.empty else None
        self._history = done
        self._schedule = self._build_schedule()
        self._teams = self.data.teams.set_index("team_key")
        return self

    def _goal_signal(self, hist: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        hg, ag = hist["home_goals"].to_numpy(float), hist["away_goals"].to_numpy(float)
        xs, ys = hg.copy(), ag.copy()
        sot = hist[["home_sot", "away_sot"]].to_numpy(float) if "home_sot" in hist else np.full((len(hist), 2), np.nan)
        has_sot = ~np.isnan(sot).any(axis=1)
        if has_sot.sum() > 50 and self.p.sot_weight > 0:
            conv = (hg[has_sot].sum() + ag[has_sot].sum()) / max(sot[has_sot].sum(), 1.0)
            w = self.p.sot_weight
            xs[has_sot] = (1 - w) * hg[has_sot] + w * conv * sot[has_sot, 0]
            ys[has_sot] = (1 - w) * ag[has_sot] + w * conv * sot[has_sot, 1]
        xg = hist[["home_xg", "away_xg"]].to_numpy(float) if "home_xg" in hist else np.full((len(hist), 2), np.nan)
        has_xg = ~np.isnan(xg).any(axis=1)
        if has_xg.any() and self.p.xg_weight > 0:
            w = self.p.xg_weight
            xs[has_xg] = (1 - w) * hg[has_xg] + w * xg[has_xg, 0]
            ys[has_xg] = (1 - w) * ag[has_xg] + w * xg[has_xg, 1]
        return xs, ys

    def _build_schedule(self) -> dict[str, np.ndarray]:
        m = self.data.matches[~self.data.matches["status"].isin(["CANC", "PST"])]
        parts = [pd.DataFrame({"team": m["home_key"], "kickoff": m["kickoff"]}),
                 pd.DataFrame({"team": m["away_key"], "kickoff": m["kickoff"]})]
        if not self.data.team_schedule.empty:
            parts.append(self.data.team_schedule.rename(columns={"team_key": "team"})[["team", "kickoff"]])
        allk = pd.concat(parts).drop_duplicates()
        return {team: np.sort(_epoch_ns(g["kickoff"])) for team, g in allk.groupby("team")}

    # ------------------------------------------------------------------ prediction
    def upcoming(self, days: int) -> pd.DataFrame:
        m = self.data.matches
        end = self.as_of + pd.Timedelta(days=days)
        return m[(m["status"] == SCHEDULED) & (m["kickoff"] >= self.as_of) & (m["kickoff"] <= end)]

    def predict_upcoming(self, days: int) -> list[MatchPrediction]:
        return [self.predict(row) for _, row in self.upcoming(days).iterrows()]

    def predict(self, fx: pd.Series, odds: pd.DataFrame | None = None) -> MatchPrediction:
        home, away, kickoff = fx["home_key"], fx["away_key"], pd.Timestamp(fx["kickoff"])
        lam0, mu0 = self.dc.expected_goals(home, away)
        adjustments: list[Adjustment] = []  # multiplicative, applied after the Elo blend
        ctx: dict = {}

        lam, mu = lam0, mu0
        elo_adj = None
        if self.elo is not None and self.p.elo_weight > 0:
            total = lam + mu
            sup = (1 - self.p.elo_weight) * (lam - mu) + self.p.elo_weight * self.elo.supremacy(home, away)
            new_lam, new_mu = max((total + sup) / 2, 0.05), max((total - sup) / 2, 0.05)
            elo_adj = Adjustment("elo blend", new_lam / lam, new_mu / mu,
                                 f"Elo {self.elo.rating(home):.0f} vs {self.elo.rating(away):.0f}")
            lam, mu = new_lam, new_mu

        player_adj: dict[str, TeamPlayerAdjustment] = {}
        if self.players is not None:
            fixture_id = fx.get("api_fixture_id")
            for team in (home, away):
                player_adj[team] = self.players.adjustment(team, self._unavailable(fixture_id, team),
                                                           self._lineup(fixture_id, team))
            ph, pa = player_adj[home], player_adj[away]
            adjustments.append(Adjustment(
                "players", ph.attack_mult * pa.concede_mult, pa.attack_mult * ph.concede_mult,
                f"home {ph.source}, {len(ph.missing)} out/doubtful; away {pa.source}, {len(pa.missing)} out/doubtful"))

        coaches = {t: context.current_coach(self.data.coaches, t, kickoff) for t in (home, away)}
        ctx["coaches"] = {t: (c[0], c[1].strftime("%Y-%m-%d"), int((kickoff - c[1]).days)) if c else None
                          for t, c in coaches.items()}
        tenure = [f"{side}: {c[0]} ({(kickoff - c[1]).days} days)"
                  for side, c in (("home", coaches[home]), ("away", coaches[away])) if c]
        adjustments.append(Adjustment("new manager", context.new_manager(coaches[home], kickoff, self.p),
                                      context.new_manager(coaches[away], kickoff, self.p),
                                      "; ".join(tenure) or "coach data unavailable"))

        if self.use_weather:
            w = self._weather(fx["match_id"])
            ctx["weather"] = w
            adjustments.append(context.weather(w, self.p))

        venue, visitors = self._location(home), self._location(away)
        ctx["venue"] = venue
        adjustments.append(context.altitude(venue.get("elevation_m"), visitors.get("elevation_m"), self.p))
        travel_adj, km = context.travel(venue.get("loc"), visitors.get("loc"), self.p)
        ctx["travel_km"] = km
        adjustments.append(travel_adj)
        rest_h, rest_a = self._rest_days(home, kickoff), self._rest_days(away, kickoff)
        ctx["rest_days"] = (rest_h, rest_a)
        adjustments.append(context.rest(rest_h, rest_a, self.p))

        if self.use_h2h:
            meetings = self._meetings(home, away, kickoff)
            h2h_adj, ctx["h2h"] = context.head_to_head(
                meetings, home, lambda h, a: np.subtract(*self.dc.expected_goals(h, a)), kickoff, lam0, mu0, self.p)
            adjustments.append(h2h_adj)

        if self.matchup is not None:
            adjustments.append(self.matchup.adjustment(home, away))
            ctx["style"] = {home: self.matchup.style(home), away: self.matchup.style(away)}

        for adj in adjustments:
            lam *= adj.home_mult
            mu *= adj.away_mult
        if elo_adj is not None:
            adjustments.insert(0, elo_adj)
        lam = float(np.clip(lam, 0.6 * lam0, 1.6 * lam0))
        mu = float(np.clip(mu, 0.6 * mu0, 1.6 * mu0))

        model_matrix = dixon_coles.score_matrix(lam, mu, self.dc.rho, self.p.max_goals)
        if odds is None:
            odds = self._odds(fx["match_id"])
        mkt = market_probabilities(odds, self.t)
        final_lam, final_mu = lam, mu
        if self.market_weight > 0 and ("1x2", 0.0) in mkt:
            implied_lam, implied_mu = implied_goals(mkt, self.dc.rho, self.p.max_goals, start=(lam, mu))
            w = self.market_weight
            final_lam = float(np.exp((1 - w) * np.log(lam) + w * np.log(implied_lam)))
            final_mu = float(np.exp((1 - w) * np.log(mu) + w * np.log(implied_mu)))
            ctx["market_xg"] = (implied_lam, implied_mu)
        matrix = model_matrix if (final_lam, final_mu) == (lam, mu) else \
            dixon_coles.score_matrix(final_lam, final_mu, self.dc.rho, self.p.max_goals)

        pred = MatchPrediction(
            match_id=fx["match_id"], league=self.data.league.key, kickoff=kickoff, home=home, away=away,
            base_xg=(lam0, mu0), model_xg=(lam, mu), final_xg=(final_lam, final_mu), rho=self.dc.rho,
            adjustments=adjustments, model_matrix=model_matrix, matrix=matrix,
            markets=markets.summarize(matrix), model_markets=markets.summarize(model_matrix),
            market_probs=mkt, players=player_adj, context=ctx,
        )
        pred.signals = find_signals(matrix, odds, self.t, match_id=pred.match_id, kickoff=kickoff,
                                    home=home, away=away)
        return pred

    # ------------------------------------------------------------------ lookups
    def _odds(self, match_id: str) -> pd.DataFrame:
        """Latest pre-match quotes; closing quotes only when there are none
        (Liga MX history from football-data.co.uk is closing-only)."""
        o = self.data.odds[self.data.odds["match_id"] == match_id]
        pre = o[o["is_closing"] == 0]
        if pre.empty:
            return o[o["is_closing"] == 1]
        if pre["captured_at"].notna().any():
            pre = pre.sort_values("captured_at").groupby(["bookmaker", "market", "selection", "line"]).tail(1)
        return pre

    def _unavailable(self, fixture_id, team: str) -> dict[int, tuple[str, str]]:
        inj = self.data.injuries
        if fixture_id is None or pd.isna(fixture_id) or inj.empty:
            return {}
        rows = inj[(inj["fixture_api_id"] == int(fixture_id)) & (inj["team_key"] == team)]
        return {int(r.player_id): ("doubtful" if str(r.type).lower().startswith("question") else "out",
                                   str(r.reason or "")) for r in rows.itertuples()}

    def _lineup(self, fixture_id, team: str) -> list[int] | None:
        lu = self.data.lineups
        if fixture_id is None or pd.isna(fixture_id) or lu.empty:
            return None
        rows = lu[(lu["fixture_api_id"] == int(fixture_id)) & (lu["team_key"] == team) & (lu["is_starter"] == 1)]
        return [int(x) for x in rows["player_id"]] or None

    def _weather(self, match_id: str) -> dict | None:
        w = self.data.weather
        rows = w[w["match_id"] == match_id]
        return None if rows.empty else rows.iloc[-1].to_dict()

    def _location(self, team: str) -> dict:
        if team not in self._teams.index:
            return {}
        t = self._teams.loc[team]
        return {"stadium": t.get("stadium"), "city": t.get("city"), "elevation_m": _float(t.get("elevation_m")),
                "loc": (_float(t.get("lat")), _float(t.get("lon"))) if pd.notna(t.get("lat")) else None}

    def _rest_days(self, team: str, kickoff: pd.Timestamp) -> float | None:
        times = self._schedule.get(team)
        if times is None:
            return None
        # Ignore anything within 36h: the same fixture can appear twice with
        # slightly different kick-off times from different sources.
        now = kickoff.tz_convert("UTC").as_unit("ns").value
        idx = np.searchsorted(times, now - 36 * 3600 * 10**9) - 1
        if idx < 0:
            return None
        days = (now - times[idx]) / (86400 * 10**9)
        return float(days) if days < 30 else None

    def _meetings(self, home: str, away: str, kickoff: pd.Timestamp) -> pd.DataFrame:
        h = self._history
        pair = ((h["home_key"] == home) & (h["away_key"] == away)) | ((h["home_key"] == away) & (h["away_key"] == home))
        recent = h["kickoff"] >= kickoff - pd.Timedelta(days=365 * self.p.h2h_max_years)
        return h[pair & recent & (h["kickoff"] < kickoff)]


def implied_goals(mkt: dict, rho: float, max_goals: int, start: tuple[float, float]) -> tuple[float, float]:
    """Expected goals that reproduce the market's 1X2 (and over/under 2.5 if quoted)."""
    target_1x2 = mkt[("1x2", 0.0)]
    target_ou = mkt.get(("ou", 2.5), {}).get("over")

    def loss(x):
        m = dixon_coles.score_matrix(np.exp(x[0]), np.exp(x[1]), rho, max_goals)
        i, j = np.indices(m.shape)
        err = (m[i > j].sum() - target_1x2["home"]) ** 2 + (np.trace(m) - target_1x2["draw"]) ** 2 \
            + (m[i < j].sum() - target_1x2["away"]) ** 2
        if target_ou is not None:
            err += (m[(i + j) > 2].sum() - target_ou) ** 2
        return err

    res = minimize(loss, np.log(start), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10})
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


def _epoch_ns(series: pd.Series) -> np.ndarray:
    return series.dt.tz_convert("UTC").dt.tz_localize(None).dt.as_unit("ns").astype("int64").to_numpy()


def _float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f
