"""Match-context adjustments: weather, altitude, travel, rest, head-to-head,
coaching changes.

Each function returns an ``Adjustment``: multipliers on the home and away
expected goals plus a human-readable note for the report. The coefficients
live in ModelParams and are deliberately conservative priors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ModelParams
from ..venues import haversine_km


@dataclass
class Adjustment:
    name: str
    home_mult: float = 1.0
    away_mult: float = 1.0
    detail: str = ""

    @property
    def active(self) -> bool:
        return abs(self.home_mult - 1) > 1e-4 or abs(self.away_mult - 1) > 1e-4


def weather(w: dict | None, p: ModelParams) -> Adjustment:
    if not w:
        return Adjustment("weather", detail="no forecast yet")
    mult, notes = 1.0, []
    rain, wind, temp = w.get("precip_mm_h") or 0.0, w.get("wind_kmh") or 0.0, w.get("temp_c")
    if rain >= p.rain_heavy_mm_h:
        mult *= p.rain_heavy_mult
        notes.append("heavy rain")
    elif rain >= p.rain_moderate_mm_h:
        mult *= p.rain_moderate_mult
        notes.append("rain")
    if wind >= p.wind_severe_kmh:
        mult *= p.wind_severe_mult
        notes.append("severe wind")
    elif wind >= p.wind_strong_kmh:
        mult *= p.wind_strong_mult
        notes.append("strong wind")
    if temp is not None:
        if temp >= p.extreme_heat_c:
            mult *= p.extreme_heat_mult
            notes.append("extreme heat")
        elif temp >= p.heat_c:
            mult *= p.heat_mult
            notes.append("heat")
        elif temp <= p.cold_c:
            mult *= p.cold_mult
            notes.append("freezing")
    desc = f"{temp:.0f}C, {rain:.1f} mm/h rain, wind {wind:.0f} km/h" if temp is not None else ""
    if notes:
        desc += " -> " + ", ".join(notes)
    return Adjustment("weather", mult, mult, desc)


def altitude(venue_elev: float | None, away_home_elev: float | None, p: ModelParams) -> Adjustment:
    if venue_elev is None or away_home_elev is None or np.isnan(venue_elev) or np.isnan(away_home_elev):
        return Adjustment("altitude", detail="elevation unknown")
    gap_km = (venue_elev - away_home_elev) / 1000.0
    excess = max(0.0, gap_km - p.altitude_threshold_km)
    detail = f"venue {venue_elev:.0f} m vs visitors' home {away_home_elev:.0f} m"
    return Adjustment("altitude", float(np.exp(p.altitude_home_coef * excess)),
                      float(np.exp(-p.altitude_away_coef * excess)), detail)


def travel(home_loc: tuple[float, float] | None, away_loc: tuple[float, float] | None,
           p: ModelParams) -> tuple[Adjustment, float | None]:
    if not home_loc or not away_loc or any(np.isnan(v) for v in (*home_loc, *away_loc)):
        return Adjustment("travel", detail="locations unknown"), None
    km = haversine_km(*home_loc, *away_loc)
    excess = max(0.0, km - p.travel_free_km) / 1000.0
    return Adjustment("travel", 1.0, float(np.exp(-p.travel_coef * excess)), f"visitors travel {km:.0f} km"), km


def rest(home_days: float | None, away_days: float | None, p: ModelParams) -> Adjustment:
    if home_days is None or away_days is None:
        return Adjustment("rest", detail="schedule unknown")
    h, a = 1.0, 1.0
    if home_days <= p.short_rest_days and away_days >= p.rested_days:
        h *= p.short_rest_attack_mult
        a *= p.short_rest_concede_mult
    if away_days <= p.short_rest_days and home_days >= p.rested_days:
        a *= p.short_rest_attack_mult
        h *= p.short_rest_concede_mult
    return Adjustment("rest", h, a, f"rest days {home_days:.0f} vs {away_days:.0f}")


def head_to_head(meetings: pd.DataFrame, home: str, expected_gd, kickoff: pd.Timestamp,
                 lam: float, mu: float, p: ModelParams) -> tuple[Adjustment, dict]:
    """Shrunken mean of (actual - expected) goal difference in past meetings.

    ``expected_gd(home, away)`` gives today's model view of a past fixture, so
    the residual isolates how this pairing has played out beyond team strength.
    """
    if meetings.empty:
        return Adjustment("head-to-head", detail="no recent meetings"), {"n": 0}
    m = meetings.sort_values("kickoff").tail(p.h2h_max_matches)
    age = (kickoff - m["kickoff"]).dt.days.to_numpy(dtype=float)
    weights = 0.5 ** (age / p.h2h_half_life_days)
    sign = np.where(m["home_key"] == home, 1.0, -1.0)
    actual = (m["home_goals"] - m["away_goals"]).to_numpy(dtype=float)
    expected = np.array([expected_gd(h, a) for h, a in zip(m["home_key"], m["away_key"])])
    resid = sign * (actual - expected)  # from today's home team's perspective
    shift = float(np.clip((weights * resid).sum() / (weights.sum() + p.h2h_shrink), -p.h2h_cap, p.h2h_cap))
    new_lam, new_mu = max(lam + shift / 2, 0.05), max(mu - shift / 2, 0.05)
    team_gd = sign * actual
    summary = {
        "n": len(m),
        "wins": int((team_gd > 0).sum()), "draws": int((team_gd == 0).sum()), "losses": int((team_gd < 0).sum()),
        "goals_for": float(np.where(sign > 0, m["home_goals"], m["away_goals"]).sum()),
        "goals_against": float(np.where(sign > 0, m["away_goals"], m["home_goals"]).sum()),
        "last": [(f"{r.kickoff:%Y-%m-%d}", r.home_key, int(r.home_goals), int(r.away_goals), r.away_key)
                 for r in m.tail(5).itertuples()],
    }
    detail = (f"last {len(m)}: W{summary['wins']} D{summary['draws']} L{summary['losses']}, "
              f"supremacy shift {shift:+.2f}")
    return Adjustment("head-to-head", new_lam / lam, new_mu / mu, detail), summary


def coach_weights(matches: pd.DataFrame, coaches: pd.DataFrame, as_of: pd.Timestamp,
                  p: ModelParams) -> np.ndarray:
    """Down-weight matches a team played before its current coach took over."""
    weights = np.ones(len(matches))
    for team, start in current_coach_starts(coaches, as_of).items():
        before = (matches["kickoff"] < start) & ((matches["home_key"] == team) | (matches["away_key"] == team))
        weights[before.to_numpy()] *= p.coach_pre_weight
    return weights


def current_coach_starts(coaches: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, pd.Timestamp]:
    if coaches.empty:
        return {}
    c = coaches.copy()
    c["start"] = pd.to_datetime(c["start_date"], utc=True, errors="coerce")
    c["end"] = pd.to_datetime(c["end_date"], utc=True, errors="coerce")
    c = c[(c["start"] <= as_of) & (c["end"].isna() | (c["end"] >= as_of))]
    return c.sort_values("start").groupby("team_key")["start"].last().to_dict()


def current_coach(coaches: pd.DataFrame, team: str, as_of: pd.Timestamp) -> tuple[str, pd.Timestamp] | None:
    if coaches.empty:
        return None
    c = coaches[coaches["team_key"] == team].copy()
    c["start"] = pd.to_datetime(c["start_date"], utc=True, errors="coerce")
    c["end"] = pd.to_datetime(c["end_date"], utc=True, errors="coerce")
    c = c[(c["start"] <= as_of) & (c["end"].isna() | (c["end"] >= as_of))].sort_values("start")
    if c.empty:
        return None
    return c["name"].iloc[-1], c["start"].iloc[-1]


def new_manager(team_coach: tuple[str, pd.Timestamp] | None, kickoff: pd.Timestamp, p: ModelParams) -> float:
    if team_coach is None:
        return 1.0
    days = (kickoff - team_coach[1]).days
    return 1.0 + p.new_manager_bounce if 0 <= days <= p.new_manager_days else 1.0
