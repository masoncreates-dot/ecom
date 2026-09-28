"""In-memory snapshot of everything the models need for one league."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import League
from .storage import FINISHED, Store
from .venues import VENUES


@dataclass
class Dataset:
    league: League
    matches: pd.DataFrame
    odds: pd.DataFrame
    teams: pd.DataFrame
    players: pd.DataFrame
    squads: pd.DataFrame
    injuries: pd.DataFrame
    lineups: pd.DataFrame
    coaches: pd.DataFrame
    transfers: pd.DataFrame
    weather: pd.DataFrame
    team_schedule: pd.DataFrame

    @classmethod
    def from_store(cls, store: Store, league: League) -> "Dataset":
        matches = store.frame("matches", "league=?", (league.key,))
        ids = tuple(matches["match_id"])
        team_keys = set(matches["home_key"]) | set(matches["away_key"])
        teams = store.frame("teams")
        teams = teams[teams["team_key"].isin(team_keys) | (teams["league"] == league.key)]
        return cls(
            league=league,
            matches=prepare_matches(matches),
            odds=_filter(store.frame("odds"), "match_id", ids),
            teams=with_venues(teams, team_keys),
            players=store.frame("players"),
            squads=_filter(store.frame("squads"), "team_key", team_keys),
            injuries=_filter(store.frame("injuries"), "team_key", team_keys),
            lineups=_filter(store.frame("lineups"), "team_key", team_keys),
            coaches=_filter(store.frame("coaches"), "team_key", team_keys),
            transfers=store.frame("transfers"),
            weather=_filter(store.frame("weather"), "match_id", ids),
            team_schedule=_to_utc(_filter(store.frame("team_schedule"), "team_key", team_keys), "kickoff"),
        )

    @property
    def completed(self) -> pd.DataFrame:
        m = self.matches
        return m[(m["status"] == FINISHED) & m["home_goals"].notna() & m["away_goals"].notna()]

    def team_api_ids(self) -> dict[str, int]:
        t = self.teams.dropna(subset=["api_id"])
        return {k: int(v) for k, v in zip(t["team_key"], t["api_id"])}


def prepare_matches(matches: pd.DataFrame) -> pd.DataFrame:
    m = _to_utc(matches.copy(), "kickoff")
    for col in ("home_goals", "away_goals", "home_xg", "away_xg", "home_shots", "away_shots", "home_sot",
                "away_sot", "home_corners", "away_corners", "home_fouls", "away_fouls"):
        if col in m:
            m[col] = pd.to_numeric(m[col], errors="coerce")
    return m.sort_values("kickoff").reset_index(drop=True)


def with_venues(teams: pd.DataFrame, team_keys: set[str]) -> pd.DataFrame:
    """Fill stadium coordinates from the static table where the store has none."""
    teams = teams.set_index("team_key") if len(teams) else pd.DataFrame(
        columns=["league", "name", "api_id", "stadium", "city", "lat", "lon", "elevation_m"]).rename_axis("team_key")
    for key in team_keys:
        venue = VENUES.get(key)
        if key not in teams.index:
            teams.loc[key] = pd.Series(dtype=object)
        if venue is None:
            continue
        for col, value in (("stadium", venue.stadium), ("city", venue.city), ("lat", venue.lat),
                           ("lon", venue.lon), ("elevation_m", venue.elevation_m)):
            if pd.isna(teams.at[key, col]):
                teams.at[key, col] = value
    for col in ("lat", "lon", "elevation_m"):
        teams[col] = pd.to_numeric(teams[col], errors="coerce")
    return teams.reset_index()


def _filter(frame: pd.DataFrame, col: str, values) -> pd.DataFrame:
    return frame[frame[col].isin(set(values))].reset_index(drop=True)


def _to_utc(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    frame[col] = pd.to_datetime(frame[col], utc=True, format="ISO8601")
    return frame
