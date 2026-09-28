"""SQLite store. Every source writes here; models read Datasets built from it.

Upserts use COALESCE so a later source never blanks a column an earlier
source filled (football-data.co.uk brings shots and closing odds,
API-Football brings xG, fixture ids and kickoff times).
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

TABLES: dict[str, tuple[list[str], list[str]]] = {
    # table: (columns, primary key)
    "teams": (
        ["team_key", "league", "name", "api_id", "stadium", "city", "lat", "lon", "elevation_m"],
        ["team_key"],
    ),
    "matches": (
        ["match_id", "league", "season", "kickoff", "local_date", "home_key", "away_key", "status",
         "home_goals", "away_goals", "home_xg", "away_xg", "home_shots", "away_shots", "home_sot", "away_sot",
         "home_corners", "away_corners", "home_fouls", "away_fouls", "home_yellow", "away_yellow",
         "home_red", "away_red", "api_fixture_id", "venue", "venue_city", "referee", "round", "source"],
        ["match_id"],
    ),
    "odds": (
        ["match_id", "bookmaker", "market", "selection", "line", "price", "is_closing", "captured_at"],
        ["match_id", "bookmaker", "market", "selection", "line", "is_closing"],
    ),
    "players": (
        ["player_id", "season", "team_api_id", "league_api_id", "team_key", "name", "position", "age",
         "appearances", "lineups", "minutes", "rating", "goals", "assists", "shots", "shots_on",
         "key_passes", "tackles", "interceptions", "duels_won", "fetched_at"],
        ["player_id", "season", "team_api_id", "league_api_id"],
    ),
    "squads": (
        ["team_key", "player_id", "name", "position", "number", "age", "fetched_at"],
        ["team_key", "player_id"],
    ),
    "injuries": (
        ["fixture_api_id", "team_key", "player_id", "player_name", "type", "reason", "fetched_at"],
        ["fixture_api_id", "player_id"],
    ),
    "lineups": (
        ["fixture_api_id", "team_key", "player_id", "player_name", "position", "is_starter", "formation"],
        ["fixture_api_id", "player_id"],
    ),
    "transfers": (
        ["player_id", "player_name", "date", "type", "team_in_api_id", "team_in_name",
         "team_out_api_id", "team_out_name"],
        ["player_id", "date", "team_in_api_id", "team_out_api_id"],
    ),
    "coaches": (
        ["team_key", "coach_id", "name", "start_date", "end_date", "fetched_at"],
        ["team_key", "coach_id", "start_date"],
    ),
    "weather": (
        ["match_id", "temp_c", "precip_mm_h", "wind_kmh", "humidity", "source", "fetched_at"],
        ["match_id"],
    ),
    "team_schedule": (
        ["team_key", "kickoff", "competition", "fixture_api_id"],
        ["team_key", "fixture_api_id"],
    ),
    "signals": (
        ["signal_id", "created_at", "match_id", "kickoff", "home_key", "away_key", "market", "selection",
         "line", "bookmaker", "price", "model_prob", "market_prob", "ev", "stake", "closing_price", "clv",
         "result", "pnl"],
        ["signal_id"],
    ),
}

FINISHED = "FT"
SCHEDULED = "NS"


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        if str(path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        for table, (cols, pk) in TABLES.items():
            col_sql = ", ".join(cols)
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_sql}, PRIMARY KEY ({', '.join(pk)}))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_matches_league ON matches(league, kickoff)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_matches_fixture ON matches(api_fixture_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_odds_match ON odds(match_id)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def upsert(self, table: str, rows: Iterable[dict] | pd.DataFrame) -> int:
        cols, pk = TABLES[table]
        if isinstance(rows, pd.DataFrame):
            rows = rows.astype(object).where(rows.notna(), None).to_dict("records")
        rows = [r for r in rows if r]
        if not rows:
            return 0
        present = [c for c in cols if any(c in r for r in rows)]
        missing_pk = set(pk) - set(present)
        if missing_pk:
            raise ValueError(f"{table}: rows missing primary key columns {missing_pk}")
        updates = ", ".join(f"{c}=COALESCE(excluded.{c}, {table}.{c})" for c in present if c not in pk)
        sql = (f"INSERT INTO {table} ({', '.join(present)}) VALUES ({', '.join('?' for _ in present)}) "
               f"ON CONFLICT ({', '.join(pk)}) DO " + (f"UPDATE SET {updates}" if updates else "NOTHING"))
        self.conn.executemany(sql, [tuple(_sqlite_value(r.get(c)) for c in present) for r in rows])
        self.conn.commit()
        return len(rows)

    def delete(self, table: str, where: str, params: tuple = ()) -> None:
        self.conn.execute(f"DELETE FROM {table} WHERE {where}", params)
        self.conn.commit()

    def frame(self, table: str, where: str = "", params: tuple = ()) -> pd.DataFrame:
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "")
        return pd.read_sql_query(sql, self.conn, params=params)

    def resolve_match_id(self, league: str, home: str, away: str, local_date: str,
                         api_fixture_id: int | None = None, tolerance_days: int = 1) -> str:
        """Reuse an existing id for the same fixture so sources merge into one row."""
        if api_fixture_id is not None:
            row = self.conn.execute("SELECT match_id FROM matches WHERE api_fixture_id=?", (api_fixture_id,)).fetchone()
            if row:
                return row[0]
        d = date.fromisoformat(local_date)
        lo, hi = (d - timedelta(days=tolerance_days)).isoformat(), (d + timedelta(days=tolerance_days)).isoformat()
        row = self.conn.execute(
            "SELECT match_id FROM matches WHERE league=? AND home_key=? AND away_key=? AND local_date BETWEEN ? AND ? "
            "AND (api_fixture_id IS NULL OR ? IS NULL OR api_fixture_id=?) "
            "ORDER BY ABS(julianday(local_date)-julianday(?)) LIMIT 1",
            (league, home, away, lo, hi, api_fixture_id, api_fixture_id, local_date),
        ).fetchone()
        return row[0] if row else make_match_id(league, local_date, home, away)


def make_match_id(league: str, local_date: str, home: str, away: str) -> str:
    return f"{league}-{local_date}-{home}-{away}"


def _sqlite_value(value):
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
