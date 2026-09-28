"""Historical results, match stats and bookmaker odds from football-data.co.uk.

EPL files ("main" format, one per season) carry opening and closing odds from
Pinnacle, Bet365, market average and maximum, plus shots, shots on target,
corners, fouls and cards. The Liga MX file ("extra" format, all seasons in
one file) carries closing odds only and no match stats.

AH odds are stored with ``line`` = the home team's handicap for both the
home and the away selection.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ..config import League
from ..http import FOREVER, HttpClient
from ..storage import FINISHED, SCHEDULED, Store
from ..teams import TeamResolver

log = logging.getLogger(__name__)

MAIN_URL = "https://football-data.co.uk/mmz4281/{season}/{code}.csv"
EXTRA_URL = "https://football-data.co.uk/new/{code}.csv"
# Upcoming fixtures with pre-match odds for the "main" leagues (EPL included).
FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
UK = ZoneInfo("Europe/London")

# bookmaker -> (opening prefix, closing prefix)
_1X2_BOOKS = {"bet365": ("B365", "B365C"), "pinnacle": ("PS", "PSC"), "max": ("Max", "MaxC"),
              "avg": ("Avg", "AvgC"), "betfair_ex": ("BFE", "BFEC")}
_OU_BOOKS = {"bet365": ("B365", "B365C"), "pinnacle": ("P", "PC"), "max": ("Max", "MaxC"),
             "avg": ("Avg", "AvgC"), "betfair_ex": ("BFE", "BFEC")}
_AH_BOOKS = {"bet365": ("B365AH", "B365CAH"), "pinnacle": ("PAH", "PCAH"), "max": ("MaxAH", "MaxCAH"),
             "avg": ("AvgAH", "AvgCAH"), "betfair_ex": ("BFEAH", "BFECAH")}

_STAT_COLS = {"HS": "home_shots", "AS": "away_shots", "HST": "home_sot", "AST": "away_sot",
              "HC": "home_corners", "AC": "away_corners", "HF": "home_fouls", "AF": "away_fouls",
              "HY": "home_yellow", "AY": "away_yellow", "HR": "home_red", "AR": "away_red"}


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def parse_csv(text: str, league: League, resolver: TeamResolver, season: str | None = None,
              scheduled: bool = False) -> tuple[list[dict], list[dict]]:
    """Return (match rows, odds rows) with provisional match ids (see ``load``).

    ``scheduled``: the file lists upcoming fixtures (no scores yet)."""
    df = pd.read_csv(io.StringIO(text.lstrip("\ufeff").removeprefix("\u00ef\u00bb\u00bf")), dtype=str, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    if scheduled and "Div" in df:
        df = df[df["Div"].str.strip() == league.football_data_code]
    home_col, away_col = ("HomeTeam", "AwayTeam") if "HomeTeam" in df else ("Home", "Away")
    hg_col, ag_col = ("FTHG", "FTAG") if "FTHG" in df else ("HG", "AG")
    df = df.dropna(subset=[home_col, away_col, "Date"])
    tz = ZoneInfo(league.timezone)
    matches, odds = [], []
    for _, row in df.iterrows():
        kickoff = _kickoff(row["Date"], row.get("Time"), league)
        if kickoff is None:
            continue
        home, away = resolver.resolve(row[home_col]), resolver.resolve(row[away_col])
        local_date = kickoff.astimezone(tz).date().isoformat()
        raw_season = row.get("Season")
        row_season = season or (raw_season[:4] if isinstance(raw_season, str) else str(league.season_for(kickoff)))
        match = {
            "league": league.key, "season": row_season, "kickoff": kickoff.isoformat(), "local_date": local_date,
            "home_key": home, "away_key": away, "status": FINISHED,
            "home_goals": _num(row.get(hg_col)), "away_goals": _num(row.get(ag_col)),
            "referee": row.get("Referee") if isinstance(row.get("Referee"), str) else None,
            "source": "football-data",
        }
        for src, dst in _STAT_COLS.items():
            match[dst] = _num(row.get(src))
        if match["home_goals"] is None or match["away_goals"] is None:
            if not scheduled:
                continue
            match["status"] = SCHEDULED
        matches.append(match)
        key = (home, away, local_date)
        odds.extend(_odds_rows(row, key))
    return matches, odds


def _odds_rows(row: pd.Series, key: tuple) -> list[dict]:
    out = []

    def add(bookmaker, market, selection, line, col, closing):
        price = _num(row.get(col))
        if price is not None and price > 1.0:
            out.append({"_key": key, "bookmaker": bookmaker, "market": market, "selection": selection,
                        "line": line, "price": price, "is_closing": int(closing)})

    for book, prefixes in _1X2_BOOKS.items():
        for closing, prefix in enumerate(prefixes):
            for sel, suffix in (("home", "H"), ("draw", "D"), ("away", "A")):
                add(book, "1x2", sel, 0.0, prefix + suffix, closing)
    for book, prefixes in _OU_BOOKS.items():
        for closing, prefix in enumerate(prefixes):
            add(book, "ou", "over", 2.5, f"{prefix}>2.5", closing)
            add(book, "ou", "under", 2.5, f"{prefix}<2.5", closing)
    for closing, line_col in enumerate(("AHh", "AHCh")):
        line = _num(row.get(line_col))
        if line is None:
            continue
        for book, prefixes in _AH_BOOKS.items():
            add(book, "ah", "home", line, prefixes[closing] + "H", closing)
            add(book, "ah", "away", line, prefixes[closing] + "A", closing)
    return out


def _kickoff(date_str: str, time_str, league: League) -> datetime | None:
    day = None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            day = datetime.strptime(date_str.strip(), fmt)
            break
        except ValueError:
            continue
    if day is None:
        return None
    if isinstance(time_str, str) and ":" in time_str:
        hh, mm = time_str.strip().split(":")[:2]
        # football-data.co.uk publishes kick-off times in UK time.
        return day.replace(hour=int(hh), minute=int(mm), tzinfo=UK)
    default_hour = 15 if league.key == "epl" else 19
    return day.replace(hour=default_hour, tzinfo=ZoneInfo(league.timezone))


def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def load(client: HttpClient, store: Store, league: League, resolver: TeamResolver,
         history_seasons: int, today: datetime) -> int:
    """Download history into the store. Returns number of match rows written."""
    current = league.season_for(today)
    texts: list[tuple[str, str | None]] = []
    if league.football_data_format == "main":
        for year in range(current - history_seasons, current + 1):
            url = MAIN_URL.format(season=season_code(year), code=league.football_data_code)
            ttl = 12 * 3600 if year == current else FOREVER
            try:
                texts.append((client.get(url, ttl=ttl, as_json=False), str(year)))
            except requests.HTTPError as exc:
                log.warning("football-data %s %s unavailable: %s", league.key, season_code(year), exc)
    else:
        url = EXTRA_URL.format(code=league.football_data_code)
        texts.append((client.get(url, ttl=12 * 3600, as_json=False), None))

    batches = [(*parse_csv(text, league, resolver, season), season) for text, season in texts]
    if league.football_data_format == "main":
        try:
            text = client.get(FIXTURES_URL, ttl=6 * 3600, as_json=False)
            matches, odds = parse_csv(text, league, resolver, str(current), scheduled=True)
            batches.append((matches, [{**o, "captured_at": today.isoformat()} for o in odds], str(current)))
        except requests.RequestException as exc:
            log.warning("football-data fixtures unavailable: %s", exc)
    written = 0
    for matches, odds, season in batches:
        if season is None:  # "extra" files hold every season; keep the recent ones
            cutoff = str(current - history_seasons)
            matches = [m for m in matches if m["season"] >= cutoff]
        ids = {}
        for m in matches:
            m["match_id"] = store.resolve_match_id(league.key, m["home_key"], m["away_key"], m["local_date"])
            ids[(m["home_key"], m["away_key"], m["local_date"])] = m["match_id"]
            # A kickoff already set by API-Football is more precise than a UK-time CSV.
            if store.conn.execute("SELECT 1 FROM matches WHERE match_id=? AND api_fixture_id IS NOT NULL",
                                  (m["match_id"],)).fetchone():
                m.pop("kickoff")
                m.pop("local_date")
        store.upsert("matches", matches)
        odds_rows = []
        for o in odds:
            match_id = ids.get(o.pop("_key"))
            if match_id:
                odds_rows.append({"captured_at": None, **o, "match_id": match_id})
        store.upsert("odds", odds_rows)
        written += len(matches)
    log.info("football-data: %d %s matches", written, league.key)
    return written
