"""API-Football (api-sports.io) connector: fixtures, xG, players, squads,
injuries, lineups, transfers, coaches, schedules and bookmaker odds.

Get a key at https://dashboard.api-football.com (or via RapidAPI) and export
it as API_FOOTBALL_KEY. The free plan allows 100 requests/day, which the
TTL cache stretches to a daily refresh of one league; a full two-league
refresh with player data needs a paid plan (~7,500 requests/day).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ..config import League
from ..http import FOREVER, BudgetExceeded, HttpClient, RequestBudget
from ..storage import FINISHED, SCHEDULED, Store
from ..teams import TeamResolver

log = logging.getLogger(__name__)

DIRECT_BASE = "https://v3.football.api-sports.io"
RAPID_BASE = "https://api-football-v1.p.rapidapi.com/v3"

HOUR = 3600
STATUS_MAP = {
    "FT": FINISHED, "AET": FINISHED, "PEN": FINISHED,
    "NS": SCHEDULED, "TBD": SCHEDULED,
    "PST": "PST", "CANC": "CANC", "ABD": "CANC", "AWD": "CANC", "WO": "CANC",
}
POSITIONS = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Attacker"}
BET_MARKETS = {"Match Winner": "1x2", "Goals Over/Under": "ou", "Both Teams Score": "btts", "Asian Handicap": "ah"}


class ApiFootball:
    def __init__(self, client: HttpClient, key: str, budget: RequestBudget, rapidapi: bool = False):
        self.client = client
        self.budget = budget
        if rapidapi:
            self.base = RAPID_BASE
            self.headers = {"x-rapidapi-key": key, "x-rapidapi-host": "api-football-v1.p.rapidapi.com"}
        else:
            self.base = DIRECT_BASE
            self.headers = {"x-apisports-key": key}

    def get(self, path: str, params: dict, ttl: float | None) -> list[dict]:
        """GET an endpoint, following pagination. Returns the concatenated ``response``."""
        out, page = [], 1
        while True:
            query = dict(params, page=page) if page > 1 else dict(params)
            body = self.client.get(f"{self.base}/{path}", params=query, headers=self.headers, ttl=ttl,
                                   budget=self.budget, validate=_raise_on_errors)
            out.extend(body.get("response", []))
            paging = body.get("paging") or {}
            if paging.get("current", 1) >= paging.get("total", 1):
                return out
            page += 1

    def fixtures(self, league_id: int, season: int) -> list[dict]:
        return self.get("fixtures", {"league": league_id, "season": season}, ttl=1 * HOUR)

    def team_fixtures(self, team_id: int, season: int) -> list[dict]:
        return self.get("fixtures", {"team": team_id, "season": season}, ttl=12 * HOUR)

    def fixture_statistics(self, fixture_id: int) -> list[dict]:
        return self.get("fixtures/statistics", {"fixture": fixture_id}, ttl=FOREVER)

    def lineups(self, fixture_id: int) -> list[dict]:
        return self.get("fixtures/lineups", {"fixture": fixture_id}, ttl=5 * 60)

    def injuries_on(self, league_id: int, season: int, day: str) -> list[dict]:
        return self.get("injuries", {"league": league_id, "season": season, "date": day}, ttl=3 * HOUR)

    def players(self, team_id: int, season: int, ttl: float | None) -> list[dict]:
        return self.get("players", {"team": team_id, "season": season}, ttl=ttl)

    def player(self, player_id: int, season: int) -> list[dict]:
        return self.get("players", {"id": player_id, "season": season}, ttl=7 * 24 * HOUR)

    def squad(self, team_id: int) -> list[dict]:
        return self.get("players/squads", {"team": team_id}, ttl=48 * HOUR)

    def transfers(self, team_id: int) -> list[dict]:
        return self.get("transfers", {"team": team_id}, ttl=48 * HOUR)

    def coaches(self, team_id: int) -> list[dict]:
        return self.get("coachs", {"team": team_id}, ttl=48 * HOUR)

    def odds_on(self, league_id: int, season: int, day: str) -> list[dict]:
        return self.get("odds", {"league": league_id, "season": season, "date": day}, ttl=30 * 60)


def _raise_on_errors(body: dict) -> None:
    # API-Football answers 200 with {"errors": {...}} for bad keys, quota and plan limits.
    if body.get("errors"):
        raise RuntimeError(f"API-Football error: {body['errors']}")


# --------------------------------------------------------------------------- transforms
def fixture_row(item: dict, league: League, resolver: TeamResolver) -> dict:
    fx, teams, score = item["fixture"], item["teams"], item.get("score") or {}
    kickoff = datetime.fromisoformat(fx["date"]).astimezone(timezone.utc)
    status = STATUS_MAP.get(fx["status"]["short"], "LIVE")
    # 90-minute score: AET/PEN matches report extra-time goals in "goals".
    full = score.get("fulltime") or {}
    goals = item.get("goals") or {}
    hg = full.get("home") if full.get("home") is not None else goals.get("home")
    ag = full.get("away") if full.get("away") is not None else goals.get("away")
    venue = fx.get("venue") or {}
    return {
        "league": league.key,
        "season": str(item["league"]["season"]),
        "kickoff": kickoff.isoformat(),
        "local_date": kickoff.astimezone(ZoneInfo(league.timezone)).date().isoformat(),
        "home_key": resolver.resolve(teams["home"]["name"]),
        "away_key": resolver.resolve(teams["away"]["name"]),
        "status": status,
        "home_goals": hg if status == FINISHED else None,
        "away_goals": ag if status == FINISHED else None,
        "api_fixture_id": fx["id"],
        "venue": venue.get("name"),
        "venue_city": venue.get("city"),
        "referee": fx.get("referee"),
        "round": item["league"].get("round"),
        "source": "api-football",
    }


def team_rows(item: dict, league: League, resolver: TeamResolver) -> list[dict]:
    out = []
    for side in ("home", "away"):
        t = item["teams"][side]
        out.append({"team_key": resolver.resolve(t["name"]), "league": league.key, "name": t["name"], "api_id": t["id"]})
    return out


def statistics_update(payload: list[dict], home_api_id: int) -> dict:
    """Map /fixtures/statistics onto match columns (xG, shots, corners...)."""
    names = {"expected_goals": "xg", "Total Shots": "shots", "Shots on Goal": "sot", "Corner Kicks": "corners",
             "Fouls": "fouls", "Yellow Cards": "yellow", "Red Cards": "red"}
    out = {}
    for team in payload:
        side = "home" if team["team"]["id"] == home_api_id else "away"
        for stat in team.get("statistics", []):
            col = names.get(stat.get("type"))
            if col is None or stat.get("value") in (None, ""):
                continue
            try:
                out[f"{side}_{col}"] = float(str(stat["value"]).rstrip("%"))
            except ValueError:
                continue
    return out


def injury_rows(payload: list[dict], resolver: TeamResolver, now: str) -> list[dict]:
    return [{
        "fixture_api_id": it["fixture"]["id"],
        "team_key": resolver.resolve(it["team"]["name"]),
        "player_id": it["player"]["id"],
        "player_name": it["player"]["name"],
        "type": it["player"].get("type"),
        "reason": it["player"].get("reason"),
        "fetched_at": now,
    } for it in payload]


def lineup_rows(payload: list[dict], fixture_id: int, resolver: TeamResolver) -> list[dict]:
    rows = []
    for team in payload:
        team_key = resolver.resolve(team["team"]["name"])
        for is_starter, group in ((1, "startXI"), (0, "substitutes")):
            for entry in team.get(group) or []:
                p = entry["player"]
                if p.get("id") is None:
                    continue
                rows.append({"fixture_api_id": fixture_id, "team_key": team_key, "player_id": p["id"],
                             "player_name": p.get("name"), "position": POSITIONS.get(p.get("pos"), p.get("pos")),
                             "is_starter": is_starter, "formation": team.get("formation")})
    return rows


def player_rows(payload: list[dict], team_keys_by_api: dict[int, str], now: str) -> list[dict]:
    rows = []
    for item in payload:
        p = item["player"]
        for st in item.get("statistics", []):
            games = st.get("games") or {}
            minutes = games.get("minutes") or 0
            if not minutes:
                continue
            team_id = (st.get("team") or {}).get("id")
            league_id = (st.get("league") or {}).get("id")
            rows.append({
                "player_id": p["id"], "season": (st.get("league") or {}).get("season"),
                "team_api_id": team_id, "league_api_id": league_id, "team_key": team_keys_by_api.get(team_id),
                "name": p.get("name"), "position": games.get("position"), "age": p.get("age"),
                "appearances": games.get("appearences"), "lineups": games.get("lineups"), "minutes": minutes,
                "rating": _float(games.get("rating")),
                "goals": (st.get("goals") or {}).get("total"), "assists": (st.get("goals") or {}).get("assists"),
                "shots": (st.get("shots") or {}).get("total"), "shots_on": (st.get("shots") or {}).get("on"),
                "key_passes": (st.get("passes") or {}).get("key"),
                "tackles": (st.get("tackles") or {}).get("total"),
                "interceptions": (st.get("tackles") or {}).get("interceptions"),
                "duels_won": (st.get("duels") or {}).get("won"),
                "fetched_at": now,
            })
    return rows


def squad_rows(payload: list[dict], team_key: str, now: str) -> list[dict]:
    rows = []
    for item in payload:
        for p in item.get("players", []):
            rows.append({"team_key": team_key, "player_id": p["id"], "name": p.get("name"),
                         "position": p.get("position"), "number": p.get("number"), "age": p.get("age"),
                         "fetched_at": now})
    return rows


def transfer_rows(payload: list[dict]) -> list[dict]:
    rows = []
    for item in payload:
        p = item["player"]
        for t in item.get("transfers", []):
            teams = t.get("teams") or {}
            rows.append({"player_id": p["id"], "player_name": p.get("name"), "date": t.get("date"),
                         "type": t.get("type"),
                         "team_in_api_id": (teams.get("in") or {}).get("id"),
                         "team_in_name": (teams.get("in") or {}).get("name"),
                         "team_out_api_id": (teams.get("out") or {}).get("id"),
                         "team_out_name": (teams.get("out") or {}).get("name")})
    return rows


def coach_rows(payload: list[dict], team_key: str, team_api_id: int, now: str) -> list[dict]:
    rows = []
    for coach in payload:
        for stint in coach.get("career") or []:
            if (stint.get("team") or {}).get("id") != team_api_id or not stint.get("start"):
                continue
            rows.append({"team_key": team_key, "coach_id": coach["id"], "name": coach.get("name"),
                         "start_date": stint["start"], "end_date": stint.get("end"), "fetched_at": now})
    return rows


def odds_rows(item: dict, match_id: str, now: str) -> list[dict]:
    rows = []
    for book in item.get("bookmakers", []):
        bookmaker = book["name"].lower().replace(" ", "_")
        for bet in book.get("bets", []):
            market = BET_MARKETS.get(bet.get("name"))
            if market is None:
                continue
            for v in bet.get("values", []):
                parsed = _parse_selection(market, str(v.get("value")))
                price = _float(v.get("odd"))
                if parsed is None or price is None or price <= 1.0:
                    continue
                selection, line = parsed
                rows.append({"match_id": match_id, "bookmaker": bookmaker, "market": market,
                             "selection": selection, "line": line, "price": price, "is_closing": 0,
                             "captured_at": now})
    return rows


def _parse_selection(market: str, value: str) -> tuple[str, float] | None:
    if market == "1x2":
        sel = {"Home": "home", "Draw": "draw", "Away": "away"}.get(value)
        return (sel, 0.0) if sel else None
    if market == "btts":
        sel = {"Yes": "yes", "No": "no"}.get(value)
        return (sel, 0.0) if sel else None
    parts = value.split()
    if len(parts) != 2:
        return None
    side, line = parts[0].lower(), _float(parts[1])
    if line is None:
        return None
    if market == "ou" and side in ("over", "under"):
        return side, line
    if market == "ah" and side in ("home", "away"):
        # Store every AH quote with the home handicap as the line.
        return side, line if side == "home" else -line
    return None


def _float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


# --------------------------------------------------------------------------- sync
def sync(api: ApiFootball, store: Store, league: League, resolver: TeamResolver, *, now: datetime,
         horizon_days: int, deep: bool = True, max_stat_fixtures: int = 40) -> dict[str, int]:
    """Refresh one league. Cheap, time-sensitive data first, so that running out
    of budget half-way still leaves fixtures, injuries, lineups and odds fresh."""
    season = league.season_for(now)
    stamp = now.isoformat()
    counts: dict[str, int] = {}
    try:
        _sync_fixtures(api, store, league, resolver, season, now, horizon_days, stamp, counts, max_stat_fixtures)
        if deep:
            _sync_teams(api, store, league, resolver, season, now, stamp, counts)
    except BudgetExceeded as exc:
        log.warning("API-Football budget exhausted during %s sync: %s", league.key, exc)
        counts["budget_exhausted"] = 1
    return counts


def _sync_fixtures(api, store, league, resolver, season, now, horizon_days, stamp, counts, max_stat_fixtures):
    items = api.fixtures(league.api_football_id, season)
    store.upsert("teams", [row for it in items for row in team_rows(it, league, resolver)])
    rows, horizon = [], now + timedelta(days=horizon_days)
    upcoming: list[tuple[dict, dict]] = []
    for it in items:
        row = fixture_row(it, league, resolver)
        row["match_id"] = store.resolve_match_id(league.key, row["home_key"], row["away_key"], row["local_date"],
                                                 row["api_fixture_id"])
        rows.append(row)
        kickoff = datetime.fromisoformat(row["kickoff"])
        if row["status"] == SCHEDULED and now <= kickoff <= horizon:
            upcoming.append((it, row))
    counts["fixtures"] = store.upsert("matches", rows)

    days = sorted({r["kickoff"][:10] for _, r in upcoming})
    injuries = []
    for day in days:
        injuries += injury_rows(api.injuries_on(league.api_football_id, season, day), resolver, stamp)
    counts["injuries"] = store.upsert("injuries", injuries)

    lineups = []
    for it, row in upcoming:
        if datetime.fromisoformat(row["kickoff"]) - now <= timedelta(minutes=90):
            lineups += lineup_rows(api.lineups(row["api_fixture_id"]), row["api_fixture_id"], resolver)
    counts["lineups"] = store.upsert("lineups", lineups)

    by_fixture = {r["api_fixture_id"]: r["match_id"] for _, r in upcoming}
    odds = []
    for day in days:
        for item in api.odds_on(league.api_football_id, season, day):
            match_id = by_fixture.get(item["fixture"]["id"])
            if match_id:
                odds += odds_rows(item, match_id, stamp)
    counts["odds"] = store.upsert("odds", odds)

    # xG and shot stats for recently finished matches that lack them.
    missing = store.conn.execute(
        "SELECT m.match_id, m.api_fixture_id, t.api_id FROM matches m JOIN teams t ON t.team_key = m.home_key "
        "WHERE m.league=? AND m.status=? AND m.api_fixture_id IS NOT NULL AND m.home_xg IS NULL "
        "ORDER BY m.kickoff DESC LIMIT ?", (league.key, FINISHED, max_stat_fixtures)).fetchall()
    stats = []
    for match_id, fixture_id, home_api_id in missing:
        update = statistics_update(api.fixture_statistics(fixture_id), home_api_id)
        if update:
            stats.append({"match_id": match_id, **update})
    counts["fixture_stats"] = store.upsert("matches", stats)


def _sync_teams(api, store, league, resolver, season, now, stamp, counts):
    teams = store.frame("teams", "league=? AND api_id IS NOT NULL", (league.key,))
    known = store.frame("teams", "api_id IS NOT NULL")
    keys_by_api = {int(a): k for a, k in zip(known["api_id"], known["team_key"])}
    for key in ("squads", "players", "coaches", "transfers", "schedule", "signings"):
        counts.setdefault(key, 0)
    for _, team in teams.iterrows():
        team_id, team_key = int(team["api_id"]), team["team_key"]
        counts["squads"] += store.upsert("squads", squad_rows(api.squad(team_id), team_key, stamp))
        counts["coaches"] += store.upsert("coaches", coach_rows(api.coaches(team_id), team_key, team_id, stamp))
        transfers = transfer_rows(api.transfers(team_id))
        counts["transfers"] += store.upsert("transfers", transfers)
        for s, ttl in ((season, 48 * HOUR), (season - 1, FOREVER)):
            counts["players"] += store.upsert("players", player_rows(api.players(team_id, s, ttl), keys_by_api, stamp))
        schedule = [{"team_key": team_key, "kickoff": datetime.fromisoformat(it["fixture"]["date"]).isoformat(),
                     "competition": it["league"]["name"], "fixture_api_id": it["fixture"]["id"]}
                    for it in api.team_fixtures(team_id, season)]
        counts["schedule"] += store.upsert("team_schedule", schedule)
        counts["signings"] += _sync_signings(api, store, transfers, team_id, season, now, keys_by_api, stamp)


def _sync_signings(api, store, transfers, team_id, season, now, keys_by_api, stamp) -> int:
    """Pull previous-club stats for players who joined in the last 180 days."""
    cutoff = (now - timedelta(days=180)).date().isoformat()
    written = 0
    for t in transfers:
        if t["team_in_api_id"] != team_id or not t["date"] or t["date"] < cutoff:
            continue
        have = store.conn.execute("SELECT 1 FROM players WHERE player_id=? AND season=? AND team_api_id!=?",
                                  (t["player_id"], season - 1, team_id)).fetchone()
        if not have:
            written += store.upsert("players", player_rows(api.player(t["player_id"], season - 1), keys_by_api, stamp))
    return written
