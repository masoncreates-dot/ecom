from datetime import datetime, timezone

import pytest
import requests

from soccer_quant.config import LEAGUES
from soccer_quant.data import api_football, football_data
from soccer_quant.data.weather import summarize_hourly
from soccer_quant.http import BudgetExceeded, HttpClient, RequestBudget
from soccer_quant.storage import FINISHED, Store
from soccer_quant.teams import TeamResolver

EPL_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,PSH,PSD,PSA,AvgH,AvgD,AvgA,B365>2.5,B365<2.5,P>2.5,P<2.5,AHh,PAHH,PAHA,PSCH,PSCD,PSCA,PC>2.5,PC<2.5,AHCh,PCAHH,PCAHA
E0,16/08/2025,12:30,Nott'm Forest,Man United,1,3,A,M Oliver,10,15,3,7,11,9,4,6,2,1,0,0,3.1,3.5,2.3,3.2,3.6,2.35,3.1,3.5,2.3,1.8,2.0,1.85,2.05,0.25,1.95,1.95,3.3,3.6,2.3,1.9,2.0,0.25,1.93,1.97
"""

MEX_CSV = """Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA
Mexico,Liga MX,2025/2026,20/09/2025,03:05,U.N.A.M.- Pumas,Atl. San Luis,2,2,D,1.9,3.5,4.2,1.95,3.6,4.4,1.88,3.4,4.0
"""


def fixture(fid=101, status="FT", date="2025-09-20T03:05:00+00:00", home="Pumas UNAM", away="Atletico San Luis",
            goals=(2, 2), fulltime=(2, 2)):
    return {
        "fixture": {"id": fid, "date": date, "referee": "Ref", "status": {"short": status},
                    "venue": {"id": 1, "name": "Estadio Olimpico Universitario", "city": "Ciudad de Mexico"}},
        "league": {"id": 262, "season": 2025, "round": "Apertura - 9"},
        "teams": {"home": {"id": 20, "name": home}, "away": {"id": 21, "name": away}},
        "goals": {"home": goals[0], "away": goals[1]},
        "score": {"fulltime": {"home": fulltime[0], "away": fulltime[1]}},
    }


def test_parse_main_csv_matches_stats_and_odds():
    matches, odds = football_data.parse_csv(EPL_CSV, LEAGUES["epl"], TeamResolver(), "2025")
    assert len(matches) == 1
    m = matches[0]
    assert (m["home_key"], m["away_key"], m["home_goals"], m["away_goals"]) == ("nottingham-forest", "man-united", 1, 3)
    assert m["home_sot"] == 3 and m["away_corners"] == 6 and m["local_date"] == "2025-08-16"
    assert m["kickoff"].startswith("2025-08-16T12:30")
    book = {(o["bookmaker"], o["market"], o["selection"], o["line"], o["is_closing"]): o["price"] for o in odds}
    assert book[("pinnacle", "1x2", "home", 0.0, 0)] == 3.2
    assert book[("pinnacle", "1x2", "away", 0.0, 1)] == 2.3
    assert book[("pinnacle", "ou", "over", 2.5, 1)] == 1.9
    assert book[("pinnacle", "ah", "away", 0.25, 0)] == 1.95


def test_extra_csv_uk_time_maps_to_mexican_local_date():
    matches, odds = football_data.parse_csv(MEX_CSV, LEAGUES["ligamx"], TeamResolver())
    m = matches[0]
    assert m["home_key"] == "pumas" and m["away_key"] == "atletico-san-luis"
    assert m["season"] == "2025"
    assert m["local_date"] == "2025-09-19"  # 03:05 UK = 20:05 the previous evening in Mexico City
    assert {o["is_closing"] for o in odds} == {1}


def test_sources_merge_into_one_match_row():
    store = Store(":memory:")
    league, resolver = LEAGUES["ligamx"], TeamResolver()
    matches, _ = football_data.parse_csv(MEX_CSV, league, resolver)
    for m in matches:
        m["match_id"] = store.resolve_match_id(league.key, m["home_key"], m["away_key"], m["local_date"])
    store.upsert("matches", matches)
    row = api_football.fixture_row(fixture(), league, resolver)
    row["match_id"] = store.resolve_match_id(league.key, row["home_key"], row["away_key"], row["local_date"],
                                             row["api_fixture_id"])
    store.upsert("matches", [row])
    df = store.frame("matches")
    assert len(df) == 1
    assert df.loc[0, "api_fixture_id"] == 101 and df.loc[0, "source"] == "api-football"


def test_fixture_row_uses_90_minute_score():
    row = api_football.fixture_row(fixture(status="AET", goals=(3, 2), fulltime=(2, 2)), LEAGUES["ligamx"],
                                   TeamResolver())
    assert row["status"] == FINISHED and (row["home_goals"], row["away_goals"]) == (2, 2)
    upcoming = api_football.fixture_row(fixture(status="NS", goals=(None, None), fulltime=(None, None)),
                                        LEAGUES["ligamx"], TeamResolver())
    assert upcoming["status"] == "NS" and upcoming["home_goals"] is None


def test_api_transforms():
    resolver = TeamResolver()
    odds_item = {"fixture": {"id": 1}, "bookmakers": [{"name": "Pinnacle", "bets": [
        {"name": "Match Winner", "values": [{"value": "Home", "odd": "2.10"}, {"value": "Draw", "odd": "3.40"},
                                             {"value": "Away", "odd": "3.60"}]},
        {"name": "Asian Handicap", "values": [{"value": "Home -0.25", "odd": "1.95"},
                                               {"value": "Away +0.25", "odd": "1.93"}]},
        {"name": "Goals Over/Under", "values": [{"value": "Over 2.5", "odd": "1.90"}]},
        {"name": "Corners", "values": [{"value": "Over 9.5", "odd": "1.90"}]},
    ]}]}
    rows = api_football.odds_rows(odds_item, "m1", "now")
    keyed = {(r["market"], r["selection"], r["line"]): r["price"] for r in rows}
    assert keyed[("ah", "home", -0.25)] == 1.95
    assert keyed[("ah", "away", -0.25)] == 1.93  # stored with the home handicap
    assert keyed[("ou", "over", 2.5)] == 1.9 and len(rows) == 6

    coaches = [{"id": 7, "name": "New Boss", "career": [
        {"team": {"id": 33}, "start": "2026-09-01", "end": None},
        {"team": {"id": 99}, "start": "2023-07-01", "end": "2026-05-30"}]}]
    assert api_football.coach_rows(coaches, "man-united", 33, "now") == [
        {"team_key": "man-united", "coach_id": 7, "name": "New Boss", "start_date": "2026-09-01", "end_date": None,
         "fetched_at": "now"}]

    injuries = [{"player": {"id": 5, "name": "X", "type": "Questionable", "reason": "Knock"},
                 "team": {"id": 33, "name": "Manchester United"}, "fixture": {"id": 9}}]
    assert api_football.injury_rows(injuries, resolver, "now")[0]["team_key"] == "man-united"

    players = [{"player": {"id": 5, "name": "X", "age": 25}, "statistics": [
        {"team": {"id": 33}, "league": {"id": 39, "season": 2026},
         "games": {"appearences": 6, "lineups": 6, "minutes": 540, "position": "Attacker", "rating": "7.21"},
         "goals": {"total": 4, "assists": 1}},
        {"team": {"id": 33}, "league": {"id": 45, "season": 2026}, "games": {"minutes": 0}}]}]
    prow = api_football.player_rows(players, {33: "man-united"}, "now")
    assert len(prow) == 1 and prow[0]["rating"] == pytest.approx(7.21) and prow[0]["team_key"] == "man-united"

    stats = [{"team": {"id": 20}, "statistics": [{"type": "expected_goals", "value": "1.73"},
                                                 {"type": "Ball Possession", "value": "61%"},
                                                 {"type": "Shots on Goal", "value": 6}]},
             {"team": {"id": 21}, "statistics": [{"type": "expected_goals", "value": "0.41"}]}]
    assert api_football.statistics_update(stats, 20) == {"home_xg": 1.73, "home_sot": 6.0, "away_xg": 0.41}


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self._body, self.status_code, self.headers = body, status, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


class FakeSession:
    def __init__(self, bodies):
        self.bodies, self.calls = list(bodies), 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        return FakeResponse(self.bodies.pop(0), headers={"x-ratelimit-requests-remaining": "50"})


def test_http_cache_budget_and_error_payloads(tmp_path):
    session = FakeSession([{"errors": {"token": "bad key"}}, {"errors": [], "response": [1], "paging": {}},
                           {"errors": [], "response": [2]}])
    client = HttpClient(tmp_path / "cache", session=session, retries=0)
    budget = RequestBudget(tmp_path / "budget.json", daily_limit=2)
    api = api_football.ApiFootball(client, "key", budget)
    with pytest.raises(RuntimeError):
        api.get("status", {}, ttl=3600)
    assert api.get("status", {}, ttl=3600) == [1]  # the error was not cached
    assert api.get("status", {}, ttl=3600) == [1]  # served from cache
    assert session.calls == 2
    with pytest.raises(BudgetExceeded):
        api.get("other", {}, ttl=3600)


def test_weather_window_summary():
    hourly = {"time": ["2026-10-04T13:00", "2026-10-04T14:00", "2026-10-04T15:00", "2026-10-04T16:00"],
              "temperature_2m": [10, 12, 14, 30], "precipitation": [0, 2, 4, 9],
              "wind_speed_10m": [5, 10, 40, 60], "relative_humidity_2m": [70, 80, 90, 99]}
    w = summarize_hourly(hourly, datetime(2026, 10, 4, 14, 0, tzinfo=timezone.utc), "forecast")
    assert w == {"temp_c": 13.0, "precip_mm_h": 3.0, "wind_kmh": 40.0, "humidity": 85.0, "source": "forecast"}


def test_fixtures_file_gives_scheduled_matches_for_one_league():
    text = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,PSH,PSD,PSA\n"
            "E0,04/10/2026,15:00,Arsenal,Liverpool,,,2.4,3.5,2.9\n"
            "SP1,04/10/2026,20:00,Barcelona,Sevilla,,,1.4,5.0,7.0\n")
    matches, odds = football_data.parse_csv(text, LEAGUES["epl"], TeamResolver(), "2026", scheduled=True)
    assert [(m["home_key"], m["status"], m["home_goals"]) for m in matches] == [("arsenal", "NS", None)]
    assert {(o["selection"], o["price"]) for o in odds} == {("home", 2.4), ("draw", 3.5), ("away", 2.9)}


def test_fixtures_file_with_mis_decoded_bom_is_still_filtered():
    text = "ï»¿Div,Date,Time,HomeTeam,AwayTeam\nSP2,04/10/2026,15:00,Eibar,Leganes\n"
    matches, _ = football_data.parse_csv(text, LEAGUES["epl"], TeamResolver(), "2026", scheduled=True)
    assert matches == []
