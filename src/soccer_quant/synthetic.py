"""Synthetic league generator.

Fills a Store with fake but internally consistent data - results, match
stats, bookmaker odds, players, injuries, a transfer, a coaching change,
weather - so the full pipeline can be run and tested offline. Real club
names and stadiums are used so venue effects (Liga MX altitude) show up,
but every number is simulated.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import League
from .storage import FINISHED, SCHEDULED, Store, make_match_id
from .venues import VENUES

EPL_TEAMS = ["arsenal", "aston-villa", "bournemouth", "brentford", "brighton", "burnley", "chelsea",
             "crystal-palace", "everton", "fulham", "leeds", "liverpool", "man-city", "man-united",
             "newcastle", "nottingham-forest", "sunderland", "tottenham", "west-ham", "wolves"]
LIGAMX_TEAMS = ["america", "guadalajara", "cruz-azul", "pumas", "tigres", "monterrey", "toluca", "pachuca",
                "leon", "santos-laguna", "atlas", "puebla", "queretaro", "necaxa", "tijuana", "juarez",
                "atletico-san-luis", "atlante"]
POSITIONS = ["Goalkeeper"] * 3 + ["Defender"] * 8 + ["Midfielder"] * 8 + ["Attacker"] * 6


def _round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    t = list(teams)
    rounds = []
    for r in range(len(t) - 1):
        pairs = [(t[i], t[-1 - i]) for i in range(len(t) // 2)]
        rounds.append([(a, b) if (r + i) % 2 == 0 else (b, a) for i, (a, b) in enumerate(pairs)])
        t = [t[0], t[-1], *t[1:-1]]
    return rounds


def _probs_1x2(lam: float, mu: float) -> np.ndarray:
    g = np.arange(11)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    i, j = np.indices(m.shape)
    p = np.array([m[i > j].sum(), np.trace(m), m[i < j].sum()])
    over = m[(i + j) > 2].sum()
    return p / p.sum(), over


def _quote(probs: np.ndarray, margin: float, noise: float, rng) -> np.ndarray:
    logits = np.log(probs) + rng.normal(0, noise, len(probs))
    p = np.exp(logits) / np.exp(logits).sum()
    return np.round(1.0 / (p * (1 + margin)), 2)


def build(store: Store, league: League, *, today: pd.Timestamp, seasons: int = 3, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    teams = LIGAMX_TEAMS if league.key == "ligamx" else EPL_TEAMS
    n = len(teams)
    att = dict(zip(teams, rng.normal(0, 0.22, n)))
    dfn = dict(zip(teams, rng.normal(0, 0.22, n)))
    home_adv = 0.27
    api_ids = {t: 1000 + i for i, t in enumerate(teams)}
    store.upsert("teams", [{"team_key": t, "league": league.key, "name": t.replace("-", " ").title(),
                            "api_id": api_ids[t]} for t in teams])

    rounds = _round_robin(teams)
    rounds = rounds + [[(b, a) for a, b in r] for r in rounds]
    today = pd.Timestamp(today)
    first_season = league.season_for(today) - seasons
    matches, odds = [], []
    fixture_id = 500000
    for season in range(first_season, first_season + seasons + 1):
        start = pd.Timestamp(f"{season}-08-09 14:00", tz="UTC")
        for r, pairs in enumerate(rounds):
            kickoff = start + timedelta(days=7 * r + (r >= 19) * 21)  # winter break
            for team in teams:  # slow drift in team strength
                att[team] += rng.normal(0, 0.012)
                dfn[team] += rng.normal(0, 0.012)
            if kickoff > today + timedelta(days=14):
                break
            for k, (h, a) in enumerate(pairs):
                ko = kickoff + timedelta(hours=3 * (k % 3))
                lam = np.exp(0.12 + home_adv + att[h] - dfn[a] + _altitude(h, a))
                mu = np.exp(0.12 + att[a] - dfn[h])
                fixture_id += 1
                local_date = ko.tz_convert(league.timezone).date().isoformat()
                match_id = make_match_id(league.key, local_date, h, a)
                row = {"match_id": match_id, "league": league.key, "season": str(season), "kickoff": ko.isoformat(),
                       "local_date": local_date, "home_key": h, "away_key": a, "api_fixture_id": fixture_id,
                       "source": "synthetic"}
                probs, p_over = _probs_1x2(lam, mu)
                played = ko < today
                if played:
                    hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
                    hs, as_ = int(rng.poisson(8 * lam + 5)), int(rng.poisson(8 * mu + 5))
                    row.update(status=FINISHED, home_goals=hg, away_goals=ag, home_shots=hs, away_shots=as_,
                               home_sot=int(rng.binomial(hs, 0.34)), away_sot=int(rng.binomial(as_, 0.34)),
                               home_corners=int(rng.poisson(3 * lam + 2)), away_corners=int(rng.poisson(3 * mu + 2)),
                               home_xg=round(float(lam * rng.gamma(20, 1 / 20)), 2) if season == first_season + seasons else None,
                               away_xg=round(float(mu * rng.gamma(20, 1 / 20)), 2) if season == first_season + seasons else None)
                else:
                    row["status"] = SCHEDULED
                matches.append(row)
                odds += _odds_rows(match_id, probs, p_over, lam - mu, rng, closing=played,
                                   captured=(today - timedelta(hours=2)).isoformat())
    store.upsert("matches", matches)
    store.upsert("odds", odds)
    truth = {"attack": att, "defence": dfn, "home_advantage": home_adv}
    _players_and_news(store, league, teams, api_ids, att, dfn, today, rng)
    return truth


def _altitude(home: str, away: str) -> float:
    vh, va = VENUES.get(home), VENUES.get(away)
    if not vh or not va:
        return 0.0
    return 0.05 * max(0.0, (vh.elevation_m - va.elevation_m) / 1000 - 0.5)


def _odds_rows(match_id, probs, p_over, supremacy, rng, *, closing: bool, captured: str) -> list[dict]:
    rows = []
    books = [("pinnacle", 0.025, 0.03), ("bet365", 0.05, 0.05), ("avg", 0.055, 0.03)]
    for book, margin, noise in books:
        pre = _quote(probs, margin, noise, rng)
        for sel, price in zip(("home", "draw", "away"), pre):
            rows.append({"match_id": match_id, "bookmaker": book, "market": "1x2", "selection": sel, "line": 0.0,
                         "price": float(price), "is_closing": 0, "captured_at": captured})
        ou = _quote(np.array([p_over, 1 - p_over]), margin, noise, rng)
        for sel, price in zip(("over", "under"), ou):
            rows.append({"match_id": match_id, "bookmaker": book, "market": "ou", "selection": sel, "line": 2.5,
                         "price": float(price), "is_closing": 0, "captured_at": captured})
        line = -round(supremacy * 4) / 4 if abs(supremacy) > 0.2 else 0.0
        ah = _quote(np.array([0.5, 0.5]), margin, noise, rng)
        for sel, price in zip(("home", "away"), ah):
            rows.append({"match_id": match_id, "bookmaker": book, "market": "ah", "selection": sel, "line": line,
                         "price": float(price), "is_closing": 0, "captured_at": captured})
        if closing:
            close = _quote(probs, margin, noise / 2, rng)
            for sel, price in zip(("home", "draw", "away"), close):
                rows.append({"match_id": match_id, "bookmaker": book, "market": "1x2", "selection": sel,
                             "line": 0.0, "price": float(price), "is_closing": 1, "captured_at": None})
    return rows


def _players_and_news(store, league, teams, api_ids, att, dfn, today, rng) -> None:
    season = league.season_for(today)
    players, squads, pid = [], [], 10000
    stars: dict[str, list[int]] = {}
    for team in teams:
        strength = att[team] + dfn[team]
        for k, pos in enumerate(POSITIONS):
            pid += 1
            starter = k in (0, 3, 4, 5, 6, 11, 12, 13, 19, 20, 21)
            base = 6.75 + 0.8 * strength + (0.25 if starter else -0.2) + rng.normal(0, 0.18)
            squads.append({"team_key": team, "player_id": pid, "name": f"{team[:3].upper()} {pos[:3]} {k + 1}",
                           "position": pos, "fetched_at": today.isoformat()})
            for s, games in ((season - 1, 38), (season, 7)):
                mins = int(games * 90 * (rng.uniform(0.75, 0.97) if starter else rng.uniform(0.02, 0.3)))
                players.append({"player_id": pid, "season": s, "team_api_id": api_ids[team],
                                "league_api_id": league.api_football_id, "team_key": team,
                                "name": f"{team[:3].upper()} {pos[:3]} {k + 1}", "position": pos,
                                "appearances": max(1, mins // 75), "lineups": int(mins / 90 * 0.95) if starter else 0,
                                "minutes": max(mins, 1), "rating": round(base + rng.normal(0, 0.05), 3)})
            if starter and pos == "Attacker":
                stars.setdefault(team, []).append(pid)
    # A star signing arrives from La Liga for the second-ranked team.
    buyer = sorted(teams, key=lambda t: -(att[t] + dfn[t]))[1]
    pid += 1
    signing_date = (today - timedelta(days=40)).date().isoformat()
    squads.append({"team_key": buyer, "player_id": pid, "name": "New Signing (from La Liga)", "position": "Attacker",
                   "fetched_at": today.isoformat()})
    players.append({"player_id": pid, "season": season - 1, "team_api_id": 9999, "league_api_id": 140,
                    "name": "New Signing (from La Liga)", "position": "Attacker", "appearances": 34, "lineups": 33,
                    "minutes": 2900, "rating": 7.45})
    store.upsert("transfers", [{"player_id": pid, "player_name": "New Signing (from La Liga)", "date": signing_date,
                                "type": "Transfer", "team_in_api_id": api_ids[buyer], "team_in_name": buyer,
                                "team_out_api_id": 9999, "team_out_name": "Spanish club"}])
    store.upsert("players", players)
    store.upsert("squads", squads)

    # Injuries: the best forward of the three strongest teams misses the next fixture.
    upcoming = store.frame("matches", "league=? AND status=?", (league.key, SCHEDULED))
    injuries = []
    for team in sorted(teams, key=lambda t: -(att[t] + dfn[t]))[:3]:
        fx = upcoming[(upcoming["home_key"] == team) | (upcoming["away_key"] == team)].sort_values("kickoff")
        if fx.empty or team not in stars:
            continue
        injuries.append({"fixture_api_id": int(fx["api_fixture_id"].iloc[0]), "team_key": team,
                         "player_id": stars[team][0], "player_name": f"{team[:3].upper()} Att",
                         "type": "Missing Fixture", "reason": "Hamstring", "fetched_at": today.isoformat()})
    store.upsert("injuries", injuries)

    # Coaching change: the weakest team sacked its coach three weeks ago.
    sacked = sorted(teams, key=lambda t: att[t] + dfn[t])[0]
    store.upsert("coaches", [
        {"team_key": sacked, "coach_id": 1, "name": "Old Coach", "start_date": f"{season - 2}-07-01",
         "end_date": (today - timedelta(days=22)).date().isoformat()},
        {"team_key": sacked, "coach_id": 2, "name": "New Coach", "start_date": (today - timedelta(days=21)).date().isoformat(),
         "end_date": None},
    ])

    # Weather: the first upcoming fixture is played in a storm.
    weather = []
    for k, row in enumerate(upcoming.sort_values("kickoff").itertuples()):
        storm = k == 0
        weather.append({"match_id": row.match_id, "temp_c": 11.0 if storm else float(rng.uniform(12, 24)),
                        "precip_mm_h": 5.5 if storm else 0.0, "wind_kmh": 48.0 if storm else float(rng.uniform(5, 20)),
                        "humidity": 80.0, "source": "synthetic", "fetched_at": today.isoformat()})
    store.upsert("weather", weather)
