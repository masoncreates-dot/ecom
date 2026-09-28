"""High-level jobs used by the CLI: update data, predict, settle and track bets."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import report
from .config import LEAGUES, League, Settings
from .data import api_football, football_data, weather
from .dataset import Dataset
from .http import HttpClient, RequestBudget
from .models import markets
from .predict import MatchPrediction, Predictor
from .storage import FINISHED, Store
from .teams import TeamResolver
from .trading.signals import market_probabilities, size_positions

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def update(settings: Settings, leagues: list[League], *, players: bool = True, with_weather: bool = True,
           past_weather: bool = False, now: datetime | None = None) -> dict:
    """Pull history, fixtures, team news, odds and weather into the store.

    ``past_weather`` backfills archived kick-off weather for finished matches
    (one Open-Meteo call per match, cached forever) so backtests can use it."""
    now = now or utcnow()
    store = Store(settings.db_path)
    client = HttpClient(settings.cache_dir)
    resolver = TeamResolver(settings.team_aliases)
    summary: dict = {}
    api = None
    if settings.api_football_key:
        budget = RequestBudget(settings.data_dir / "api_football_budget.json", settings.api_daily_budget)
        api = api_football.ApiFootball(client, settings.api_football_key, budget, settings.api_football_rapidapi)
    else:
        log.warning("API_FOOTBALL_KEY not set: no fixtures, lineups, injuries, transfers, coaches or live odds. "
                    "Only football-data.co.uk history will be loaded.")
    for league in leagues:
        s = summary.setdefault(league.key, {})
        try:
            s["history_matches"] = football_data.load(client, store, league, resolver, settings.history_seasons, now)
        except Exception as exc:  # noqa: BLE001 - keep going with other sources
            log.error("football-data %s failed: %s", league.key, exc)
        if api is not None:
            s.update(api_football.sync(api, store, league, resolver, now=now, horizon_days=settings.horizon_days,
                                       deep=players))
        if with_weather:
            data = Dataset.from_store(store, league)
            s["weather"] = weather.sync(client, store, data.teams, data.matches, now, include_past=past_weather)
    store.close()
    return summary


def predict(settings: Settings, leagues: list[League], *, days: int | None = None, now: datetime | None = None,
            out_dir: Path | None = None, store: Store | None = None, record: bool = True,
            **predictor_kwargs) -> tuple[list[tuple[Predictor, list[MatchPrediction]]], list, dict[str, Path]]:
    now = pd.Timestamp(now or utcnow())
    own_store = store is None
    store = store or Store(settings.db_path)
    resolver = TeamResolver(settings.team_aliases)
    runs, signals = [], []
    for league in leagues:
        data = Dataset.from_store(store, league)
        if data.completed.empty:
            log.warning("no finished %s matches in the store - run `soccer-quant update` first", league.key)
            continue
        predictor = Predictor(data, settings.model, settings.trading, **predictor_kwargs).fit(now)
        preds = predictor.predict_upcoming(days or settings.horizon_days)
        if not preds:
            log.warning("no scheduled %s fixtures in the next %s days (fixtures come from API-Football)",
                        league.key, days or settings.horizon_days)
        runs.append((predictor, preds))
        signals += [s for p in preds for s in p.signals]
    signals = size_positions(signals, settings.trading.bankroll, settings.trading)
    if record:
        _record_signals(store, signals, now)
    out_dir = out_dir or settings.reports_dir / now.strftime("%Y-%m-%d")
    paths = report.write_predictions(runs, signals, out_dir, resolver)
    if own_store:
        store.close()
    return runs, signals, paths


def _record_signals(store: Store, signals, now: pd.Timestamp) -> None:
    rows = []
    for s in signals:
        rows.append({"signal_id": f"{s.match_id}|{s.selection.market}|{s.selection.pick}|{s.selection.line:g}",
                     "created_at": now.isoformat(), "match_id": s.match_id, "kickoff": s.kickoff.isoformat(),
                     "home_key": s.home, "away_key": s.away, "market": s.selection.market,
                     "selection": s.selection.pick, "line": s.selection.line, "bookmaker": s.bookmaker,
                     "price": s.price, "model_prob": s.model_prob, "market_prob": s.market_prob, "ev": s.ev,
                     "stake": s.stake})
    store.upsert("signals", rows)


def settle(settings: Settings, store: Store | None = None) -> pd.DataFrame:
    """Settle recorded bets and measure closing-line value."""
    own = store is None
    store = store or Store(settings.db_path)
    sig = store.frame("signals")
    if sig.empty:
        return sig
    matches = store.frame("matches").set_index("match_id")
    odds = store.frame("odds")
    updates = []
    for r in sig.itertuples():
        if r.match_id not in matches.index:
            continue
        m = matches.loc[r.match_id]
        sel = markets.Selection(r.market, r.selection, float(r.line))
        # Odds rows keep the home handicap on both AH sides.
        stored_line = (-sel.line or 0.0) if (sel.market == "ah" and sel.pick == "away") else sel.line
        o = odds[odds["match_id"] == r.match_id]
        close = o[o["is_closing"] == 1]
        if close.empty:
            close = o[(o["is_closing"] == 0) & (o["captured_at"].fillna("") <= str(m["kickoff"]))]
        fair = market_probabilities(close, settings.trading).get((sel.market, float(stored_line)), {}).get(sel.pick)
        row = {"signal_id": r.signal_id}
        if fair:
            row.update(closing_price=round(1 / fair, 3), clv=round(r.price * fair - 1, 4))
        if m["status"] == FINISHED and pd.notna(m["home_goals"]):
            ret = markets.settle(sel, int(m["home_goals"]), int(m["away_goals"]), r.price,
                                 settings.trading.commission)
            row.update(result=f"{int(m['home_goals'])}-{int(m['away_goals'])}", pnl=round(r.stake * ret, 2))
        updates.append(row)
    store.upsert("signals", updates)
    out = store.frame("signals")
    if own:
        store.close()
    return out


def parse_leagues(value: str) -> list[League]:
    if value == "all":
        return list(LEAGUES.values())
    keys = [v.strip() for v in value.split(",")]
    unknown = [k for k in keys if k not in LEAGUES]
    if unknown:
        raise ValueError(f"unknown league(s) {unknown}; choose from {list(LEAGUES)} or 'all'")
    return [LEAGUES[k] for k in keys]
