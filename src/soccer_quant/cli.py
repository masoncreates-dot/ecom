"""Command line interface: ``soccer-quant <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import backtest, pipeline, report, synthetic
from .config import LEAGUES, Settings
from .dataset import Dataset
from .predict import Predictor
from .storage import Store
from .teams import TeamResolver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soccer-quant", description=__doc__)
    parser.add_argument("--config", help="path to soccer_quant.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def league_arg(p, default="all"):
        p.add_argument("--league", default=default, help="epl, ligamx, or all")

    p = sub.add_parser("update", help="download history, fixtures, team news, odds and weather")
    league_arg(p)
    p.add_argument("--no-players", action="store_true", help="skip squads/players/coaches/transfers (saves API calls)")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--past-weather", action="store_true", help="backfill historical kick-off weather for backtests")

    for name, text in (("predict", "price upcoming fixtures and find value bets"),
                       ("run", "update then predict")):
        p = sub.add_parser(name, help=text)
        league_arg(p)
        p.add_argument("--days", type=int, help="prediction window in days")
        p.add_argument("--bankroll", type=float)
        p.add_argument("--market-weight", type=float, help="0 = pure model, 1 = pure market")
        p.add_argument("--out", type=Path, help="report directory")
        p.add_argument("--no-players", action="store_true")
        p.add_argument("--no-weather", action="store_true")

    p = sub.add_parser("backtest", help="walk-forward backtest against historical odds")
    league_arg(p, "epl")
    p.add_argument("--start", help="first date to predict (default: two seasons ago)")
    p.add_argument("--refit-days", type=int, default=7)
    p.add_argument("--market-weight", type=float)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("tune", help="grid-search model parameters on walk-forward log loss")
    league_arg(p, "epl")
    p.add_argument("--start")

    sub.add_parser("settle", help="settle recorded bets and report closing-line value")

    p = sub.add_parser("ratings", help="current team ratings")
    league_arg(p, "epl")

    p = sub.add_parser("team", help="squad impact sheet for one team")
    p.add_argument("name")

    p = sub.add_parser("demo", help="run the whole pipeline on synthetic data (no network needed)")
    league_arg(p, "all")
    p.add_argument("--out", type=Path, default=Path("reports/demo"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load(args.config)
    try:
        return COMMANDS[args.command](args, settings) or 0
    except (ValueError, RuntimeError) as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _trading(settings: Settings, args) -> Settings:
    if getattr(args, "bankroll", None):
        settings = replace(settings, trading=replace(settings.trading, bankroll=args.bankroll))
    return settings


def _predictor_kwargs(args) -> dict:
    kw = {}
    if getattr(args, "market_weight", None) is not None:
        kw["market_weight"] = args.market_weight
    if getattr(args, "no_players", False):
        kw["use_players"] = False
    if getattr(args, "no_weather", False):
        kw["use_weather"] = False
    return kw


def cmd_update(args, settings):
    summary = pipeline.update(settings, pipeline.parse_leagues(args.league), players=not args.no_players,
                              with_weather=not args.no_weather, past_weather=getattr(args, "past_weather", False))
    print(json.dumps(summary, indent=2))


def cmd_predict(args, settings):
    settings = _trading(settings, args)
    runs, signals, paths = pipeline.predict(settings, pipeline.parse_leagues(args.league), days=args.days,
                                            out_dir=args.out, **_predictor_kwargs(args))
    _print_summary(runs, signals)
    print(f"\nReport: {paths['markdown']}")


def cmd_run(args, settings):
    cmd_update(argparse.Namespace(league=args.league, no_players=args.no_players, no_weather=args.no_weather),
               settings)
    cmd_predict(args, settings)


def _print_summary(runs, signals):
    resolver = TeamResolver()
    for predictor, preds in runs:
        print(f"\n{predictor.data.league.name}")
        for p in sorted(preds, key=lambda p: p.kickoff):
            m = p.markets["1x2"]
            print(f"  {p.kickoff:%a %d %b %H:%M}Z  {resolver.display(p.home):>18} v {resolver.display(p.away):<18}"
                  f" xG {p.final_xg[0]:.2f}-{p.final_xg[1]:.2f}  1 {m['home']:.0%}  X {m['draw']:.0%}  2 {m['away']:.0%}"
                  f"  O2.5 {p.markets['over'][2.5]:.0%}")
    staked = [s for s in signals if s.stake > 0]
    print(f"\n{len(staked)} value bet(s)")
    for s in staked:
        print(f"  {resolver.display(s.home)} v {resolver.display(s.away)}: {s.selection.label()} @ {s.price:.2f} "
              f"({s.bookmaker}) EV {s.ev:+.1%} stake {s.stake:.2f}")


def cmd_backtest(args, settings):
    store = Store(settings.db_path)
    for league in pipeline.parse_leagues(args.league):
        data = Dataset.from_store(store, league)
        start = pd.Timestamp(args.start, tz="UTC") if args.start else _default_start(data)
        kw = _predictor_kwargs(args)
        res = backtest.run(data, settings.model, settings.trading, start=start, refit_days=args.refit_days, **kw)
        path = report.write_backtest(res, league.key, args.out or settings.reports_dir / "backtests")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.metrics.items()}, indent=2))
        print(f"Report: {path}")


def _default_start(data: Dataset) -> pd.Timestamp:
    done = data.completed
    if done.empty:
        raise SystemExit("no finished matches in the store - run `soccer-quant update` first")
    return max(done["kickoff"].min() + pd.Timedelta(days=400), done["kickoff"].max() - pd.Timedelta(days=730))


def cmd_tune(args, settings):
    store = Store(settings.db_path)
    for league in pipeline.parse_leagues(args.league):
        data = Dataset.from_store(store, league)
        start = pd.Timestamp(args.start, tz="UTC") if args.start else _default_start(data)
        print(backtest.tune(data, settings.model, settings.trading, start=start).to_string())


def cmd_settle(args, settings):
    df = pipeline.settle(settings)
    if df.empty:
        print("No recorded bets yet.")
        return
    settled = df[df["pnl"].notna()]
    print(df[["kickoff", "home_key", "away_key", "market", "selection", "line", "price", "stake", "closing_price",
              "clv", "result", "pnl"]].to_string(index=False))
    if not settled.empty:
        print(f"\nSettled {len(settled)}: staked {settled['stake'].sum():.2f}, P&L {settled['pnl'].sum():+.2f}")
    if df["clv"].notna().any():
        print(f"Average CLV {df['clv'].mean():+.2%} over {df['clv'].notna().sum()} bets")


def cmd_ratings(args, settings):
    store = Store(settings.db_path)
    resolver = TeamResolver(settings.team_aliases)
    for league in pipeline.parse_leagues(args.league):
        pred = Predictor(Dataset.from_store(store, league), settings.model, settings.trading,
                         use_players=False).fit(pd.Timestamp.now(tz="UTC"))
        rows = [{"team": resolver.display(t), "attack": a, "defence": d, "overall": a + d,
                 "elo": pred.elo.rating(t) if pred.elo else None} for t, a, d in pred.dc.table()]
        print(f"\n{league.name} (home advantage x{math.exp(pred.dc.home_advantage):.2f} goals)")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def cmd_team(args, settings):
    store = Store(settings.db_path)
    resolver = TeamResolver(settings.team_aliases)
    key = resolver.resolve(args.name)
    league = LEAGUES.get(resolver.league_of(key) or "epl")
    data = Dataset.from_store(store, league)
    pred = Predictor(data, settings.model, settings.trading).fit(pd.Timestamp.now(tz="UTC"))
    if pred.players is None:
        raise SystemExit("no player data - run `soccer-quant update` with an API_FOOTBALL_KEY")
    print(pred.players.team_sheet(key).to_string())


def cmd_demo(args, settings):
    """Synthetic end-to-end run: build a fake league, predict, backtest, report."""
    today = pd.Timestamp.now(tz="UTC").floor("h")
    args.out.mkdir(parents=True, exist_ok=True)
    db = args.out / "demo.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    store = Store(db)
    for league in pipeline.parse_leagues(args.league):
        synthetic.build(store, league, today=today)
    demo_settings = replace(settings, reports_dir=args.out)
    runs, signals, paths = pipeline.predict(demo_settings, pipeline.parse_leagues(args.league), days=14,
                                            now=today, out_dir=args.out, store=store)
    _print_summary(runs, signals)
    for league in pipeline.parse_leagues(args.league):
        data = Dataset.from_store(store, league)
        res = backtest.run(data, settings.model, settings.trading, start=today - pd.Timedelta(days=400),
                           refit_days=14)
        report.write_backtest(res, league.key, args.out)
        m = res.metrics
        print(f"\n{league.name} synthetic backtest: model log loss {m.get('model_log_loss', float('nan')):.4f} vs "
              f"closing market {m.get('close_log_loss', float('nan')):.4f}; {m.get('bets', 0)} bets, "
              f"ROI {100 * m.get('roi', 0):+.1f}%")
    print(f"\nDemo report (synthetic data): {paths['markdown']}")


COMMANDS = {"update": cmd_update, "predict": cmd_predict, "run": cmd_run, "backtest": cmd_backtest,
            "tune": cmd_tune, "settle": cmd_settle, "ratings": cmd_ratings, "team": cmd_team, "demo": cmd_demo}

if __name__ == "__main__":
    sys.exit(main())
