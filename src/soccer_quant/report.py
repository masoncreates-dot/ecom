"""Markdown / CSV / JSON output for predictions, bets and backtests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import BacktestResult
from .config import LEAGUES
from .predict import MatchPrediction, Predictor
from .teams import TeamResolver

DISCLAIMER = ("> Model output for research. Betting markets are efficient and most models lose to the closing "
              "line; paper-trade and track closing-line value before staking real money.")


def _pct(x) -> str:
    return "-" if x is None or x != x else f"{100 * x:.0f}%"


def _local(ts: pd.Timestamp, league: str) -> str:
    return ts.tz_convert(ZoneInfo(LEAGUES[league].timezone)).strftime("%a %d %b %H:%M")


def write_predictions(runs: list[tuple[Predictor, list[MatchPrediction]]], signals, out_dir: Path,
                      resolver: TeamResolver) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = resolver.display
    now = datetime.now(timezone.utc)
    lines = [f"# Match predictions - generated {now:%Y-%m-%d %H:%M} UTC", "", DISCLAIMER, ""]
    rows, payload = [], []
    for predictor, preds in runs:
        league = predictor.data.league
        lines += [f"## {league.name}", ""]
        if not preds:
            lines += ["No fixtures in the prediction window.", ""]
            continue
        lines += ["| Kick-off (local) | Match | xG | 1 | X | 2 | Over 2.5 | BTTS | Fair AH (home) | Likely score | Market 1/X/2 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in sorted(preds, key=lambda p: p.kickoff):
            m, mkt = p.markets, p.market_probs.get(("1x2", 0.0), {})
            top = m["top_scores"][0]
            lines.append(
                f"| {_local(p.kickoff, league.key)} | {name(p.home)} v {name(p.away)} | "
                f"{p.final_xg[0]:.2f}-{p.final_xg[1]:.2f} | {_pct(m['1x2']['home'])} | {_pct(m['1x2']['draw'])} | "
                f"{_pct(m['1x2']['away'])} | {_pct(m['over'][2.5])} | {_pct(m['btts']['yes'])} | "
                f"{m['ah_fair_line']:+g} | {top[0]}-{top[1]} ({_pct(top[2])}) | "
                f"{_pct(mkt.get('home'))}/{_pct(mkt.get('draw'))}/{_pct(mkt.get('away'))} |")
            rows.append({"league": league.key, **p.as_row()})
            payload.append(_json_prediction(p))
        lines.append("")
        league_signals = [s for s in signals if any(s.match_id == p.match_id for p in preds)]
        lines += _signal_table(league_signals, league.key, name)
        lines += ["### Match notes", ""]
        for p in sorted(preds, key=lambda p: p.kickoff):
            lines += _match_notes(p, league.key, name)
        lines += _team_sheets(predictor, preds, name)

    paths = {"markdown": out_dir / "predictions.md", "csv": out_dir / "predictions.csv",
             "signals": out_dir / "signals.csv", "json": out_dir / "predictions.json"}
    paths["markdown"].write_text("\n".join(lines))
    pd.DataFrame(rows).to_csv(paths["csv"], index=False)
    pd.DataFrame([s.as_row() for s in signals]).to_csv(paths["signals"], index=False)
    paths["json"].write_text(json.dumps(payload, indent=2, default=str))
    return paths


def _signal_table(signals, league: str, name) -> list[str]:
    lines = ["### Value bets", ""]
    staked = [s for s in signals if s.stake > 0]
    if not staked:
        return lines + ["No bets clear the edge, probability and exposure filters.", ""]
    lines += ["| Kick-off | Match | Bet | Book | Price | Fair | Model | Market | EV | Stake |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in staked:
        lines.append(f"| {_local(s.kickoff, league)} | {name(s.home)} v {name(s.away)} | {s.selection.label()} | "
                     f"{s.bookmaker} | {s.price:.2f} | {s.fair_odds:.2f} | {_pct(s.model_prob)} | "
                     f"{_pct(s.market_prob)} | {100 * s.ev:+.1f}% | {s.stake:.2f} |")
    return lines + [""]


def _match_notes(p: MatchPrediction, league: str, name) -> list[str]:
    m = p.markets
    ctx = p.context
    out = [f"#### {name(p.home)} v {name(p.away)} - {_local(p.kickoff, league)}", ""]
    out.append(f"- Expected goals: ratings {p.base_xg[0]:.2f}-{p.base_xg[1]:.2f} -> after adjustments "
               f"{p.model_xg[0]:.2f}-{p.model_xg[1]:.2f} -> final {p.final_xg[0]:.2f}-{p.final_xg[1]:.2f}")
    venue = ctx.get("venue") or {}
    if venue.get("stadium"):
        out.append(f"- Venue: {venue['stadium']}, {venue.get('city')} ({venue.get('elevation_m') or 0:.0f} m)")
    out += ["", "| Factor | Home x | Away x | Detail |", "|---|---|---|---|"]
    for a in p.adjustments:
        out.append(f"| {a.name} | {a.home_mult:.3f} | {a.away_mult:.3f} | {a.detail} |")
    out.append("")
    for team, adj in p.players.items():
        if adj.missing:
            missing = ", ".join(f"{pl.name} ({pl.position}, {status}{': ' + reason if reason else ''}, "
                                f"player value {pl.value:+.2f})" for pl, status, reason in adj.missing)
            out.append(f"- {name(team)} team news: {missing}")
        signings = [pl.name for pl in adj.expected_xi if pl.new_signing]
        if signings:
            out.append(f"- {name(team)} new signings in projected XI: {', '.join(signings)}")
    for team, coach in (ctx.get("coaches") or {}).items():
        if coach:
            out.append(f"- {name(team)} coach: {coach[0]} since {coach[1]} ({coach[2]} days)")
    h2h = ctx.get("h2h") or {}
    if h2h.get("n"):
        recent = "; ".join(f"{d} {name(h)} {hg}-{ag} {name(a)}" for d, h, hg, ag, a in h2h["last"])
        out.append(f"- Head to head ({name(p.home)}'s view): W{h2h['wins']} D{h2h['draws']} L{h2h['losses']}, "
                   f"goals {h2h['goals_for']:.0f}-{h2h['goals_against']:.0f}; recent: {recent}")
    scores = ", ".join(f"{h}-{a} {_pct(pr)}" for h, a, pr in m["top_scores"][:5])
    out.append(f"- Likely scores: {scores}")
    out.append(f"- Goals: O1.5 {_pct(m['over'][1.5])}, O2.5 {_pct(m['over'][2.5])}, O3.5 {_pct(m['over'][3.5])}; "
               f"BTTS {_pct(m['btts']['yes'])}; clean sheet {name(p.home)} {_pct(m['home_clean_sheet'])}, "
               f"{name(p.away)} {_pct(m['away_clean_sheet'])}; fair total line {m['total_fair_line']:g}")
    out.append("")
    return out


def _team_sheets(predictor: Predictor, preds: list[MatchPrediction], name) -> list[str]:
    if predictor.players is None:
        return []
    out = ["### Squad impact ratings", "",
           "Value = match rating above a replacement-level player at the position (league-adjusted).", ""]
    api_ids = predictor.data.team_api_ids()
    news: dict[str, dict] = {}
    for p in sorted(preds, key=lambda p: p.kickoff, reverse=True):  # next fixture's news wins
        for team, adj in p.players.items():
            news[team] = {pl.player_id: (status, reason) for pl, status, reason in adj.missing}
    for team in sorted({t for p in preds for t in (p.home, p.away)}):
        sheet = predictor.players.team_sheet(team, news.get(team))
        if sheet.empty:
            continue
        top = sheet.head(8)
        out += [f"**{name(team)}**", "", "| Player | Pos | Rating | Value | Minutes | Status |", "|---|---|---|---|---|---|"]
        for r in top.itertuples():
            tag = " (new)" if r.new_signing else ""
            rating = "-" if r.rating is None or r.rating != r.rating else f"{r.rating:.2f}"
            out.append(f"| {r.player}{tag} | {r.position} | {rating} | {r.value:+.2f} | {r.minutes} | {r.status} |")
        moves = recent_transfers(predictor.data.transfers, api_ids.get(team), predictor.as_of)
        if moves:
            out += ["", "Recent moves: " + "; ".join(moves)]
        out.append("")
    return out


def recent_transfers(transfers: pd.DataFrame, api_id: int | None, as_of: pd.Timestamp, days: int = 150) -> list[str]:
    if api_id is None or transfers.empty:
        return []
    cutoff = (as_of - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    t = transfers[(transfers["date"].fillna("") >= cutoff)
                  & ((transfers["team_in_api_id"] == api_id) | (transfers["team_out_api_id"] == api_id))]
    moves = []
    for r in t.sort_values("date", ascending=False).itertuples():
        arrow = f"in from {r.team_out_name}" if r.team_in_api_id == api_id else f"out to {r.team_in_name}"
        moves.append(f"{r.player_name} {arrow} ({r.type or 'transfer'}, {r.date})")
    return moves


def _json_prediction(p: MatchPrediction) -> dict:
    return {
        **p.as_row(),
        "adjustments": [{"name": a.name, "home_mult": a.home_mult, "away_mult": a.away_mult, "detail": a.detail}
                        for a in p.adjustments],
        "markets": {k: v for k, v in p.markets.items() if k != "top_scores"},
        "top_scores": p.markets["top_scores"],
        "signals": [s.as_row() for s in p.signals],
        "context": {k: v for k, v in p.context.items() if k in ("weather", "travel_km", "rest_days", "coaches", "h2h")},
    }


def write_backtest(result: BacktestResult, league: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(out_dir / f"backtest_{league}_predictions.csv", index=False)
    result.bets.to_csv(out_dir / f"backtest_{league}_bets.csv", index=False)
    result.equity.to_csv(out_dir / f"backtest_{league}_equity.csv", index=False)
    m = result.metrics
    lines = [f"# Backtest - {LEAGUES[league].name}", "", DISCLAIMER, "", "## Forecast quality (1X2)", "",
             "| Source | Matches | Log loss | RPS | Brier |", "|---|---|---|---|---|"]
    labels = {"model": "Model", "final": "Model + market blend", "pre": "Market (pre-match, de-vigged)",
              "close": "Market (closing, de-vigged)"}
    for key, label in labels.items():
        if f"{key}_log_loss" in m:
            lines.append(f"| {label} | {m[f'{key}_n']} | {m[f'{key}_log_loss']:.4f} | {m[f'{key}_rps']:.4f} | "
                         f"{m[f'{key}_brier']:.4f} |")
    if "model_vs_close_log_loss" in m:
        lines += ["", f"Model minus closing-line log loss: {m['model_vs_close_log_loss']:+.4f} "
                      "(negative would mean the model beats the closing market)."]
    lines += ["", "## Betting", ""]
    if m.get("bets"):
        clv = "-" if m.get("avg_clv") is None else f"{100 * m['avg_clv']:+.2f}%"
        lines += [f"- Bets: {m['bets']}, staked {m['staked']:.2f}, P&L {m['pnl']:+.2f}, ROI {100 * m['roi']:+.2f}%",
                  f"- Hit rate {100 * m['hit_rate']:.1f}%, average model EV {100 * m['avg_ev']:+.2f}%",
                  f"- Average closing-line value {clv} (share of bets beating the close: "
                  f"{_pct(m.get('positive_clv_share'))})",
                  f"- Final bankroll {m['final_bankroll']:.2f}, max drawdown {100 * m['max_drawdown']:.1f}%"]
    else:
        lines.append("No bets placed.")
    if not result.calibration.empty:
        lines += ["", "## Calibration (final probabilities)", "", "| Bin | n | Predicted | Observed |", "|---|---|---|---|"]
        for r in result.calibration.itertuples():
            lines.append(f"| {r.bin} | {r.n} | {r.predicted:.3f} | {r.observed:.3f} |")
    path = out_dir / f"backtest_{league}.md"
    path.write_text("\n".join(lines) + "\n")
    return path
