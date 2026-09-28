"""Walk-forward backtest: refit on data available at each date, predict the
next window, bet against pre-match prices, settle, repeat.

The honest scorecard is the comparison with the de-vigged *closing* line: a
model that doesn't beat the closing market on log loss, or doesn't get
positive closing-line value (CLV) on its bets, has no edge no matter what
the ROI of a small sample says.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, replace

import pandas as pd

from . import metrics
from .config import ModelParams, TradingParams
from .dataset import Dataset
from .models import markets
from .predict import Predictor
from .trading.signals import market_probabilities, size_positions

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    bets: pd.DataFrame
    metrics: dict
    calibration: pd.DataFrame
    equity: pd.DataFrame


def run(data: Dataset, model: ModelParams, trading: TradingParams, *, start: pd.Timestamp,
        end: pd.Timestamp | None = None, refit_days: int = 7, bet: bool = True, **predictor_kwargs) -> BacktestResult:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else data.matches["kickoff"].max() + pd.Timedelta(days=1)
    predictor_kwargs.setdefault("use_players", False)  # historical team news is not in the store
    predictor = Predictor(data, model, trading, **predictor_kwargs)
    done = data.completed
    bankroll = trading.bankroll
    pred_rows, bet_rows, equity = [], [], [{"date": start, "bankroll": bankroll}]

    t = start
    while t < end:
        window = done[(done["kickoff"] >= t) & (done["kickoff"] < t + pd.Timedelta(days=refit_days))]
        if window.empty:
            t += pd.Timedelta(days=refit_days)
            continue
        try:
            predictor.fit(t)
        except ValueError as exc:
            log.info("skip %s: %s", t.date(), exc)
            t += pd.Timedelta(days=refit_days)
            continue
        signals = []
        scores = {}
        for _, fx in window.iterrows():
            pred = predictor.predict(fx)
            closing = data.odds[(data.odds["match_id"] == fx["match_id"]) & (data.odds["is_closing"] == 1)]
            close_probs = market_probabilities(closing, trading)
            pred_rows.append(_prediction_row(pred, fx, close_probs))
            scores[fx["match_id"]] = (int(fx["home_goals"]), int(fx["away_goals"]))
            signals += pred.signals
        if bet:
            for s in size_positions(signals, bankroll, trading):
                hg, ag = scores[s.match_id]
                pnl = s.stake * markets.settle(s.selection, hg, ag, s.price, trading.commission)
                close = _closing_prob(data, s, trading)
                bet_rows.append({**s.as_row(), "score": f"{hg}-{ag}", "pnl": round(pnl, 2),
                                 "closing_fair_prob": close,
                                 "clv": None if close is None else s.price * close - 1.0})
                bankroll += pnl
        equity.append({"date": t + pd.Timedelta(days=refit_days), "bankroll": bankroll})
        t += pd.Timedelta(days=refit_days)

    preds = pd.DataFrame(pred_rows)
    bets = pd.DataFrame(bet_rows)
    eq = pd.DataFrame(equity)
    return BacktestResult(preds, bets, _summary(preds, bets, eq, trading), _calibration(preds), eq)


def _prediction_row(pred, fx, close_probs) -> dict:
    close = close_probs.get(("1x2", 0.0), {})
    pre = pred.market_probs.get(("1x2", 0.0), {})
    return {
        "match_id": pred.match_id, "kickoff": pred.kickoff, "home": pred.home, "away": pred.away,
        "home_goals": fx["home_goals"], "away_goals": fx["away_goals"],
        "model_home": pred.model_markets["1x2"]["home"], "model_draw": pred.model_markets["1x2"]["draw"],
        "model_away": pred.model_markets["1x2"]["away"],
        "final_home": pred.markets["1x2"]["home"], "final_draw": pred.markets["1x2"]["draw"],
        "final_away": pred.markets["1x2"]["away"],
        "pre_home": pre.get("home"), "pre_draw": pre.get("draw"), "pre_away": pre.get("away"),
        "close_home": close.get("home"), "close_draw": close.get("draw"), "close_away": close.get("away"),
        "model_over25": pred.model_markets["over"][2.5],
        "home_xg": pred.final_xg[0], "away_xg": pred.final_xg[1],
    }


def _closing_prob(data: Dataset, s, trading: TradingParams) -> float | None:
    closing = data.odds[(data.odds["match_id"] == s.match_id) & (data.odds["is_closing"] == 1)]
    probs = market_probabilities(closing, trading)
    line = -s.selection.line if (s.selection.market == "ah" and s.selection.pick == "away") else s.selection.line
    return probs.get((s.selection.market, float(line)), {}).get(s.selection.pick)


def _summary(preds: pd.DataFrame, bets: pd.DataFrame, equity: pd.DataFrame, trading: TradingParams) -> dict:
    out: dict = {"matches": len(preds)}
    if preds.empty:
        return out
    y = metrics.outcome_index(preds["home_goals"], preds["away_goals"])
    for name in ("model", "final", "pre", "close"):
        cols = [f"{name}_home", f"{name}_draw", f"{name}_away"]
        mask = preds[cols].notna().all(axis=1).to_numpy()
        if mask.sum() == 0:
            continue
        p = preds.loc[mask, cols].to_numpy(float)
        out[f"{name}_log_loss"] = metrics.log_loss(p, y[mask])
        out[f"{name}_rps"] = metrics.rps(p, y[mask])
        out[f"{name}_brier"] = metrics.brier(p, y[mask])
        out[f"{name}_n"] = int(mask.sum())
    both = preds[["model_home", "close_home"]].notna().all(axis=1).to_numpy()
    if both.any():
        m = preds.loc[both, ["model_home", "model_draw", "model_away"]].to_numpy(float)
        c = preds.loc[both, ["close_home", "close_draw", "close_away"]].to_numpy(float)
        out["model_vs_close_log_loss"] = metrics.log_loss(m, y[both]) - metrics.log_loss(c, y[both])
    if not bets.empty:
        staked = bets["stake"].sum()
        out.update({
            "bets": len(bets), "staked": float(staked), "pnl": float(bets["pnl"].sum()),
            "roi": float(bets["pnl"].sum() / staked) if staked else 0.0,
            "hit_rate": float((bets["pnl"] > 0).mean()), "avg_ev": float(bets["ev"].mean()),
            "avg_clv": float(bets["clv"].dropna().mean()) if bets["clv"].notna().any() else None,
            "positive_clv_share": float((bets["clv"].dropna() > 0).mean()) if bets["clv"].notna().any() else None,
            "final_bankroll": float(equity["bankroll"].iloc[-1]),
            "max_drawdown": metrics.max_drawdown(equity["bankroll"].to_numpy()),
        })
    else:
        out["bets"] = 0
    return out


def _calibration(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    y = metrics.outcome_index(preds["home_goals"], preds["away_goals"])
    return metrics.calibration(preds[["final_home", "final_draw", "final_away"]].to_numpy(float), y)


def tune(data: Dataset, model: ModelParams, trading: TradingParams, *, start: pd.Timestamp,
         grid: dict[str, list] | None = None, refit_days: int = 14) -> pd.DataFrame:
    """Grid search model parameters on walk-forward log loss (pure model, no market blend)."""
    grid = grid or {"time_decay_xi": [0.001, 0.0019, 0.003], "sot_weight": [0.0, 0.3, 0.5],
                    "elo_weight": [0.0, 0.25, 0.5]}
    rows = []
    for values in itertools.product(*grid.values()):
        params = replace(model, **dict(zip(grid.keys(), values)), market_weight=0.0)
        res = run(data, params, trading, start=start, refit_days=refit_days, bet=False)
        rows.append({**dict(zip(grid.keys(), values)), "log_loss": res.metrics.get("model_log_loss"),
                     "rps": res.metrics.get("model_rps"), "matches": res.metrics.get("matches")})
        log.info("tune %s -> %.4f", rows[-1], rows[-1]["log_loss"] or float("nan"))
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
