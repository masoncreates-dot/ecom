"""Compare model prices with bookmaker prices and size the resulting bets.

Guards against the usual ways a model loses money:
* line shopping - the best available price per selection is used;
* sharp reference - the de-vigged sharp price (Pinnacle by default) is shown
  next to the model; a model/market gap above ``max_disagreement`` is treated
  as a data problem (missing team news, wrong lineup) and skipped;
* long shots, tiny edges and implausibly large edges are filtered out
  (``min_prob``, ``max_odds``, ``min_ev``, ``max_ev``);
* fractional Kelly with per-bet, per-match and per-day exposure caps, and at
  most ``max_bets_per_match`` correlated positions per fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import TradingParams
from ..models import markets
from ..models.markets import Selection
from .odds import devig

GROUPS = {"1x2": ("home", "draw", "away"), "ou": ("over", "under"), "btts": ("yes", "no"), "ah": ("home", "away")}


@dataclass
class Signal:
    match_id: str
    kickoff: pd.Timestamp
    home: str
    away: str
    selection: Selection
    bookmaker: str
    price: float
    model_prob: float
    market_prob: float | None
    fair_odds: float
    ev: float
    kelly: float
    stake: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {"match_id": self.match_id, "kickoff": self.kickoff, "home": self.home, "away": self.away,
                "market": self.selection.market, "selection": self.selection.pick, "line": self.selection.line,
                "bet": self.selection.label(), "bookmaker": self.bookmaker, "price": round(self.price, 3),
                "fair_odds": round(self.fair_odds, 3), "model_prob": round(self.model_prob, 4),
                "market_prob": None if self.market_prob is None else round(self.market_prob, 4),
                "ev": round(self.ev, 4), "kelly": round(self.kelly, 4), "stake": round(self.stake, 2)}


def selection_for(market: str, pick: str, line: float) -> Selection:
    """Stored odds keep the home handicap on both AH sides; flip it for away."""
    if market == "ah" and pick == "away":
        return Selection("ah", "away", -line or 0.0)
    return Selection(market, pick, line)


def market_probabilities(odds: pd.DataFrame, params: TradingParams) -> dict[tuple[str, float], dict[str, float]]:
    """De-vigged probability per market group, from the sharpest complete quote."""
    out: dict[tuple[str, float], dict[str, float]] = {}
    if odds.empty:
        return out
    for (market, line), group in odds.groupby(["market", "line"]):
        picks = GROUPS.get(market)
        if not picks:
            continue
        books: dict[str, np.ndarray] = {}
        for book, rows in group.groupby("bookmaker"):
            if book == "max":
                continue
            prices = rows.set_index("selection")["price"]
            if all(p in prices.index for p in picks):
                books[book] = devig([prices[p] for p in picks], params.devig_method)
        if not books:
            continue
        sharp = next((b for b in params.sharp_bookmakers if b in books), None)
        probs = books[sharp] if sharp else np.mean(list(books.values()), axis=0)
        out[(market, float(line))] = dict(zip(picks, (float(p) for p in probs)))
    return out


def best_prices(odds: pd.DataFrame, params: TradingParams) -> pd.DataFrame:
    usable = odds[~odds["bookmaker"].isin(params.excluded_bookmakers) & odds["market"].isin(params.markets)]
    if usable.empty:
        return usable
    idx = usable.groupby(["market", "selection", "line"])["price"].idxmax()
    return usable.loc[idx]


def find_signals(matrix: np.ndarray, odds: pd.DataFrame, params: TradingParams, *, match_id: str,
                 kickoff: pd.Timestamp, home: str, away: str) -> list[Signal]:
    if odds.empty:
        return []
    mkt = market_probabilities(odds, params)
    signals = []
    for row in best_prices(odds, params).itertuples():
        sel = selection_for(row.market, row.selection, float(row.line))
        price = float(row.price)
        model_p = markets.probability(matrix, sel)
        ev = markets.expected_value(matrix, sel, price, params.commission)
        if not params.min_ev <= ev <= params.max_ev or model_p < params.min_prob or price > params.max_odds:
            continue
        market_p = mkt.get((row.market, float(row.line)), {}).get(row.selection)
        sig = Signal(match_id, kickoff, home, away, sel, row.bookmaker, price, model_p, market_p,
                     markets.fair_odds(matrix, sel), ev, markets.kelly(matrix, sel, price, params.commission))
        if market_p is not None and abs(model_p - market_p) > params.max_disagreement:
            continue  # almost always stale team news or a data error, not an edge
        signals.append(sig)
    return signals


def size_positions(signals: list[Signal], bankroll: float, params: TradingParams) -> list[Signal]:
    """Fractional Kelly with per-bet, per-match and per-day caps."""
    kept: list[Signal] = []
    by_match: dict[str, list[Signal]] = {}
    for s in sorted(signals, key=lambda s: -s.ev):
        by_match.setdefault(s.match_id, []).append(s)
    for group in by_match.values():
        room = params.max_match_exposure_pct * bankroll
        for s in group[:params.max_bets_per_match]:
            s.stake = min(s.kelly * params.kelly_fraction * bankroll, params.max_stake_pct * bankroll, room)
            room -= s.stake
            if s.stake >= 0.001 * bankroll:
                kept.append(s)
    by_day: dict[str, list[Signal]] = {}
    for s in kept:
        by_day.setdefault(pd.Timestamp(s.kickoff).strftime("%Y-%m-%d"), []).append(s)
    cap = params.max_daily_exposure_pct * bankroll
    for day in by_day.values():
        total = sum(s.stake for s in day)
        if total > cap:
            for s in day:
                s.stake *= cap / total
    for s in kept:
        s.stake = round(s.stake, 2)
    return sorted(kept, key=lambda s: (s.kickoff, -s.ev))
