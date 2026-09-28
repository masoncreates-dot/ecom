import pandas as pd
import pytest

from soccer_quant.config import TradingParams
from soccer_quant.models import markets
from soccer_quant.models.dixon_coles import score_matrix
from soccer_quant.models.markets import Selection
from soccer_quant.trading.odds import devig, overround
from soccer_quant.trading.signals import Signal, find_signals, market_probabilities, size_positions

M = score_matrix(1.5, 1.1, -0.08)


def test_1x2_and_complements():
    s = markets.summarize(M)
    assert sum(s["1x2"].values()) == pytest.approx(1.0)
    assert s["btts"]["yes"] + s["btts"]["no"] == pytest.approx(1.0)
    assert s["over"][2.5] == pytest.approx(1 - markets.probability(M, Selection("ou", "under", 2.5)))
    assert s["home_xg"] == pytest.approx(1.5, abs=0.05)


def test_asian_handicap_settlement():
    # Home -0.25: a draw loses half the stake, a home win wins in full.
    assert markets.settle(Selection("ah", "home", -0.25), 1, 1, 2.0) == pytest.approx(-0.5)
    assert markets.settle(Selection("ah", "home", -0.25), 2, 1, 2.0) == pytest.approx(1.0)
    # Home -0.75 winning by one: half wins, half pushes.
    assert markets.settle(Selection("ah", "home", -0.75), 2, 1, 2.0) == pytest.approx(0.5)
    # Away +0.5 on a draw wins.
    assert markets.settle(Selection("ah", "away", 0.5), 0, 0, 1.9) == pytest.approx(0.9)
    # Over 3.0 with three goals pushes; over 2.75 wins half.
    assert markets.settle(Selection("ou", "over", 3.0), 2, 1, 1.9) == pytest.approx(0.0)
    assert markets.settle(Selection("ou", "over", 2.75), 2, 1, 2.0) == pytest.approx(0.5)


def test_fair_odds_have_zero_ev_and_kelly_matches_closed_form():
    for sel in (Selection("1x2", "home"), Selection("ah", "home", -0.25), Selection("ou", "over", 3.0)):
        assert markets.expected_value(M, sel, markets.fair_odds(M, sel)) == pytest.approx(0.0, abs=1e-12)
    sel = Selection("1x2", "home")
    p = markets.probability(M, sel)
    odds = 1.1 / p
    assert markets.kelly(M, sel, odds) == pytest.approx((p * odds - 1) / (odds - 1), rel=1e-3)
    assert markets.kelly(M, sel, 0.9 / p) == 0.0


def test_devig_methods():
    prices = [1.80, 3.60, 4.80]
    assert overround(prices) > 0
    for method in ("proportional", "power", "shin"):
        p = devig(prices, method)
        assert p.sum() == pytest.approx(1.0)
        assert p[0] > p[1] > p[2]
    # Shin and power move probability away from the long shot.
    assert devig(prices, "shin")[2] < devig(prices, "proportional")[2]
    assert devig(prices, "power")[2] < devig(prices, "proportional")[2]


def _odds(rows):
    return pd.DataFrame([{"match_id": "m", "is_closing": 0, "captured_at": None, **r} for r in rows])


def test_signals_use_best_price_and_filters():
    p = markets.summarize(M)["1x2"]
    fair = {k: 1 / v for k, v in p.items()}
    odds = _odds([
        {"bookmaker": "pinnacle", "market": "1x2", "selection": s, "line": 0.0, "price": round(fair[s] / 1.03, 2)}
        for s in ("home", "draw", "away")] + [
        {"bookmaker": "softbook", "market": "1x2", "selection": "home", "line": 0.0, "price": round(fair["home"] * 1.08, 2)},
        {"bookmaker": "max", "market": "1x2", "selection": "home", "line": 0.0, "price": 99.0},
    ])
    params = TradingParams()
    mkt = market_probabilities(odds, params)
    assert sum(mkt[("1x2", 0.0)].values()) == pytest.approx(1.0)
    sigs = find_signals(M, odds, params, match_id="m", kickoff=pd.Timestamp("2026-10-01", tz="UTC"),
                        home="h", away="a")
    assert [(s.selection.pick, s.bookmaker) for s in sigs] == [("home", "softbook")]
    assert sigs[0].ev == pytest.approx(0.08, abs=0.01)
    # A huge model/market disagreement is treated as a data problem.
    strict = TradingParams(max_disagreement=0.0)
    assert find_signals(M, odds, strict, match_id="m", kickoff=pd.Timestamp("2026-10-01", tz="UTC"),
                        home="h", away="a") == []


def test_position_sizing_caps():
    params = TradingParams(bankroll=1000, max_stake_pct=0.02, max_daily_exposure_pct=0.05, max_bets_per_match=1)
    kickoff = pd.Timestamp("2026-10-03 15:00", tz="UTC")
    sigs = [Signal(f"m{i}", kickoff, "h", "a", Selection("1x2", "home"), "b", 2.2, 0.5, 0.45, 2.0,
                   0.1 + i / 100, 0.2) for i in range(5)]
    sigs.append(Signal("m0", kickoff, "h", "a", Selection("ou", "over", 2.5), "b", 2.0, 0.55, 0.5, 1.8, 0.05, 0.1))
    sized = size_positions(sigs, 1000, params)
    assert len(sized) == 5  # one bet per match
    assert all(s.stake <= 20 for s in sized)
    assert sum(s.stake for s in sized) == pytest.approx(50, abs=0.05)
