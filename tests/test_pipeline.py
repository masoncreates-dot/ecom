from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from soccer_quant import backtest, pipeline
from soccer_quant.config import LEAGUES, ModelParams, Settings, TradingParams
from soccer_quant.models.players import PlayerModel
from soccer_quant.predict import Predictor
from soccer_quant.storage import Store

from .conftest import TODAY


def test_predictions_cover_upcoming_fixtures(epl_data):
    predictor = Predictor(epl_data, ModelParams(), TradingParams()).fit(TODAY)
    preds = predictor.predict_upcoming(14)
    assert len(preds) == 20
    for p in preds:
        assert p.matrix.sum() == pytest.approx(1.0)
        assert sum(p.markets["1x2"].values()) == pytest.approx(1.0)
        assert 0.2 < p.final_xg[0] < 4.5 and 0.2 < p.final_xg[1] < 4.5
        assert {a.name for a in p.adjustments} >= {"elo blend", "players", "weather", "altitude", "travel", "rest",
                                                   "head-to-head", "matchup"}
    storm = min(preds, key=lambda p: p.kickoff)
    weather = next(a for a in storm.adjustments if a.name == "weather")
    assert weather.home_mult < 0.9  # the synthetic storm


def test_ratings_track_true_strength(epl_data, synthetic_store):
    predictor = Predictor(epl_data, ModelParams(), TradingParams()).fit(TODAY)
    truth = synthetic_store.truth["epl"]
    teams = predictor.dc.teams
    true_overall = np.array([truth["attack"][t] + truth["defence"][t] for t in teams])
    fitted = predictor.dc.attack + predictor.dc.defence
    assert np.corrcoef(true_overall, fitted)[0, 1] > 0.8


def test_player_adjustments(epl_data):
    season = LEAGUES["epl"].season_for(TODAY)
    model = PlayerModel(epl_data.players, epl_data.squads, epl_data.teams, LEAGUES["epl"], season, ModelParams())
    team = "arsenal"
    assert model.adjustment(team).attack_mult == pytest.approx(1.0)
    best_attacker = max((p for p in model.impacts(team).values() if p.position == "Attacker" and p.share > 0.5),
                        key=lambda p: p.value)
    out = model.adjustment(team, {best_attacker.player_id: ("out", "hamstring")})
    assert out.attack_mult < 0.99 and out.missing[0][0].player_id == best_attacker.player_id
    doubtful = model.adjustment(team, {best_attacker.player_id: ("doubtful", "knock")})
    assert out.attack_mult < doubtful.attack_mult < 1.0
    lineup = [p.player_id for p in out.expected_xi]
    confirmed = model.adjustment(team, lineup=lineup)
    assert confirmed.source == "confirmed lineup"
    assert confirmed.attack_mult == pytest.approx(out.attack_mult)
    buyer = next(t for t in model.api_ids if any(p.new_signing for p in model.impacts(t).values()))
    assert model.adjustment(buyer).attack_mult > 1.0  # the La Liga signing strengthens the attack


def test_liga_mx_altitude_shows_up(ligamx_data):
    predictor = Predictor(ligamx_data, ModelParams(), TradingParams()).fit(TODAY)
    fx = pd.Series({"match_id": "x", "home_key": "toluca", "away_key": "tijuana",
                    "kickoff": TODAY + pd.Timedelta(days=3), "api_fixture_id": None})
    pred = predictor.predict(fx, odds=ligamx_data.odds.iloc[:0])
    alt = next(a for a in pred.adjustments if a.name == "altitude")
    trip = next(a for a in pred.adjustments if a.name == "travel")
    assert alt.home_mult > 1.05 and alt.away_mult < 0.95 and trip.away_mult < 1.0


def test_market_blend_moves_toward_market(epl_data):
    fx = epl_data.matches[epl_data.matches["status"] == "NS"].iloc[0]
    pure = Predictor(epl_data, ModelParams(), TradingParams(), market_weight=0.0).fit(TODAY).predict(fx)
    blended = Predictor(epl_data, ModelParams(), TradingParams(), market_weight=1.0).fit(TODAY).predict(fx)
    mkt = blended.market_probs[("1x2", 0.0)]
    gap_pure = abs(pure.markets["1x2"]["home"] - mkt["home"])
    gap_blend = abs(blended.markets["1x2"]["home"] - mkt["home"])
    assert gap_blend < 0.02 and gap_blend <= gap_pure


def test_backtest_runs_and_reports(epl_data):
    start = TODAY - pd.Timedelta(days=120)
    res = backtest.run(epl_data, ModelParams(), TradingParams(), start=start, refit_days=28)
    m = res.metrics
    assert m["matches"] > 50
    for key in ("model_log_loss", "final_log_loss", "close_log_loss", "model_rps"):
        assert 0 < m[key] < 1.5
    assert (res.predictions["kickoff"] >= start).all()
    if m["bets"]:
        assert {"pnl", "clv", "stake"} <= set(res.bets.columns)


def test_predict_writes_reports_and_settles(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    from soccer_quant import synthetic
    synthetic.build(store, LEAGUES["epl"], today=TODAY, seasons=1)
    settings = replace(Settings(), data_dir=tmp_path, reports_dir=tmp_path / "reports",
                       trading=replace(TradingParams(), min_ev=0.0, max_ev=1.0, max_disagreement=1.0))
    runs, signals, paths = pipeline.predict(settings, [LEAGUES["epl"]], days=14, now=TODAY, store=store)
    assert all(path.exists() for path in paths.values())
    text = paths["markdown"].read_text()
    assert "Premier League" in text and "Match notes" in text and "Squad impact ratings" in text
    assert signals, "loose filters should produce at least one signal on synthetic odds"
    first = signals[0]
    store.conn.execute("UPDATE matches SET status='FT', home_goals=2, away_goals=1 WHERE match_id=?", (first.match_id,))
    store.conn.commit()
    settled = pipeline.settle(settings, store=store)
    row = settled[settled["match_id"] == first.match_id].iloc[0]
    assert row["result"] == "2-1" and row["pnl"] == row["pnl"]  # not NaN
