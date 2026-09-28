import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from soccer_quant.config import ModelParams
from soccer_quant.models import context, dixon_coles
from soccer_quant.models.elo import Elo
from soccer_quant.teams import TeamResolver, normalize


def _league(seed=0, seasons=2, n=16):
    rng = np.random.default_rng(seed)
    att, dfn = rng.normal(0, 0.3, n), rng.normal(0, 0.3, n)
    teams = [f"t{i:02d}" for i in range(n)]
    rows = []
    for _ in range(seasons):
        for i in range(n):
            for j in range(n):
                if i != j:
                    lam, mu = np.exp(0.1 + 0.25 + att[i] - dfn[j]), np.exp(0.1 + att[j] - dfn[i])
                    rows.append((teams[i], teams[j], rng.poisson(lam), rng.poisson(mu)))
    h, a, x, y = map(np.array, zip(*rows))
    return h, a, x.astype(float), y.astype(float), att, dfn


def test_dixon_coles_gradient_matches_finite_differences():
    h, a, x, y, *_ = _league(seasons=1, n=8)
    index = {t: i for i, t in enumerate(sorted(set(h)))}
    hi, ai = np.array([index[t] for t in h]), np.array([index[t] for t in a])
    masks = ((x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1))
    n = len(index)
    args = (hi, ai, x * 0.8 + 0.3, y, np.linspace(0.3, 1, len(x)), n, np.zeros(n), np.ones(n), masks)
    x0 = np.concatenate([np.linspace(-0.2, 0.2, 2 * n), [0.2, 0.1, -0.08]])
    numeric = approx_fprime(x0, lambda v: dixon_coles._objective(v, *args)[0], 1e-6)
    assert np.allclose(numeric, dixon_coles._objective(x0, *args)[1], atol=1e-3)


def test_dixon_coles_recovers_strengths_and_home_advantage():
    h, a, x, y, att, dfn = _league(seasons=3)
    model = dixon_coles.fit(h, a, x, y, ridge=0.5)
    assert np.corrcoef(model.attack, att)[0, 1] > 0.85
    assert np.corrcoef(model.defence, dfn)[0, 1] > 0.85
    assert 0.15 < model.home_advantage < 0.35
    lam, mu = model.expected_goals("t00", "t01")
    assert lam > 0 and mu > 0


def test_unknown_team_gets_prior():
    h, a, x, y, *_ = _league(seasons=1, n=6)
    model = dixon_coles.fit(h, a, x, y, unknown_prior=-0.2)
    assert model.rating("promoted") == (-0.2, -0.2)


def test_score_matrix_is_a_distribution():
    m = dixon_coles.score_matrix(1.6, 1.1, -0.1)
    assert m.shape == (11, 11)
    assert m.sum() == pytest.approx(1.0)
    independent = dixon_coles.score_matrix(1.6, 1.1, 0.0)
    assert m[0, 0] > independent[0, 0] and m[1, 1] > independent[1, 1]  # rho < 0 inflates low draws


def test_elo_rewards_winners_and_fits_supremacy():
    rng = np.random.default_rng(1)
    rows, start = [], pd.Timestamp("2024-08-01", tz="UTC")
    for k in range(400):
        h, a = ("strong", "weak") if k % 2 == 0 else ("weak", "strong")
        hg = rng.poisson(2.2 if h == "strong" else 0.8)
        ag = rng.poisson(0.8 if h == "strong" else 2.2)
        rows.append({"kickoff": start + pd.Timedelta(days=k), "home_key": h, "away_key": a,
                     "home_goals": hg, "away_goals": ag})
    elo = Elo().fit(pd.DataFrame(rows))
    assert elo.rating("strong") > elo.rating("weak") + 100
    assert elo.supremacy("strong", "weak") > 0.5


def test_weather_altitude_travel_rest():
    p = ModelParams()
    assert context.weather(None, p).home_mult == 1.0
    storm = context.weather({"precip_mm_h": 6, "wind_kmh": 50, "temp_c": 10}, p)
    assert storm.home_mult == pytest.approx(p.rain_heavy_mult * p.wind_severe_mult)
    alt = context.altitude(2660, 30, p)  # Toluca hosting Tijuana
    assert alt.home_mult > 1.05 and alt.away_mult < 0.95
    assert context.altitude(30, 2660, p).active is False  # no penalty for playing low
    trip, km = context.travel((19.29, -99.67), (32.51, -116.99), p)
    assert km > 2000 and trip.away_mult < 1
    tired = context.rest(3, 7, p)
    assert tired.home_mult < 1 < tired.away_mult


def test_head_to_head_is_shrunk_and_capped():
    p = ModelParams()
    meetings = pd.DataFrame({
        "kickoff": pd.to_datetime(["2025-01-01", "2025-06-01", "2026-01-01"], utc=True),
        "home_key": ["a", "b", "a"], "away_key": ["b", "a", "b"],
        "home_goals": [5, 0, 6], "away_goals": [0, 5, 0]})
    adj, summary = context.head_to_head(meetings, "a", lambda h, a: 0.0, pd.Timestamp("2026-09-01", tz="UTC"),
                                        1.5, 1.2, p)
    assert summary["wins"] == 3
    shift = (1.5 * adj.home_mult - 1.5) - (1.2 * adj.away_mult - 1.2)
    assert 0 < shift <= p.h2h_cap + 1e-9


def test_team_names_resolve_across_sources():
    r = TeamResolver()
    assert r.resolve("Nott'm Forest") == r.resolve("Nottingham Forest") == "nottingham-forest"
    assert r.resolve("Man United") == r.resolve("Manchester United") == "man-united"
    assert r.resolve("U.N.A.M.- Pumas") == r.resolve("Pumas UNAM") == "pumas"
    assert r.resolve("Atl. San Luis") == r.resolve("Atlético de San Luis") == "atletico-san-luis"
    assert r.resolve("Club América") == "america"
    assert r.resolve("AFC Bournemouth") == "bournemouth"
    assert normalize("  Brighton & Hove Albion ") == "brighton hove albion"
    assert r.resolve("Racing Santander FC") == "racing-santander"
