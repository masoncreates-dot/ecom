# soccer-quant

A match-prediction and value-trading engine for the **Premier League** and **Liga MX**.

For every upcoming fixture it estimates expected goals for both teams and builds a full score
distribution. From that it prices 1X2, double chance, over/under, both-teams-to-score, Asian
handicaps and correct scores. Those prices are compared with bookmaker odds, and the bets that
clear its filters are sized with fractional Kelly.

```
soccer-quant run --league all
```

```
Premier League
  Sun 04 Oct 14:00Z   Arsenal v Liverpool   xG 1.48-1.73  1 34%  X 22%  2 44%  O2.5 62%
  ...
3 value bet(s)
  Everton v Newcastle: Over 2.5 @ 4.22 (bet365) EV +7.7% stake 3.97
```

The full report (`reports/<date>/predictions.md`) contains, for each match, every adjustment that
moved the numbers: team news with each missing player's value, new signings in the projected XI,
coach tenure, weather, altitude, travel, rest days, head-to-head record and style matchup. It also
lists fair Asian-handicap and total-goals lines and squad impact tables. CSV and JSON copies are
written alongside it.

## What goes into a prediction

| Layer | What it does | Data |
|---|---|---|
| Dixon-Coles ratings | Attack/defence per team with time decay and low-score correction. xG (or a shots-on-target proxy) is blended into goals to reduce noise. Promoted teams start from a below-average prior. | results, xG, shots |
| Elo | Goal-margin Elo, converted to goal supremacy and blended in | results |
| Coaching changes | Matches played under the previous coach are down-weighted in the ratings. An optional new-manager bounce is available. | API-Football `/coachs` |
| Players | Each player's value is his match rating above a replacement-level player at his position. Ratings from other leagues are translated to this one. The projected XI (or the confirmed lineup, about 40 minutes before kick-off) is compared with the XI that produced the ratings. Injuries, suspensions, doubts, departures and new signings all move expected goals. | squads, player season stats, injuries, lineups, transfers |
| Weather | Heavy rain, strong wind, heat and frost reduce expected goals | Open-Meteo forecast / archive |
| Altitude & travel | Liga MX: a highland home side against lowland visitors (e.g. Toluca at 2,660 m v Tijuana) and long away trips | stadium table |
| Rest | Short rest against a rested opponent. Cup and continental fixtures are included. | team schedules |
| Head-to-head | Residual of past meetings versus what the ratings expected, heavily shrunk and capped | results |
| Matchups | Ridge regression of attack style × defence style (shot volume, set pieces, shot quality) on rating residuals, capped | shots, corners |
| Market blend | Blends the model's expected goals with those implied by the de-vigged sharp price (Pinnacle) | odds |

Trading layer:
- Margin removal: Shin (default), power or proportional.
- Line shopping: the best price per selection is used.
- Exact EV and Kelly for pushes and quarter lines.
- Filters on minimum and maximum EV, minimum probability, maximum odds, and maximum model/market
  disagreement (a large gap is treated as a data problem).
- Caps per bet, per match and per day, with one position per match.
- Every bet is recorded so `settle` can report P&L and **closing-line value**.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export API_FOOTBALL_KEY=...          # https://dashboard.api-football.com
cp soccer_quant.example.toml soccer_quant.toml   # optional overrides
```

Data sources:

| Source | Used for | Key |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | Results, shots, opening and closing odds (EPL back to the 1990s, Liga MX since 2012). Also upcoming EPL fixtures with odds. | none |
| [API-Football](https://www.api-football.com) | Fixtures (both leagues), xG, squads, player stats, injuries, lineups, transfers, coaches, schedules, live odds | `API_FOOTBALL_KEY` |
| [Open-Meteo](https://open-meteo.com) | Kick-off weather forecast and historical weather, geocoding | none |

The free API-Football plan (100 requests/day) supports a daily refresh of fixtures, injuries and
odds. Full player/transfer/coach coverage for both leagues needs a paid plan. Responses are cached
on disk with per-endpoint lifetimes: squads and coaches for 48 h, injuries for 3 h, odds for 30 min
and lineups for 5 min. Run `update` as often as you like and only stale data is refetched. Without
a key, EPL predictions still work from football-data.co.uk fixtures and odds, but without team news.

## Commands

```bash
soccer-quant update  --league all        # pull history, fixtures, team news, odds, weather
soccer-quant update  --past-weather      # also backfill historical kick-off weather for backtests
soccer-quant predict --league all --days 7 --bankroll 500
soccer-quant run     --league epl        # update + predict
soccer-quant settle                      # settle recorded bets, report CLV and P&L
soccer-quant backtest --league epl       # walk-forward test against historical odds
soccer-quant tune     --league epl       # grid-search decay / shots / Elo weights
soccer-quant ratings  --league ligamx    # current attack/defence/Elo table
soccer-quant team "Man City"             # squad impact sheet
soccer-quant demo                        # whole pipeline on synthetic data, no network
```

To keep predictions current with team news, schedule `run` a few times a day. Around kick-off,
lineups arrive and the prediction switches to the confirmed XI:

```cron
15 */3 * * * cd /path/to/repo && .venv/bin/soccer-quant run --league all >> data/run.log 2>&1
```

## Before betting real money

1. Run `soccer-quant backtest`. The number that matters is the model's log loss against the
   de-vigged closing line, and the average CLV of its bets. Most models, including good ones, don't
   beat the Pinnacle closing line. A positive ROI over a few hundred bets with negative CLV is luck.
2. The player, weather, altitude, travel and rest coefficients in `ModelParams` are conservative
   priors from published research, not fitted values. Tune them once you have your own history,
   and keep `market_weight` high until the backtest earns you a lower one.
3. Paper-trade first: `predict` records every recommended bet and `settle` tracks how they did.

## Layout

```
src/soccer_quant/
  config.py            leagues, model priors, trading limits, TOML overrides
  teams.py, venues.py  name resolution across sources; stadium coordinates and elevation
  http.py, storage.py  cached HTTP with request budget; SQLite store with merge-safe upserts
  data/                football-data.co.uk, API-Football, Open-Meteo connectors
  models/              dixon_coles, elo, players, context (weather/altitude/travel/rest/H2H/coach),
                       matchup, markets (pricing of every selection)
  trading/             de-vig, signals, position sizing
  predict.py           fits everything at a point in time and prices fixtures
  backtest.py          walk-forward evaluation and parameter tuning
  pipeline.py, cli.py  update / predict / settle jobs and the command line
  report.py            Markdown, CSV and JSON output
  synthetic.py         fake-but-consistent league for tests and the demo
```

Run the tests with `pytest`.
