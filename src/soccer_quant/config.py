"""Leagues, model priors, trading limits and file locations.

Every number in ModelParams / TradingParams can be overridden from a
``soccer_quant.toml`` file (see ``soccer_quant.example.toml``). The defaults
are priors taken from the public research literature; run ``soccer-quant
backtest`` / ``tune`` on your own data before trusting them with money.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class League:
    key: str
    name: str
    api_football_id: int
    football_data_code: str
    # "main": one CSV per season with opening + closing odds and match stats.
    # "extra": one CSV for all seasons, closing odds only, no match stats.
    football_data_format: str
    timezone: str
    elo_home_advantage: float
    # Month in which an API-Football "season" year begins (EPL: August,
    # Liga MX: July, i.e. Apertura N + Clausura N+1 = season N).
    season_start_month: int

    def season_for(self, when: date | datetime) -> int:
        return when.year if when.month >= self.season_start_month else when.year - 1


LEAGUES: dict[str, League] = {
    "epl": League("epl", "Premier League", 39, "E0", "main", "Europe/London", 55.0, 7),
    "ligamx": League("ligamx", "Liga MX", 262, "MEX", "extra", "America/Mexico_City", 75.0, 7),
}

# How strong a player's source league is relative to the Premier League. Used to
# translate a new signing's match ratings from another league into this one.
# Keys are API-Football league ids.
LEAGUE_STRENGTH: dict[int, float] = {
    39: 1.00,  # Premier League
    140: 0.97,  # La Liga
    78: 0.95,  # Bundesliga
    135: 0.95,  # Serie A
    61: 0.90,  # Ligue 1
    94: 0.82,  # Primeira Liga
    88: 0.80,  # Eredivisie
    71: 0.78,  # Brasileirao
    144: 0.75,  # Belgian Pro League
    203: 0.75,  # Super Lig
    40: 0.72,  # Championship
    128: 0.72,  # Liga Profesional Argentina
    262: 0.70,  # Liga MX
    253: 0.65,  # MLS
    179: 0.65,  # Scottish Premiership
    2: 1.00,  # Champions League
    3: 0.92,  # Europa League
    848: 0.85,  # Conference League
}
DEFAULT_LEAGUE_STRENGTH = 0.68


@dataclass(frozen=True)
class ModelParams:
    # --- Dixon-Coles team ratings -------------------------------------------
    time_decay_xi: float = 0.0019  # per day; ~1 year half-life of 365 days
    max_history_days: int = 1100
    ridge: float = 1.0  # shrinkage of attack/defence toward their prior
    new_team_ridge_mult: float = 3.0  # stronger prior for promoted / new teams
    new_team_min_matches: int = 19
    new_team_prior: float = -0.15  # log-scale attack and defence prior for new teams
    xg_weight: float = 0.5  # blend of xG into the goal signal when xG exists
    sot_weight: float = 0.3  # blend of shots-on-target proxy when xG is absent
    max_goals: int = 10

    # --- Elo ------------------------------------------------------------------
    elo_k: float = 20.0
    elo_weight: float = 0.25  # share of goal supremacy taken from Elo

    # --- Coaching changes -----------------------------------------------------
    coach_pre_weight: float = 0.65  # down-weight matches before the current coach
    new_manager_days: int = 45
    new_manager_bounce: float = 0.0  # attack multiplier bump; evidence is weak

    # --- Players --------------------------------------------------------------
    player_att_k: float = 0.12  # attack log-multiplier per rating point above replacement
    player_def_k: float = 0.10
    player_min_minutes_shrink: float = 600.0
    questionable_play_prob: float = 0.5
    player_mult_floor: float = 0.75
    player_mult_cap: float = 1.25

    # --- Weather (multiplies both teams' expected goals) ----------------------
    rain_moderate_mm_h: float = 1.5
    rain_heavy_mm_h: float = 4.0
    rain_moderate_mult: float = 0.97
    rain_heavy_mult: float = 0.94
    wind_strong_kmh: float = 30.0
    wind_severe_kmh: float = 45.0
    wind_strong_mult: float = 0.96
    wind_severe_mult: float = 0.93
    heat_c: float = 28.0
    extreme_heat_c: float = 32.0
    heat_mult: float = 0.97
    extreme_heat_mult: float = 0.95
    cold_c: float = -2.0
    cold_mult: float = 0.97

    # --- Altitude / travel / rest --------------------------------------------
    altitude_threshold_km: float = 0.5
    altitude_home_coef: float = 0.05  # per km of altitude advantage
    altitude_away_coef: float = 0.05
    travel_free_km: float = 500.0
    travel_coef: float = 0.02  # away attack log-penalty per 1000 km beyond free km
    short_rest_days: float = 3.0
    rested_days: float = 5.0
    short_rest_attack_mult: float = 0.97
    short_rest_concede_mult: float = 1.02

    # --- Head to head ---------------------------------------------------------
    h2h_max_matches: int = 10
    h2h_max_years: float = 6.0
    h2h_half_life_days: float = 730.0
    h2h_shrink: float = 25.0  # pseudo-matches of zero residual; H2H is mostly noise
    h2h_cap: float = 0.10  # max goal-supremacy adjustment

    # --- Stylistic matchups ---------------------------------------------------
    matchup_ridge: float = 200.0
    matchup_min_matches: int = 150
    matchup_cap: float = 0.08

    # --- Market blending ------------------------------------------------------
    market_weight: float = 0.5  # 0 = pure model, 1 = pure market; tune with `backtest`


@dataclass(frozen=True)
class TradingParams:
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    min_ev: float = 0.03
    max_ev: float = 0.25  # bigger "edges" are nearly always stale news or bad data
    min_prob: float = 0.08
    max_odds: float = 8.0
    max_disagreement: float = 0.15  # |model - market| above this = suspect data, skip
    max_stake_pct: float = 0.02
    max_match_exposure_pct: float = 0.03
    max_daily_exposure_pct: float = 0.15
    max_bets_per_match: int = 1
    commission: float = 0.0  # exchange commission on net winnings, e.g. 0.02
    devig_method: str = "shin"
    sharp_bookmakers: tuple[str, ...] = ("pinnacle",)
    excluded_bookmakers: tuple[str, ...] = ("avg", "max")
    markets: tuple[str, ...] = ("1x2", "ou", "btts", "ah")


@dataclass
class Settings:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    api_football_key: str | None = None
    api_football_rapidapi: bool = False
    api_daily_budget: int = 100
    history_seasons: int = 4
    horizon_days: int = 10
    model: ModelParams = field(default_factory=ModelParams)
    trading: TradingParams = field(default_factory=TradingParams)
    team_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "soccer_quant.sqlite"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "http_cache"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        """Defaults <- soccer_quant.toml (if present) <- environment variables."""
        settings = cls()
        path = Path(path) if path else Path(os.environ.get("SOCCER_QUANT_CONFIG", "soccer_quant.toml"))
        if path.exists():
            with path.open("rb") as fh:
                settings = settings._apply(tomllib.load(fh))
        key = os.environ.get("API_FOOTBALL_KEY")
        if key:
            settings.api_football_key = key
        if os.environ.get("API_FOOTBALL_RAPIDAPI"):
            settings.api_football_rapidapi = os.environ["API_FOOTBALL_RAPIDAPI"].lower() in {"1", "true", "yes"}
        return settings

    def _apply(self, doc: dict) -> "Settings":
        paths = doc.get("paths", {})
        api = doc.get("api", {})
        general = doc.get("general", {})
        return replace(
            self,
            data_dir=Path(paths.get("data_dir", self.data_dir)),
            reports_dir=Path(paths.get("reports_dir", self.reports_dir)),
            api_football_key=api.get("api_football_key", self.api_football_key),
            api_football_rapidapi=api.get("rapidapi", self.api_football_rapidapi),
            api_daily_budget=int(api.get("daily_budget", self.api_daily_budget)),
            history_seasons=int(general.get("history_seasons", self.history_seasons)),
            horizon_days=int(general.get("horizon_days", self.horizon_days)),
            model=_override(self.model, doc.get("model", {})),
            trading=_override(self.trading, doc.get("trading", {})),
            team_aliases={**self.team_aliases, **doc.get("aliases", {})},
        )


def _override(obj, values: dict):
    known = {f.name: f for f in fields(obj)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {type(obj).__name__} settings: {sorted(unknown)}")
    coerced = {}
    for name, value in values.items():
        current = getattr(obj, name)
        coerced[name] = tuple(value) if isinstance(current, tuple) else type(current)(value)
    return replace(obj, **coerced)
