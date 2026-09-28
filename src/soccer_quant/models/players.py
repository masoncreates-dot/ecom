"""Player impact ratings and team-strength adjustments.

Team ratings from Dixon-Coles describe the side that actually played over the
rating window. This module estimates how today's XI differs from that side:

* each player gets a value = match rating above a positional replacement
  level, translated across leagues (for new signings) and shrunk toward zero
  when he has few minutes;
* the *baseline XI* is picked from everyone who played for the club in the
  window (including players since sold), the *projected XI* from today's squad
  minus injured / suspended players (or the confirmed lineup, once published);
* the attack / defence value gap between the two becomes a multiplier on the
  team's expected goals for and against.

So a sold star, a long-term injury, a suspension and a big new signing all
move the prediction, and a full-strength XI leaves it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DEFAULT_LEAGUE_STRENGTH, LEAGUE_STRENGTH, League, ModelParams

REPLACEMENT = {"Goalkeeper": 6.45, "Defender": 6.55, "Midfielder": 6.60, "Attacker": 6.55}
ATTACK_SHARE = {"Goalkeeper": 0.0, "Defender": 0.25, "Midfielder": 0.55, "Attacker": 0.90}
MIN_OUTFIELD = {"Defender": 3, "Midfielder": 2, "Attacker": 1}
UNKNOWN_VALUE = -0.10
SEASON_WEIGHTS = (1.0, 0.5)  # current season, previous season
_POS_ALIASES = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Attacker"}


@dataclass
class PlayerImpact:
    player_id: int
    name: str
    position: str
    value: float  # rating points above replacement, league-adjusted and shrunk
    rating: float | None
    minutes: float  # recent minutes for this club (season-weighted)
    share: float  # share of the club's minutes x 11
    pick: float  # how likely he starts when fit
    new_signing: bool = False

    @property
    def attack(self) -> float:
        return self.value * ATTACK_SHARE[self.position]

    @property
    def defence(self) -> float:
        return self.value * (1.0 - ATTACK_SHARE[self.position])


@dataclass
class TeamPlayerAdjustment:
    team: str
    attack_mult: float = 1.0
    concede_mult: float = 1.0  # multiplies the opponent's expected goals
    source: str = "no player data"
    expected_xi: list[PlayerImpact] = field(default_factory=list)
    missing: list[tuple[PlayerImpact, str, str]] = field(default_factory=list)  # player, status, reason
    delta_attack: float = 0.0
    delta_defence: float = 0.0


class PlayerModel:
    def __init__(self, players: pd.DataFrame, squads: pd.DataFrame, teams: pd.DataFrame,
                 league: League, season: int, params: ModelParams):
        self.league = league
        self.season = season
        self.params = params
        p = players.copy()
        for col in ("season", "minutes", "rating", "lineups", "appearances", "team_api_id", "league_api_id"):
            p[col] = pd.to_numeric(p.get(col), errors="coerce")
        self.players = p[p["season"].isin([season, season - 1]) & (p["minutes"] > 0)]
        self.squads = squads
        cols = ("season", "minutes", "rating", "league_api_id", "team_api_id", "appearances", "lineups",
                "position", "name")
        self._by_player = {int(pid): {c: g[c].to_numpy(dtype=float if c not in ("position", "name") else object)
                                      for c in cols}
                           for pid, g in self.players.groupby("player_id")}
        self._squad_rows = {(r.team_key, int(r.player_id)): r for r in squads.itertuples()}
        self.api_ids = {k: int(v) for k, v in zip(teams["team_key"], teams["api_id"]) if pd.notna(v)}
        self._cache: dict[str, dict[int, PlayerImpact]] = {}

    def impacts(self, team: str) -> dict[int, PlayerImpact]:
        """Every player relevant to ``team``: current squad plus recent departures."""
        if team in self._cache:
            return self._cache[team]
        api_id = self.api_ids.get(team)
        out: dict[int, PlayerImpact] = {}
        if api_id is not None:
            club = self.players[self.players["team_api_id"] == api_id]
            domestic = club[club["league_api_id"] == self.league.api_football_id]
            weights = domestic["season"].map({self.season: SEASON_WEIGHTS[0], self.season - 1: SEASON_WEIGHTS[1]})
            wmin = (domestic["minutes"] * weights).groupby(domestic["player_id"]).sum()
            total = wmin.sum()
            squad_ids = set(self.squads.loc[self.squads["team_key"] == team, "player_id"].astype(int))
            candidate_ids = set(wmin.index.astype(int)) | squad_ids
            for pid in candidate_ids:
                share = float(wmin.get(pid, 0.0) / total * 11) if total > 0 else 0.0
                out[pid] = self._impact(pid, team, api_id, share, float(wmin.get(pid, 0.0)))
        self._cache[team] = out
        return out

    def current_squad(self, team: str) -> set[int]:
        squad = set(self.squads.loc[self.squads["team_key"] == team, "player_id"].astype(int))
        if squad:
            return squad
        api_id = self.api_ids.get(team)
        rows = self.players[(self.players["team_api_id"] == api_id) & (self.players["season"] == self.season)]
        if rows.empty:
            rows = self.players[(self.players["team_api_id"] == api_id) & (self.players["season"] == self.season - 1)]
        return set(rows["player_id"].astype(int))

    def adjustment(self, team: str, unavailable: dict[int, tuple[str, str]] | None = None,
                   lineup: list[int] | None = None) -> TeamPlayerAdjustment:
        """``unavailable``: player_id -> (status "out" | "doubtful", reason)."""
        impacts = self.impacts(team)
        if not impacts:
            return TeamPlayerAdjustment(team)
        unavailable = unavailable or {}
        baseline_pool = [p for p in impacts.values() if p.share > 0]
        baseline_xi = _pick_xi(baseline_pool, key=lambda p: p.share + 0.5 * p.value)
        shape = _shape(baseline_xi)
        squad = self.current_squad(team)
        pool = [p for pid, p in impacts.items() if pid in squad]

        if lineup and len(lineup) >= 11:
            source = "confirmed lineup"
            xi = [impacts.get(pid) or PlayerImpact(pid, str(pid), "Midfielder", UNKNOWN_VALUE, None, 0, 0, 0)
                  for pid in lineup[:11]]
            today_att, today_def = _sum(xi)
            missing = []
        else:
            source = "projected XI"
            out_ids = {pid for pid, (status, _) in unavailable.items() if status == "out"}
            doubt_ids = {pid for pid, (status, _) in unavailable.items() if status == "doubtful"}
            fit_pool = [p for p in pool if p.player_id not in out_ids]
            key = lambda p: p.pick + 0.5 * p.value  # noqa: E731 - managers pick on form and quality
            xi = _pick_xi(fit_pool, key, shape)
            xi_without = _pick_xi([p for p in fit_pool if p.player_id not in doubt_ids], key, shape)
            q = self.params.questionable_play_prob if doubt_ids & {p.player_id for p in xi} else 1.0
            a1, d1 = _sum(xi)
            a0, d0 = _sum(xi_without)
            today_att, today_def = q * a1 + (1 - q) * a0, q * d1 + (1 - q) * d0
            missing = [(impacts[pid], status, reason) for pid, (status, reason) in unavailable.items()
                       if pid in impacts]
            missing.sort(key=lambda m: -m[0].value)

        base_att, base_def = _sum(baseline_xi)
        d_att, d_def = today_att - base_att, today_def - base_def
        lo, hi = self.params.player_mult_floor, self.params.player_mult_cap
        return TeamPlayerAdjustment(
            team=team,
            attack_mult=float(np.clip(np.exp(self.params.player_att_k * d_att), lo, hi)),
            concede_mult=float(np.clip(np.exp(-self.params.player_def_k * d_def), lo, hi)),
            source=source, expected_xi=xi, missing=missing, delta_attack=d_att, delta_defence=d_def,
        )

    def team_sheet(self, team: str, unavailable: dict[int, tuple[str, str]] | None = None) -> pd.DataFrame:
        impacts = self.impacts(team)
        squad = self.current_squad(team)
        unavailable = unavailable or {}
        rows = []
        for pid, p in impacts.items():
            status = unavailable.get(pid, ("available", ""))[0] if pid in squad else "left club"
            rows.append({"player": p.name, "position": p.position, "rating": p.rating, "value": round(p.value, 3),
                         "minutes": round(p.minutes), "share": round(p.share, 2), "status": status,
                         "new_signing": p.new_signing})
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True) if rows else \
            pd.DataFrame(columns=["player", "position", "rating", "value", "minutes", "share", "status", "new_signing"])

    # ------------------------------------------------------------------ internals
    def _impact(self, pid: int, team: str, api_id: int, share: float, club_minutes: float) -> PlayerImpact:
        rows = self._by_player.get(pid)
        squad_row = self._squad_rows.get((team, pid))
        position = _position(rows, squad_row)
        names = [n for n in (rows["name"] if rows else []) if isinstance(n, str)]
        name = names[0] if names else (squad_row.name if squad_row is not None else str(pid))
        value, rating, new_signing, pick = UNKNOWN_VALUE, None, False, min(share, 1.0)
        if rows:
            rated = ~np.isnan(rows["rating"])
            if rated.any():
                target = LEAGUE_STRENGTH.get(self.league.api_football_id, DEFAULT_LEAGUE_STRENGTH)
                sw = np.where(rows["season"] == self.season, SEASON_WEIGHTS[0], SEASON_WEIGHTS[1])[rated]
                factor = np.array([
                    np.clip(LEAGUE_STRENGTH.get(int(lid), DEFAULT_LEAGUE_STRENGTH) / target, 0.5, 1.3)
                    if lid == lid else 1.0 for lid in rows["league_api_id"][rated]])
                r = rows["rating"][rated]
                raw = (r - REPLACEMENT[position]) * factor - (1 - factor) * 0.6
                w = rows["minutes"][rated] * sw
                minutes = float(w.sum())
                value = float((raw * w).sum() / minutes) * minutes / (minutes + self.params.player_min_minutes_shrink)
                rating = float((r * w).sum() / minutes)
            # New signings: no settled role here yet, so use how often they
            # started at the previous club (discounted) as the start probability.
            elsewhere = rows["team_api_id"] != api_id
            new_signing = club_minutes < 270 and bool(elsewhere.any())
            if new_signing:
                apps = float(np.nansum(rows["appearances"][elsewhere]))
                rate = float(np.nansum(rows["lineups"][elsewhere]) / apps) if apps else 0.0
                pick = max(pick, 0.85 * rate * min(1.0, apps / 10))
        return PlayerImpact(pid, name, position, value, rating, club_minutes, share, pick, new_signing)


def _position(rows: dict | None, squad_row) -> str:
    if squad_row is not None and squad_row.position in REPLACEMENT:
        return squad_row.position
    if rows:
        totals: dict[str, float] = {}
        for pos, mins in zip(rows["position"], rows["minutes"]):
            pos = _POS_ALIASES.get(pos, pos)
            if pos in REPLACEMENT:
                totals[pos] = totals.get(pos, 0.0) + mins
        if totals:
            return max(totals, key=totals.get)
    return "Midfielder"


def _pick_xi(pool: list[PlayerImpact], key, shape: dict[str, int] | None = None) -> list[PlayerImpact]:
    """Best XI by ``key``: one keeper, then outfield players.

    With ``shape`` (players per position, from the baseline XI) replacements
    are like-for-like, so an injured striker is replaced by the next striker
    rather than by an extra defender.
    """
    ranked = sorted(pool, key=lambda p: (key(p), p.value), reverse=True)
    keepers = [p for p in ranked if p.position == "Goalkeeper"]
    outfield = [p for p in ranked if p.position != "Goalkeeper"]
    wanted = {k: v for k, v in (shape or MIN_OUTFIELD).items() if k != "Goalkeeper"}
    chosen: list[PlayerImpact] = []
    for pos, count in wanted.items():
        chosen += [p for p in outfield if p.position == pos][:count]
    ids = {p.player_id for p in chosen}
    for p in outfield:
        if len(chosen) >= 10:
            break
        if p.player_id not in ids:
            chosen.append(p)
            ids.add(p.player_id)
    return keepers[:1] + chosen[:10]


def _shape(xi: list[PlayerImpact]) -> dict[str, int]:
    shape: dict[str, int] = {}
    for p in xi:
        shape[p.position] = shape.get(p.position, 0) + 1
    return shape


def _sum(xi: list[PlayerImpact]) -> tuple[float, float]:
    return sum(p.attack for p in xi), sum(p.defence for p in xi)
