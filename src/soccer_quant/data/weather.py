"""Kick-off weather from Open-Meteo (free, no key).

Forecasts cover the next 16 days; the archive covers the past (with a few
days' lag) and is used to backfill weather for backtests.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from ..http import FOREVER, HttpClient
from ..storage import Store

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HOURLY = "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m"


def kickoff_weather(client: HttpClient, lat: float, lon: float, kickoff: datetime, now: datetime) -> dict | None:
    """Average conditions over the two hours from kick-off."""
    day = kickoff.date().isoformat()
    archive = kickoff < now - timedelta(days=5)
    if not archive and kickoff > now + timedelta(days=16):
        return None
    url = ARCHIVE_URL if archive else FORECAST_URL
    params = {"latitude": round(lat, 3), "longitude": round(lon, 3), "hourly": HOURLY, "timezone": "UTC",
              "start_date": day, "end_date": (kickoff + timedelta(hours=3)).date().isoformat()}
    body = client.get(url, params=params, ttl=FOREVER if archive else 3 * 3600)
    return summarize_hourly(body.get("hourly") or {}, kickoff, "archive" if archive else "forecast")


def summarize_hourly(hourly: dict, kickoff: datetime, source: str) -> dict | None:
    if not hourly.get("time"):
        return None
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    start = pd.Timestamp(kickoff).tz_convert("UTC").floor("h")
    window = frame[(frame["time"] >= start) & (frame["time"] < start + pd.Timedelta(hours=2))]
    if window.empty:
        return None
    return {
        "temp_c": float(window["temperature_2m"].mean()),
        "precip_mm_h": float(window["precipitation"].mean()),
        "wind_kmh": float(window["wind_speed_10m"].max()),
        "humidity": float(window["relative_humidity_2m"].mean()),
        "source": source,
    }


def geocode(client: HttpClient, city: str) -> dict | None:
    body = client.get(GEOCODE_URL, params={"name": city, "count": 1}, ttl=FOREVER)
    results = body.get("results") or []
    if not results:
        return None
    r = results[0]
    return {"lat": r["latitude"], "lon": r["longitude"], "elevation_m": r.get("elevation")}


def sync(client: HttpClient, store: Store, teams: pd.DataFrame, matches: pd.DataFrame, now: datetime,
         include_past: bool = False) -> int:
    """Fetch kick-off weather for upcoming matches (and past ones if asked)."""
    coords = _team_coords(client, store, teams, matches)
    have = set(store.frame("weather")["match_id"])
    now_ts = pd.Timestamp(now)
    todo = matches[(matches["kickoff"] <= now_ts + pd.Timedelta(days=16))]
    if not include_past:
        todo = todo[todo["kickoff"] >= now_ts - pd.Timedelta(hours=3)]
    rows = []
    for _, m in todo.iterrows():
        if m["match_id"] in have and m["kickoff"] < now_ts - pd.Timedelta(days=5):
            continue
        loc = coords.get(m["home_key"])
        if loc is None:
            continue
        try:
            w = kickoff_weather(client, loc[0], loc[1], m["kickoff"].to_pydatetime(), now)
        except Exception as exc:  # noqa: BLE001 - weather is optional; never block predictions
            log.warning("weather for %s failed: %s", m["match_id"], exc)
            continue
        if w:
            rows.append({"match_id": m["match_id"], **w, "fetched_at": now.isoformat()})
    return store.upsert("weather", rows)


def _team_coords(client: HttpClient, store: Store, teams: pd.DataFrame,
                 matches: pd.DataFrame) -> dict[str, tuple[float, float]]:
    coords = {}
    for _, t in teams.iterrows():
        if pd.notna(t.get("lat")) and pd.notna(t.get("lon")):
            coords[t["team_key"]] = (float(t["lat"]), float(t["lon"]))
            continue
        city = t.get("city")
        if not isinstance(city, str) and "venue_city" in matches:
            cities = matches.loc[matches["home_key"] == t["team_key"], "venue_city"].dropna()
            city = cities.iloc[-1] if len(cities) else None
        if not isinstance(city, str):
            continue
        try:
            found = geocode(client, city)
        except Exception as exc:  # noqa: BLE001
            log.warning("geocoding %s failed: %s", city, exc)
            continue
        if found:
            store.upsert("teams", [{"team_key": t["team_key"], "city": city, **found}])
            coords[t["team_key"]] = (found["lat"], found["lon"])
    return coords
