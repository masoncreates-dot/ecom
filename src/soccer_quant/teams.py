"""Canonical team keys and name resolution across data sources.

football-data.co.uk, API-Football and bookmakers all spell clubs differently
("Nott'm Forest", "Nottingham Forest", "U.N.A.M.- Pumas", "Pumas UNAM" ...).
Everything in the store is keyed by the canonical slug defined here.
"""

from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger(__name__)

# key -> (league, display name, extra aliases)
_TEAMS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # --- England (Premier League and clubs likely to be promoted) -----------
    "arsenal": ("epl", "Arsenal", ()),
    "aston-villa": ("epl", "Aston Villa", ()),
    "bournemouth": ("epl", "Bournemouth", ("AFC Bournemouth",)),
    "brentford": ("epl", "Brentford", ()),
    "brighton": ("epl", "Brighton", ("Brighton & Hove Albion", "Brighton and Hove Albion", "Brighton Hove Albion")),
    "burnley": ("epl", "Burnley", ()),
    "chelsea": ("epl", "Chelsea", ()),
    "crystal-palace": ("epl", "Crystal Palace", ("C Palace",)),
    "everton": ("epl", "Everton", ()),
    "fulham": ("epl", "Fulham", ()),
    "leeds": ("epl", "Leeds", ("Leeds United", "Leeds Utd")),
    "liverpool": ("epl", "Liverpool", ()),
    "man-city": ("epl", "Man City", ("Manchester City",)),
    "man-united": ("epl", "Man United", ("Manchester United", "Man Utd", "Manchester Utd")),
    "newcastle": ("epl", "Newcastle", ("Newcastle United", "Newcastle Utd")),
    "nottingham-forest": ("epl", "Nottingham Forest", ("Nott'm Forest", "Nottm Forest")),
    "sunderland": ("epl", "Sunderland", ()),
    "tottenham": ("epl", "Tottenham", ("Tottenham Hotspur", "Spurs")),
    "west-ham": ("epl", "West Ham", ("West Ham United", "West Ham Utd")),
    "wolves": ("epl", "Wolves", ("Wolverhampton", "Wolverhampton Wanderers")),
    "ipswich": ("epl", "Ipswich", ("Ipswich Town",)),
    "leicester": ("epl", "Leicester", ("Leicester City",)),
    "southampton": ("epl", "Southampton", ()),
    "sheffield-united": ("epl", "Sheffield United", ("Sheffield Utd", "Sheff Utd")),
    "luton": ("epl", "Luton", ("Luton Town",)),
    "middlesbrough": ("epl", "Middlesbrough", ()),
    "coventry": ("epl", "Coventry", ("Coventry City",)),
    "west-brom": ("epl", "West Brom", ("West Bromwich Albion", "West Bromwich")),
    "norwich": ("epl", "Norwich", ("Norwich City",)),
    "watford": ("epl", "Watford", ()),
    "hull": ("epl", "Hull", ("Hull City",)),
    "millwall": ("epl", "Millwall", ()),
    "stoke": ("epl", "Stoke", ("Stoke City",)),
    "wrexham": ("epl", "Wrexham", ()),
    "birmingham": ("epl", "Birmingham", ("Birmingham City",)),
    "derby": ("epl", "Derby", ("Derby County",)),
    "swansea": ("epl", "Swansea", ("Swansea City",)),
    "qpr": ("epl", "QPR", ("Queens Park Rangers",)),
    "portsmouth": ("epl", "Portsmouth", ()),
    "preston": ("epl", "Preston", ("Preston North End",)),
    "bristol-city": ("epl", "Bristol City", ()),
    "charlton": ("epl", "Charlton", ("Charlton Athletic",)),
    "blackburn": ("epl", "Blackburn", ("Blackburn Rovers",)),
    "oxford": ("epl", "Oxford", ("Oxford United",)),
    "sheffield-wednesday": ("epl", "Sheffield Wednesday", ("Sheffield Weds", "Sheff Wed")),
    "cardiff": ("epl", "Cardiff", ("Cardiff City",)),
    "huddersfield": ("epl", "Huddersfield", ("Huddersfield Town",)),
    "plymouth": ("epl", "Plymouth", ("Plymouth Argyle",)),
    # --- Mexico (Liga MX) ----------------------------------------------------
    "america": ("ligamx", "America", ("Club America",)),
    "guadalajara": ("ligamx", "Guadalajara", ("Guadalajara Chivas", "Chivas", "Chivas Guadalajara")),
    "cruz-azul": ("ligamx", "Cruz Azul", ()),
    "pumas": ("ligamx", "Pumas UNAM", ("U.N.A.M.- Pumas", "U.N.A.M. - Pumas", "UNAM", "Pumas", "Pumas U.N.A.M.")),
    "tigres": ("ligamx", "Tigres UANL", ("U.A.N.L.- Tigres", "U.A.N.L. - Tigres", "Tigres", "UANL")),
    "monterrey": ("ligamx", "Monterrey", ("Rayados",)),
    "toluca": ("ligamx", "Toluca", ("Deportivo Toluca",)),
    "pachuca": ("ligamx", "Pachuca", ()),
    "leon": ("ligamx", "Leon", ("Club Leon",)),
    "santos-laguna": ("ligamx", "Santos Laguna", ("Santos",)),
    "atlas": ("ligamx", "Atlas", ()),
    "puebla": ("ligamx", "Puebla", ()),
    "queretaro": ("ligamx", "Queretaro", ("Club Queretaro", "Gallos Blancos")),
    "necaxa": ("ligamx", "Necaxa", ()),
    "tijuana": ("ligamx", "Tijuana", ("Club Tijuana", "Xolos")),
    "juarez": ("ligamx", "Juarez", ("FC Juarez", "Bravos de Juarez")),
    "atletico-san-luis": ("ligamx", "Atletico San Luis", ("Atl. San Luis", "San Luis", "Atletico de San Luis")),
    "mazatlan": ("ligamx", "Mazatlan", ("Mazatlan FC",)),
    "atlante": ("ligamx", "Atlante", ()),
}

_STOPWORDS = {"fc", "cf", "afc", "club", "cd", "sc", "the", "de"}


def normalize(name: str) -> str:
    """Accent-, case- and punctuation-insensitive form of a club name."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = text.lower().replace(".", "").replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(tok for tok in text.split() if tok not in _STOPWORDS)


def slugify(name: str) -> str:
    return normalize(name).replace(" ", "-") or "unknown"


class TeamResolver:
    def __init__(self, extra_aliases: dict[str, str] | None = None):
        self._aliases: dict[str, str] = {}
        self._display: dict[str, str] = {}
        self._league: dict[str, str] = {}
        for key, (league, display, aliases) in _TEAMS.items():
            self._display[key] = display
            self._league[key] = league
            for alias in (key.replace("-", " "), display, *aliases):
                self._aliases[normalize(alias)] = key
        for alias, key in (extra_aliases or {}).items():
            self._aliases[normalize(alias)] = key
        self._warned: set[str] = set()

    def resolve(self, name: str) -> str:
        norm = normalize(name)
        key = self._aliases.get(norm)
        if key:
            return key
        key = slugify(name)
        if key not in self._display and name not in self._warned:
            self._warned.add(name)
            log.warning("Unknown team name %r -> %r (add an alias in soccer_quant.toml if wrong)", name, key)
        return key

    def display(self, key: str) -> str:
        return self._display.get(key, key.replace("-", " ").title())

    def league_of(self, key: str) -> str | None:
        return self._league.get(key)
