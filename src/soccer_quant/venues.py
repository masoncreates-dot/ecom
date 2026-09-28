"""Home stadium locations and elevations.

Coordinates drive the weather lookup; elevation and distance drive the Liga MX
altitude and travel adjustments (Toluca plays at ~2,660 m, Tijuana near sea
level). Clubs missing here are geocoded from the API-Football venue city.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    stadium: str
    city: str
    lat: float
    lon: float
    elevation_m: float


VENUES: dict[str, Venue] = {
    # England
    "arsenal": Venue("Emirates Stadium", "London", 51.5549, -0.1084, 40),
    "aston-villa": Venue("Villa Park", "Birmingham", 52.5092, -1.8847, 110),
    "bournemouth": Venue("Vitality Stadium", "Bournemouth", 50.7352, -1.8384, 20),
    "brentford": Venue("Gtech Community Stadium", "London", 51.4908, -0.2887, 10),
    "brighton": Venue("Amex Stadium", "Brighton", 50.8616, -0.0837, 60),
    "burnley": Venue("Turf Moor", "Burnley", 53.7890, -2.2302, 120),
    "chelsea": Venue("Stamford Bridge", "London", 51.4817, -0.1910, 10),
    "crystal-palace": Venue("Selhurst Park", "London", 51.3983, -0.0855, 50),
    "everton": Venue("Hill Dickinson Stadium", "Liverpool", 53.4263, -3.0005, 5),
    "fulham": Venue("Craven Cottage", "London", 51.4749, -0.2217, 5),
    "leeds": Venue("Elland Road", "Leeds", 53.7778, -1.5722, 40),
    "liverpool": Venue("Anfield", "Liverpool", 53.4308, -2.9608, 50),
    "man-city": Venue("Etihad Stadium", "Manchester", 53.4831, -2.2004, 40),
    "man-united": Venue("Old Trafford", "Manchester", 53.4631, -2.2913, 30),
    "newcastle": Venue("St James' Park", "Newcastle", 54.9756, -1.6217, 60),
    "nottingham-forest": Venue("City Ground", "Nottingham", 52.9400, -1.1328, 25),
    "sunderland": Venue("Stadium of Light", "Sunderland", 54.9146, -1.3884, 20),
    "tottenham": Venue("Tottenham Hotspur Stadium", "London", 51.6043, -0.0664, 20),
    "west-ham": Venue("London Stadium", "London", 51.5386, -0.0166, 10),
    "wolves": Venue("Molineux", "Wolverhampton", 52.5902, -2.1304, 140),
    "ipswich": Venue("Portman Road", "Ipswich", 52.0545, 1.1446, 10),
    "leicester": Venue("King Power Stadium", "Leicester", 52.6204, -1.1422, 60),
    "southampton": Venue("St Mary's Stadium", "Southampton", 50.9058, -1.3910, 5),
    "sheffield-united": Venue("Bramall Lane", "Sheffield", 53.3703, -1.4709, 80),
    "luton": Venue("Kenilworth Road", "Luton", 51.8843, -0.4316, 120),
    "middlesbrough": Venue("Riverside Stadium", "Middlesbrough", 54.5782, -1.2170, 5),
    "coventry": Venue("Coventry Building Society Arena", "Coventry", 52.4481, -1.4956, 90),
    "west-brom": Venue("The Hawthorns", "West Bromwich", 52.5090, -1.9639, 160),
    "norwich": Venue("Carrow Road", "Norwich", 52.6221, 1.3092, 5),
    "watford": Venue("Vicarage Road", "Watford", 51.6498, -0.4015, 75),
    "hull": Venue("MKM Stadium", "Hull", 53.7465, -0.3679, 5),
    "millwall": Venue("The Den", "London", 51.4859, -0.0508, 5),
    "stoke": Venue("bet365 Stadium", "Stoke-on-Trent", 52.9884, -2.1756, 130),
    "wrexham": Venue("Racecourse Ground", "Wrexham", 53.0521, -3.0036, 80),
    "birmingham": Venue("St Andrew's", "Birmingham", 52.4757, -1.8682, 110),
    "derby": Venue("Pride Park", "Derby", 52.9150, -1.4472, 50),
    "swansea": Venue("Swansea.com Stadium", "Swansea", 51.6427, -3.9351, 10),
    "qpr": Venue("Loftus Road", "London", 51.5093, -0.2322, 20),
    "portsmouth": Venue("Fratton Park", "Portsmouth", 50.7964, -1.0639, 5),
    "preston": Venue("Deepdale", "Preston", 53.7722, -2.6881, 40),
    "bristol-city": Venue("Ashton Gate", "Bristol", 51.4400, -2.6203, 15),
    "charlton": Venue("The Valley", "London", 51.4865, 0.0364, 20),
    "blackburn": Venue("Ewood Park", "Blackburn", 53.7286, -2.4892, 110),
    "oxford": Venue("Kassam Stadium", "Oxford", 51.7164, -1.2081, 60),
    "sheffield-wednesday": Venue("Hillsborough", "Sheffield", 53.4115, -1.5007, 90),
    "cardiff": Venue("Cardiff City Stadium", "Cardiff", 51.4728, -3.2030, 10),
    "huddersfield": Venue("John Smith's Stadium", "Huddersfield", 53.6543, -1.7684, 80),
    "plymouth": Venue("Home Park", "Plymouth", 50.3882, -4.1509, 30),
    # Mexico
    "america": Venue("Estadio Azteca", "Mexico City", 19.3029, -99.1505, 2200),
    "cruz-azul": Venue("Estadio Ciudad de los Deportes", "Mexico City", 19.3834, -99.1782, 2240),
    "pumas": Venue("Estadio Olimpico Universitario", "Mexico City", 19.3321, -99.1921, 2280),
    "atlante": Venue("Estadio Ciudad de los Deportes", "Mexico City", 19.3834, -99.1782, 2240),
    "toluca": Venue("Estadio Nemesio Diez", "Toluca", 19.2872, -99.6667, 2660),
    "pachuca": Venue("Estadio Hidalgo", "Pachuca", 20.1047, -98.7556, 2400),
    "puebla": Venue("Estadio Cuauhtemoc", "Puebla", 19.0784, -98.1664, 2150),
    "queretaro": Venue("Estadio Corregidora", "Queretaro", 20.5774, -100.3664, 1820),
    "atletico-san-luis": Venue("Estadio Alfonso Lastras", "San Luis Potosi", 22.1343, -100.9486, 1860),
    "leon": Venue("Estadio Leon", "Leon", 21.1149, -101.6594, 1800),
    "guadalajara": Venue("Estadio Akron", "Zapopan", 20.6817, -103.4627, 1560),
    "atlas": Venue("Estadio Jalisco", "Guadalajara", 20.7050, -103.3278, 1560),
    "necaxa": Venue("Estadio Victoria", "Aguascalientes", 21.8819, -102.2758, 1880),
    "juarez": Venue("Estadio Olimpico Benito Juarez", "Ciudad Juarez", 31.7361, -106.4453, 1140),
    "monterrey": Venue("Estadio BBVA", "Guadalupe", 25.6692, -100.2445, 520),
    "tigres": Venue("Estadio Universitario", "San Nicolas de los Garza", 25.7228, -100.3114, 510),
    "santos-laguna": Venue("Estadio Corona", "Torreon", 25.6264, -103.3811, 1120),
    "tijuana": Venue("Estadio Caliente", "Tijuana", 32.5070, -116.9918, 30),
    "mazatlan": Venue("Estadio El Encanto", "Mazatlan", 23.2542, -106.4290, 10),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
