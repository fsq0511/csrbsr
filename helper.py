#helper.py


import re
from math import radians, sin, cos, atan2, sqrt
import pandas as pd
import webbrowser


# --------------------------
# String / formatting helpers
# --------------------------

def make_basin_character(s: str) -> str:
    """
    Example:
      'Percent Area in Region 2 - Blue Ridge' -> 'Region 2 (Blue Ridge)'
      Otherwise returns s unchanged.
    """
    m = re.search(r'Region\s*(\d+)\s*-\s*(.+)', s)
    if m:
        region_num = m.group(1)
        name = m.group(2).strip()
        return f"Region {region_num} ({name})"
    return s

def decimal_to_dms(dd: float):
    """
    Convert decimal degrees to (deg, min, sec).
    Preserves sign on degrees, minutes/seconds are absolute.
    """
    sign = -1 if dd < 0 else 1
    dd_abs = abs(dd)
    degrees = int(dd_abs)
    minutes = int((dd_abs - degrees) * 60)
    seconds = (dd_abs - degrees - minutes/60) * 3600
    return sign * degrees, minutes, seconds


# --------------------------
# Distance / GIS helpers
# --------------------------
def haversine(lat1, lon1, lat2, lon2) -> float:
    """
    Great-circle distance in kilometers between two (lat, lon) points.
    """
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.0)**2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0-a))
    return R * c

def find_points_within_radius(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    lat_col: str = "dec_lat_va",
    lon_col: str = "dec_long_va",
) -> pd.DataFrame:
    """
    Adds/updates a 'distance' column (km) and returns rows within radius_km.
    """
    df = df.copy()
    df.loc[:, "distance"] = df.apply(
        lambda row: haversine(center_lat, center_lon, row[lat_col], row[lon_col]),
        axis=1,
    )
    return df[df["distance"] <= radius_km]

# --------------------------
# Browser helpers
# --------------------------
def open_google_maps(latitude: float, longitude: float) -> None:
    """
    Opens Google Maps pinned at (lat, lon) in default browser.
    """
    url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
    webbrowser.open(url)