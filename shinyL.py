# shinyL.py
"""
Utility helpers for Shiny + mapping + hydro workflows.
Drop this file next to app.py and:  from shinyL import *
"""

# --- Imports you asked to include ---
import os
import re
import glob
import webbrowser
from math import radians, sin, cos, atan2, sqrt
import streamstats
import numpy as np
import pandas as pd
import geopandas as gpd
# import rasterio
# from rasterio.merge import merge
# from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib import colors
# import seaborn as sns  # optional; remove if not needed
from pathlib import Path

import pyproj
from pyproj import datadir, CRS

# USGS tools (pip install dataretrieval; streamstats package may require ArcGIS)
# import dataretrieval.nwis as nwis
import streamstats  # may require specific env (ArcGIS/ArcPy often ships its own)
from pyproj import Transformer
import os
# os.environ["GDAL_DATA"] = rf"{os.environ.get('CONDA_PREFIX')}\Library\share\gdal"
# os.environ["PROJ_LIB"] = rf"{os.environ.get('CONDA_PREFIX')}\Library\share\proj"

from pyproj import datadir
# datadir.set_data_dir(rf"{os.environ.get('CONDA_PREFIX')}\Library\share\proj")

# datadir.set_data_dir(r"C:\Users\sfang\AppData\Local\anaconda3\envs\usgs_env\Library\share\proj")
_to_wgs84 = Transformer.from_crs(6543, 4326, always_xy=True)  # x,y -> lon,lat
_from_wgs84 = Transformer.from_crs(4326, 6543, always_xy=True) # lon,lat -> x,y (ftUS)

def ncft_to_wgs84(easting_ft: float, northing_ft: float):
    """EPSG:6543 (ftUS) -> EPSG:4326 (lon, lat)."""
    return _to_wgs84.transform(easting_ft, northing_ft)

def wgs84_to_ncft(lon: float, lat: float):
    """EPSG:4326 (lon, lat) -> EPSG:6543 (ftUS)."""
    return _from_wgs84.transform(lon, lat)

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
# Default data root; change if needed
NC_DATA_DIR = Path(r"C:\Users\sfang\Documents\NCdata")

def _read_gdf(path: Path, to_epsg: int = 4326):
    """Read a shapefile to GeoDataFrame, reproject to EPSG, warn if missing."""
    if not path.exists():
        warnings.warn(f"[shinyL] Missing file: {path}")
        return None
    gdf = gpd.read_file(path)
    if to_epsg is not None:
        gdf = gdf.to_crs(epsg=to_epsg)
    return gdf

def load_nc_layers(
    data_dir: Path | str = NC_DATA_DIR, to_epsg: int = 4326
) -> dict:
    """
    Loads common NC GIS layers and returns a dict of GeoDataFrames (all in EPSG:to_epsg).
    Keys: counties, streams, bridges, surface_water, roads, culverts, huc12
    """
    data_dir = Path(data_dir)

    layers = {
        "counties": _read_gdf(
            data_dir / "ncgs_state_county_boundary" / "NC_State_County_Boundary1.shp",
            to_epsg,
        ),
        "streams": _read_gdf(
            data_dir
            / "North_Carolina_Stream_Centerlines_Effective"
            / "North_Carolina_Stream_Centerlines_Effective.shp",
            to_epsg,
        ),
        "bridges": _read_gdf(
            data_dir / "Bridge_Structures" / "Bridge_Structures.shp", to_epsg
        ),
        "surface_water": _read_gdf(
            data_dir
            / "SurfaceWaterClassifications"
            / "SurfaceWaterClassifications_prj.shp",
            to_epsg,
        ),
        "roads": _read_gdf(
            data_dir
            / "State_Maintained_Roads"
            / "State_Maintained_Roads_prj.shp",
            to_epsg,
        ),
        "culverts": _read_gdf(
            data_dir / "Culverts" / "Culverts.shp", to_epsg
        ),
        "huc12": _read_gdf(
            data_dir / "hydrologic_units" / "wbdhu12_a_nc.shp", to_epsg
        ),
    }
    return layers

# ---- Helpers: nearest feature safely (avoid geographic-distance warning) ----
def nearest_feature(
    lat: float,
    lon: float,
    gdf: gpd.GeoDataFrame,
    label_col: str | None = None,
    out_epsg: int = 4326,
) -> dict | None:
    """
    Returns dict with {'label','distance_m','geometry','row'} for the nearest feature.
    Projects to WebMercator for accurate nearest-distance search.
    """
    if gdf is None or gdf.empty:
        return None

    # Build point and project both to a metric CRS
    point = gpd.GeoDataFrame(
        {"_id": [0]}, geometry=gpd.points_from_xy([lon], [lat]), crs=f"EPSG:{out_epsg}"
    )
    gdf_m = gdf.to_crs(3857)
    pt_m = point.to_crs(3857)

    joined = gpd.sjoin_nearest(gdf_m, pt_m, how="left", distance_col="_dist_m")
    if joined.empty:
        return None

    row = joined.iloc[joined["_dist_m"].idxmin()]
    label = row.get(label_col) if label_col else None

    # Return in original EPSG
    geom = gpd.GeoSeries([row.geometry], crs=3857).to_crs(out_epsg).iloc[0]
    return {
        "label": label,
        "distance_m": float(row["_dist_m"]),
        "geometry": geom,
        "row": row,
    }

# ---- ipyleaflet helpers to add GeoDataFrames as toggleable layers -----------
def gdf_to_geojson_layer(
    gdf: gpd.GeoDataFrame,
    name: str,
    style: dict | None = None,
    hover_style: dict | None = None,
):
    """Convert a GeoDataFrame to an ipyleaflet GeoJSON layer."""
    if gdf is None or gdf.empty:
        return None
    gj = json.loads(gdf.to_json())
    return ipyl.GeoJSON(
        data=gj,
        name=name,
        style=style or {"opacity": 1, "fillOpacity": 0.1, "weight": 1},
        hover_style=hover_style or {"weight": 3},
    )

# --------------------------
# Project metadata (edit as needed)
# --------------------------
Designername = "JD"         # replace with your name
AsstDesignername = "JD"     # replace with your name
Designdate = "08-05-2025"   # replace with your date (MM-DD-YYYY)


Placeholder1 = "Branch US19W South PDB"  # replace with your project ID
# State Proj. Reference No. - Specify ID No. Example: 
# B-4494 (for TIP projects), BD-5112k (For Low Impact bridge project), 
# SF-890095 (State funded bridge projects),  FA-770077(Division Force Account projects).
Placeholder2 = "10426614"  # WBS Project No. - Specify Project WBS Number.
Placeholder9 = "SINGLE 20' X20'-1 RC ARCH CULVERT;98' ALONG C/L CULVERT" 
# Recommended Structure - Specify the number, size and type of culvert(s) proposed; additionally,
# include any other design features, e.g., sill height, embedded depth, baffle placement. If structure
# is being extended, specify extension length and direction of extension (up and/or downstream).4 
# Note culvert sizing is listed as Width x Height
Placeholder10 =  "27'-10\" clearroadway" # 10 Recommended Width of Roadway
Placeholder11  =  "0 DEGREES"  # 11 Skew - Specify skew of Structure
Placeholder43 = "Date"
Placeholder44 = "Elevation"  # (ft)

# shinyL.py (append this)
import geopandas as gpd
nc_counties = gpd.read_file(r"https://hdrinc-my.sharepoint.com/:u:/p/sfang/EflLf-T2od1FpPmOIQGLn6sBe19hHVAhlvtECc3SUVqZyA?email=fsq0511%40gmail.com&e=kPWtVQ").to_crs(epsg=4326)
nc_streams = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\North_Carolina_Stream_Centerlines_Effective\North_Carolina_Stream_Centerlines_Effective.shp").to_crs(epsg=4326)
nc_bridges = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\Bridge_Structures\Bridge_Structures.shp").to_crs(epsg=4326)
SurfaceWaterClassifications_data = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\SurfaceWaterClassifications\SurfaceWaterClassifications_prj.shp").to_crs(epsg=4326)
nc_roads = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\State_Maintained_Roads\State_Maintained_Roads_prj.shp").to_crs(epsg=4326)
nc_culverts = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\Culverts\Culverts.shp").to_crs(epsg=4326)
huc12_data = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\hydrologic_units\wbdhu12_a_nc.shp").to_crs(epsg=4326)
 

def describe_point_admin_and_stream(
    lon: float,
    lat: float,
    nc_counties=nc_counties,
    nc_streams= nc_streams,
    county_name_field: str = "County",
    county_code_field: str = "FIPS",
    stream_name_candidates: tuple[str, ...] = ("NC_FLOOD25", "NAME", "GNIS_NAME"),
) -> dict:
    """
    Returns {'county_name','county_code','stream_name','stream_distance_m'} for (lon,lat).
    - Counties are spatial-joined in EPSG:4326.
    - Streams nearest is computed in EPSG:3857 to avoid geographic-distance warnings.
    """
    # Build point (WGS84)
    pt = gpd.GeoDataFrame({"_id":[0]},
                          geometry=gpd.points_from_xy([lon], [lat]),
                          crs="EPSG:4326")

    # --- County name/code via sjoin (ensure same CRS) ---
    county_name = county_code = None
    if nc_counties is not None and not nc_counties.empty:
        cty = nc_counties
        if cty.crs is None or cty.crs.to_epsg() != 4326:
            cty = cty.to_crs(4326)
        joined = gpd.sjoin(pt, cty, how="left")
        if not joined.empty:
            row = joined.iloc[0]
            county_name = row[county_name_field] if county_name_field in row else None
            county_code = row[county_code_field] if county_code_field in row else None

    # --- Nearest stream name via sjoin_nearest in meters (EPSG:3857) ---
    stream_name = None
    stream_distance_m = None
    if nc_streams is not None and not nc_streams.empty:
        str_m = nc_streams
        if str_m.crs is None or str_m.crs.to_epsg() != 3857:
            str_m = str_m.to_crs(3857)
        pt_m = pt.to_crs(3857)
        near = gpd.sjoin_nearest(str_m, pt_m, how="left", distance_col="_dist_m")
        if not near.empty:
            row = near.iloc[near["_dist_m"].idxmin()]
            stream_distance_m = float(row["_dist_m"])
            # pick first available stream name field
            for cand in stream_name_candidates:
                if cand in row and pd.notna(row[cand]):
                    stream_name = row[cand]
                    break

    return {
        "county_name": county_name,
        "county_code": county_code,
        "stream_name": stream_name,
        "stream_distance_m": stream_distance_m,
    }






