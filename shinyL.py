# shinyL.py
"""
Utility helpers for Shiny + mapping + hydro workflows.
Drop this file next to app.py and:  from shinyL import *
"""

# --- Imports you asked to include ---
import os, io, traceback
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
import certifi
import requests

import pyproj
from pyproj import datadir, CRS
import math
import ezdxf
from ezdxf.enums import TextEntityAlignment
from typing import List, Dict, Optional

# USGS tools (pip install dataretrieval; streamstats package may require ArcGIS)
# import dataretrieval.nwis as nwis
import streamstats  # may require specific env (ArcGIS/ArcPy often ships its own)
from pyproj import Transformer
import os
# CONDA_PREFIX = os.environ.get("CONDA_PREFIX", r"C:\Users\sfang\AppData\Local\anaconda3\envs\shiny_env")

os.environ["GDAL_DATA"] = rf"{os.environ.get('CONDA_PREFIX')}\Library\share\gdal"
os.environ["PROJ_LIB"] = rf"{os.environ.get('CONDA_PREFIX')}\Library\share\proj"
# os.environ["SSL_CERT_FILE"] = certifi.where()

from pyproj import datadir
datadir.set_data_dir(rf"{os.environ.get('CONDA_PREFIX')}\Library\share\proj")

datadir.set_data_dir(r"C:\Users\sfang\AppData\Local\anaconda3\usgs_env\Library\share\proj")
_to_wgs84 = Transformer.from_crs(6543, 4326, always_xy=True)  # x,y -> lon,lat
_from_wgs84 = Transformer.from_crs(4326, 6543, always_xy=True) # lon,lat -> x,y (ftUS)

def ncft_to_wgs84(easting_ft: float, northing_ft: float):
    """EPSG:6543 (ftUS) -> EPSG:4326 (lon, lat)."""
    return _to_wgs84.transform(easting_ft, northing_ft)

def wgs84_to_ncft(lon: float, lat: float):
    """EPSG:4326 (lon, lat) -> EPSG:6543 (ftUS)."""
    return _from_wgs84.transform(lon, lat)
def arcgis_list_layers(service_url: str) -> list[dict]:
    """Return [{'id': 0, 'name': 'Layer name'}, ...] for a MapServer/FeatureServer."""
    url = service_url.rstrip("/") + "?f=pjson"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    meta = r.json()
    layers = []
    for sec in ("layers", "tables", "subLayers"):
        for lyr in meta.get(sec, []) or []:
            layers.append({"id": lyr["id"], "name": lyr.get("name", f"Layer {lyr['id']}")})
    return layers


def arcgis_read_layer(service_url: str, layer_id: int,
                      where: str = "1=1",
                      out_fields: str = "*",
                      chunk_size: int = 2000,
                      timeout: int = 120) -> gpd.GeoDataFrame:
    """
    Read a MapServer/FeatureServer layer into a GeoDataFrame via GeoJSON paging.
    """
    base = service_url.rstrip("/")
    layer_url = f"{base}/{layer_id}/query"

    # Discover OBJECTID field (helpful for stable paging)
    oid_field = None
    meta = requests.get(f"{base}/{layer_id}?f=pjson", timeout=timeout).json()
    for fld in meta.get("fields", []):
        if fld.get("type") == "esriFieldTypeOID":
            oid_field = fld["name"]
            break

    offset = 0
    frames: list[gpd.GeoDataFrame] = []

    while True:
        params = {
            "f": "geojson",
            "where": where,
            "outFields": out_fields,
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": chunk_size,
            "geometryPrecision": 7,
        }
        if oid_field:
            params["orderByFields"] = f"{oid_field} ASC"

        r = requests.get(layer_url, params=params, timeout=timeout)
        r.raise_for_status()

        # If empty or not GeoJSON, stop
        if "features" not in r.json():
            break

        gdf = gpd.read_file(r.text)  # GeoPandas can read GeoJSON text directly
        if gdf.empty:
            break

        frames.append(gdf)
        if len(gdf) < chunk_size:
            break
        offset += len(gdf)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return pd.concat(frames, ignore_index=True).set_crs(4326)


WEBMAP_ID = "c91dd1eff61a4456abad89fe0383114d"

def _get_webmap_data(webmap_id: str) -> dict:
    """
    Fetch the WebMap 'item data' JSON which contains operationalLayers.
    Tries arcgis.com, then the org domain used in your URL (ncdot.maps.arcgis.com).
    """
    portals = [
        "https://www.arcgis.com",
        "https://ncdot.maps.arcgis.com",
    ]
    last_err = None
    for portal in portals:
        url = f"{portal}/sharing/rest/content/items/{webmap_id}/data"
        try:
            r = requests.get(url, params={"f": "pjson"}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not fetch webmap data for {webmap_id}: {last_err}")

def list_operational_layers(webmap_id: str) -> List[Dict]:
    """
    Returns a flat list of layers with resolved layer URLs:
    [{'title','service_url','layer_id','layer_url','type'}]
    Handles Feature Layers and Map Image Layers with sublayers.
    """
    data = _get_webmap_data(webmap_id)
    out: List[Dict] = []

    for op in data.get("operationalLayers", []):
        title = op.get("title") or op.get("id") or "Layer"
        url = (op.get("url") or "").rstrip("/")
        # Feature layer (URL may already point to a sublayer)
        if url.endswith(("/FeatureServer", "/MapServer")):
            # If a single sublayer is referenced:
            if "layerId" in op and op["layerId"] is not None:
                lyr_url = f"{url}/{op['layerId']}"
                out.append({
                    "title": title,
                    "service_url": url,
                    "layer_id": op["layerId"],
                    "layer_url": lyr_url,
                    "type": "feature_or_map_layer",
                })
            # Map Image Layer with multiple sublayers listed
            for sub in (op.get("layers") or []):
                if "id" in sub:
                    lyr_url = f"{url}/{sub['id']}"
                    sub_name = sub.get("title") or sub.get("name") or f"{title} / {sub['id']}"
                    out.append({
                        "title": sub_name,
                        "service_url": url,
                        "layer_id": sub["id"],
                        "layer_url": lyr_url,
                        "type": "map_image_sublayer",
                    })
        # Some items are FeatureCollections or Tile layers (skip if not queryable)
    return out

def arcgis_read_layer_url(layer_url: str,
                          where: str = "1=1",
                          out_fields: str = "*",
                          chunk_size: int = 2000,
                          timeout: int = 90) -> gpd.GeoDataFrame:
    """
    Read a single FeatureServer/MapServer *layer URL* to GeoDataFrame via GeoJSON paging.
    """
    qurl = f"{layer_url.rstrip('/')}/query"
    # Try to discover OBJECTID for stable paging
    meta = requests.get(layer_url, params={"f": "pjson"}, timeout=timeout).json()
    oid_field = None
    for fld in meta.get("fields", []) or []:
        if fld.get("type") == "esriFieldTypeOID":
            oid_field = fld["name"]; break

    frames = []
    offset = 0
    while True:
        params = {
            "f": "geojson",
            "where": where,
            "outFields": out_fields,
            # "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": chunk_size,
            "geometryPrecision": 7,
        }
        if oid_field:
            params["orderByFields"] = f"{oid_field} ASC"

        r = requests.get(qurl, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        if not js or "features" not in js:
            break

        gdf = gpd.read_file(r.text)
        if gdf.empty:
            break
        frames.append(gdf)
        if len(gdf) < chunk_size:
            break
        offset += len(gdf)

    if not frames:
        return gpd.GeoDataFrame(geometry=[] )
    return pd.concat(frames, ignore_index=True) 


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
# # nc_counties = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\ncgs_state_county_boundary\NC_State_County_Boundary1.shp").to_crs(epsg=4326)
# nc_counties = arcgis_read_layer_url('https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_CountyBdy_Poly/MapServer/0')
# # nc_streams = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\North_Carolina_Stream_Centerlines_Effective\North_Carolina_Stream_Centerlines_Effective.shp").to_crs(epsg=4326)
# # nc_streams = arcgis_read_layer_url('https://spartagis.ncem.org/arcgis/rest/services/Public/FRIS_FloodZones/MapServer/0')
# nc_streams = arcgis_read_layer_url('https://services2.arcgis.com/kCu40SDxsCGcuUWO/arcgis/rest/services/SurfaceWaterClassifications/FeatureServer/0')
# # nc_bridges = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\Bridge_Structures\Bridge_Structures.shp").to_crs(epsg=4326)
# # SurfaceWaterClassifications_data = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\SurfaceWaterClassifications\SurfaceWaterClassifications_prj.shp").to_crs(epsg=4326)
# # nc_roads = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\State_Maintained_Roads\State_Maintained_Roads_prj.shp").to_crs(epsg=4326)
# # nc_culverts = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\Culverts\Culverts.shp").to_crs(epsg=4326)
# huc12_data = gpd.read_file(r"C:\Users\sfang\Documents\NCdata\hydrologic_units\wbdhu12_a_nc.shp").to_crs(epsg=4326)
# # SurfaceWaterClassifications_data = gpd.read_file(r'C:\Users\sfang\Documents\NCdata\SurfaceWaterClassifications\SurfaceWaterClassifications_prj.shp')
# # SurfaceWaterClassifications_data = SurfaceWaterClassifications_data.to_crs(epsg=4326) 
# nc_bridges = arcgis_read_layer_url('https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Structures/MapServer/0')
# nc_pipes = arcgis_read_layer_url('https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Structures/MapServer/1')
# nc_culverts = arcgis_read_layer_url('https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Structures/MapServer/2')
# svc = "https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Structures/MapServer"
# layers = arcgis_list_layers(svc)
# nc_roads = arcgis_read_layer(svc, layer_id=0)


# https://ncdot.maps.arcgis.com/
def describe_point_admin_and_stream(
    lon: float,
    lat: float,
    nc_counties=nc_counties,
    nc_streams= nc_streams,
    # SurfaceWaterClassifications_data = SurfaceWaterClassifications_data,
    county_name_field: str = "CountyName",
    county_code_field: str = "FIPS",
    stream_name_candidates: tuple[str, ...] = ("BIMS_Name", "NAME", "GNIS_NAME"),
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
    # Stream classification
    stream_class = None
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
            stream_class = row['BIMS_Class'] if 'BIMS_Class' in row else None
            # pick first available stream name field
            for cand in stream_name_candidates:
                if cand in row and pd.notna(row[cand]):
                    stream_name = row[cand]
                    break



    # if SurfaceWaterClassifications_data is not None and not SurfaceWaterClassifications_data.empty:
    #     swc = SurfaceWaterClassifications_data
    #     if swc.crs is None or swc.crs.to_epsg() != 3857:
    #         swc = swc.to_crs(3857)
    #     near_swc = gpd.sjoin_nearest(swc, pt_m, how="left", distance_col="_dist_m")
    #     if not near_swc.empty:
    #         row = near_swc.iloc[near_swc["_dist_m"].idxmin()]
    #         stream_class = row['BIMS_Class'] if 'BIMS_Class' in row else None
    #         if stream_class is not None:
    #             stream_name = f"{stream_name} ({stream_class})" if stream_name else stream_class





    return {
        "county_name": county_name,
        "county_code": county_code,
        "stream_name": stream_name,
        "stream_distance_m": stream_distance_m,
        'stream_class': stream_class,
    }





def ensure_text_style(doc, name="Arial", font_path=r"C:\Windows\Fonts\arial.ttf"):
    if name not in doc.styles:  # ezdxf table names are case-insensitive
        doc.styles.new(name, dxfattribs={"font": font_path})
    return name

def as_text(x, nd=None, default=""):
    """Robust to None/float/int/DataFrame scalar."""
    if x is None:
        return default
    try:
        # Allow optional numeric formatting
        if nd is not None and isinstance(x, (int, float)) and not isinstance(x, bool):
            return f"{float(x):.{nd}f}"
        return f"{x}"
    except Exception:
        return default

def upper_or(x, default=""):
    s = as_text(x, default=default)
    return s.upper() if s else default

def qfmt(coef: float, exp: float, DA: float) -> str:
    """Format your Q strings exactly like your example."""
    return f"{int(coef)} ({DA}){exp:.3f}={round(coef * (DA ** exp), 0):.0f} cfs"


def build_and_save_dxf(ctx: dict, out_dir: str, filename: str) -> str:
    """Build the DXF and save to disk using doc.saveas(); return full path."""
    import os, ezdxf
    from ezdxf.enums import TextEntityAlignment

    doc = ezdxf.new("R12", setup=True)
    style_name = ensure_text_style(doc, "Arial", r"C:\Windows\Fonts\arial.ttf")
    msp = doc.modelspace()

    V = lambda k, **kw: as_text(ctx.get(k), **kw)
    U = lambda k: upper_or(ctx.get(k))

    # flows
    try:
        DA = float(ctx.get("DA", 0.0))
    except Exception:
        DA = 0.0
    placeholder79a    = "USGS SIR 2023-5006"
    placeholder79Q10  = qfmt(191, 0.810, DA)
    placeholder79Q25  = qfmt(275, 0.790, DA)
    placeholder79Q50  = qfmt(355, 0.778, DA)
    placeholder79Q100 = qfmt(437, 0.766, DA)
    placeholder79Q500 = qfmt(646, 0.747, DA)

    # --- your positions (trimmed here; keep your full list) ---
    text_positions = [
        ((170.2666794829044, 327.0073871457213), "L"),
        (( 78.34252392965071, 234.1736691438797), "FLOW"),
        (( 47.26048400564468, 622.6254341508684), V("Placeholder1")),
        ((201.45995550527,   621.8792068884723), V("Placeholder2")),
        ((346.25,            622.25),            V("bridge_stationtxt")),
        ((-35.48340049540275, 602.492259339422), U("Countyname")),
        ((135.3440231234563,  602.2331646537131), U("basinnames")),
        ((360.5,              601.5),            U("Bridgenum")),
        ((560.0,              644.68835),        U("Stream")),
        ((782.2742555896256,  645.0538411218076), U("new_basin_character")),
        ((798.2499999999999,  625.438354),        U("StreamClassification")),  # #32
        ((1123.5,             461.43835),         placeholder79a),
        ((1123.5,             445),               placeholder79Q10),
        ((1123.5,             426),               placeholder79Q25),
        ((1123.5,             405),               placeholder79Q50),
        ((1123.5,             386),               placeholder79Q100),
        ((1123.5,             366.6),             placeholder79Q500),
    ]

    for (x, y), raw_txt in text_positions:
        txt = as_text(raw_txt, default="")
        msp.add_text(
            txt,
            dxfattribs={"height": 5, "style": style_name},
        ).set_placement((x, y), align=TextEntityAlignment.LEFT)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    doc.saveas(out_path)  # <-- write to disk
    return out_path


def safe_filename(name: str, default="drawing"):
    # simple sanitizer for Windows/macOS
    import re
    s = (name or default).strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    return s or default










