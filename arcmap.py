# arcmap.py
# reading online arcmap database


import numpy as np
import requests

import pandas as pd
import geopandas as gpd
import webbrowser
from typing import List, Dict, Optional


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



nc_counties = arcgis_read_layer_url('https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_CountyBdy_Poly/MapServer/0')
nc_streams = arcgis_read_layer_url('https://services2.arcgis.com/kCu40SDxsCGcuUWO/arcgis/rest/services/SurfaceWaterClassifications/FeatureServer/0')
nc_bridges = arcgis_read_layer_url('https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Structures/MapServer/0')
nc_pipes = arcgis_read_layer_url('https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Structures/MapServer/1')
nc_culverts = arcgis_read_layer_url('https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Structures/MapServer/2')
svc = "https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Structures/MapServer"
layers = arcgis_list_layers(svc)
nc_roads = arcgis_read_layer(svc, layer_id=0)




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

