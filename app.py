from shiny import App, Inputs, Outputs, Session, render, ui
from shinywidgets import output_widget, render_widget
import ipyleaflet as ipyl
import streamstats
import pandas as pd
from typing import Any
import io, re, ezdxf
import os, io, traceback
import re
from math import radians, sin, cos, atan2, sqrt
import streamstats
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

app_dir = Path(__file__).parent
# USGS tools (pip install dataretrieval; streamstats package may require ArcGIS)
# import dataretrieval.nwis as nwis
import streamstats  # may require specific env (ArcGIS/ArcPy often ships its own)
import os
from arcmap import describe_point_admin_and_stream
from shiny_extra import (
    # ncft_to_wgs84, wgs84_to_ncft,
    find_points_within_radius, open_google_maps, decimal_to_dms,
    make_basin_character, Designername, AsstDesignername, Designdate,
    describe_point_admin_and_stream,
    # as_text, upper_or,qfmt, build_dxf_bytes,
)


from converting_loc import ncft_to_wgs84, wgs84_to_ncft
from helper import open_google_maps
from write_dxf import *

def safe_filename(s: str, default="drawing"):
    s = (s or default).strip()
    return re.sub(r"[^\w\-_. ]+", "_", s) or default



#ctrl+shift+p to open command palette in VS code
# ---------- UI ----------
app_ui = ui.page_fluid(
    ui.h4("Required Bridge/Culvert Locations", style="color:#cc0000;"),
    ui.layout_columns(
        ui.input_numeric("Easting", "Easting (ft US, EPSG:6543)", 995967.9559),
        ui.input_numeric("Northing", "Northing (ft US, EPSG:6543)", 826750.0042),
        ui.input_numeric("Elevation", "Elevation (ft US)", 2_382.0409),
        col_widths=(3, 3, 3),
    ),
    ui.h4("Converted WGS84 (lon/lat)"),
    ui.output_ui("latlon_out"),
    ui.h4("Required Design Inputs", style="color:#cc0000;"),
    ui.layout_columns(    
        ui.input_text("Placeholder1", "Project ID (replace with your own, #1 in CSR)", "Branch US19W South PDB"),
        ui.input_text("Placeholder2", "Feature ID (replace with your own, #2 in CSR)", "10426614"),
        col_widths=(6, 6),
    ),
    ui.layout_columns(    
        ui.input_text("StationText", "StationText #3 in CSR", "-L-STA 397+00"),
        ui.input_numeric("StationNumber", "Station Number", 397),
        col_widths=(6, 6),
    ),

    ui.layout_columns(
        ui.input_text("Designername", "Designer Name", "S. Fang"),
        ui.input_text("AsstDesignername", "Asst Designer Name", "J. Doe"),
        ui.input_text("Designdate", "Design Date", "2024-10-01"),
        col_widths=(3, 3, 3),
    ),
    ui.layout_columns(
        ui.input_text("Placeholder9", "Recommended Structure", "SINGLE 20' X20'-1 RC ARCH CULVERT;98' ALONG C/L CULVERT" ),             
        ui.input_text("Placeholder10", " Recommended Width of Roadway", "27'-10\" clearroadway"),
        ui.input_text("Placeholder11", "Skew-Specify skew of Structure", "0 DEGREES"),
        col_widths=(3, 3, 3), 
    ),
    ui.h4("Flood Survey data ", style="color:#cc0000;"),

    ui.output_ui("basin_character"),
    # ui.output_ui("watershed_map"),   # 👈 plot output
    ui.output_ui("google_link"),

    ui.h3("Nearest Stream Classification"),
    ui.layout_column_wrap(
        ui.input_text("Roadname1", "Road Name #1 in CSR", "SR-1385 (Piney Hill Rd) "),
        ui.input_text("Roadname2", "Road Name #2 in CSR", "US-19 West"),
    ),
    ui.download_button("download_dxf", "Download DXF", class_="btn-primary"),

    ui.hr(),
    # ui.download_button("download1", "Download Example CSV", class_="btn-primary"),
    ui.h3("DXF Export"),
    ui.input_text("bridge_station", "Bridge Station", "STA-10+00"),
    ui.input_numeric("DA", "Drainage Area (mi²)", 1.23),
    ui.input_text("basinchar", "Basin Character", "A/B"),
    ui.input_text("bridgenum", "Bridge Number", "B-1234"),
    ui.input_text("roadname1", "Road Name 1", "US-19W"),
    ui.input_text("roadname2", "Road Name 2", ""),
    ui.input_text("designer", "Designer", "Jane Doe"),
    ui.input_text("asst", "Asst. Designer", "John Roe"),
    ui.input_date("date", "Design Date"),
    # ui.output_text("stream_class"),
    output_widget("map"),
)

# ---------- Server ----------
def server(input, output, session):

    @render.ui
    def latlon_out():
        lon, lat = ncft_to_wgs84(input.Easting(), input.Northing())
        return ui.layout_columns(
            ui.card(
                ui.card_header("Longitude (°)"),
                ui.p(f"{lon:.6f}"),
            ),
            ui.card(
                ui.card_header("Latitude (°)"),
                ui.p(f"{lat:.6f}"),
            ),
            ui.card(
                ui.card_header("Elevation (US ft)"),
                ui.p(f"{input.Elevation():.2f}"),
            ),
            col_widths=(3, 3,3),
        )

    @render.ui
    def google_link():
        lon, lat = ncft_to_wgs84(input.Easting(), input.Northing())
        url = f"https://www.google.com/maps/search/?api=1&query={{lat}},{{lon}}".format(lat=lat, lon=lon)
        return ui.a(
            "🌍 Open Bridge/Culvert location in Google Maps",
            href=url,
            target="_blank",
            class_="gmaps-link",
        )
    
    @render.ui
    def basin_character():
        lon, lat = ncft_to_wgs84(input.Easting(), input.Northing())
        info = describe_point_admin_and_stream(
            lon, lat)
        info = describe_point_admin_and_stream(lon, lat)

        county_name = info.get("county_name") or "N/A"
        county_code = info.get("county_code") or "N/A"
        stream_name = info.get("stream_name") or "N/A"
        dist_m = info.get("stream_distance_m")
        stream_class_distance_m = info.get("stream_class_distance_m") or "N/A"
        stream_class = info.get("stream_class") or "N/A"
        dist_txt = f"{dist_m:.1f} m" if dist_m is not None else "N/A"
        return ui.div(
            ui.p(ui.strong("County: "), f"{county_name}"),
            ui.p(ui.strong("FIPS: "), f"{county_code}"),
            ui.p(ui.strong("Nearest stream: "), f"{stream_name}"),
            ui.p(ui.strong("Distance to stream: "), f"{dist_m:.1f}"),
            ui.p(ui.strong("Stream classification: "), f"{stream_class}"),
        )

    @render_widget
    def map():
        lon, lat = ncft_to_wgs84(input.Easting(), input.Northing())
        m = ipyl.Map(center=(lat, lon), zoom=18)  # ipyleaflet expects (lat, lon)
        m.add_layer(ipyl.Marker(location=(lat, lon)))
        return m
    
    @render.ui
    def watershed_map():
        # Get watershed polygon from StreamStats
        lon, lat = ncft_to_wgs84(input.Easting(), input.Northing())
        ws = streamstats.Watershed(lat=lat, lon=lon)
        ss_parameters = pd.DataFrame(ws.parameters)
        # ss_parameters.iloc[43:48,]#.values
        basin_character = ss_parameters.iloc[43:48,][ss_parameters.iloc[43:48,]['value']>1]['name'].values[0]
        # ss_parameters
        area_sq_mi = ss_parameters.iloc[3,:]['value']
        area_acres = area_sq_mi * 640 # convert to acres
        new_basin_character = make_basin_character(basin_character)      

        return new_basin_character
    

  
    @render.download(filename=lambda: f"{safe_filename(input.bridge_station())}_csrtex.dxf")
    def download_dxf():
        lon, lat = ncft_to_wgs84(input.Easting(), input.Northing())
        info = describe_point_admin_and_stream(
            lon, lat)
        info = describe_point_admin_and_stream(lon, lat)

        ctx = {
            "DA": input.DA(),
            "bridge_stationtxt": input.bridge_station(),
            "Countyname": info.get("county_name"),
            # "basinnames": input.basinchar(),
            # "Bridgenum": input.bridgenum(),
            "route_name": "",
            "roadname1": input.Roadname1(),
            "roadname2": input.Roadname2(),
            # "lat_overall": lat,
            # "lon_overall": lon,
            # "Designername": input.designer(),
            # "AsstDesignername": input.asst(),
            # "Designdate": str(input.date()) if input.date() else "",
            "area_sq_mi": None,
            "area_acres": None,
            # "Stream": res.get("stream_name"),
            # "new_basin_character": input.basinchar(),
            # "StreamClassification": res.get("stream_class"),
            "Placeholder1": "",
            "Placeholder2": "",
        }

        data = build_dxf_bytes(ctx)

        # Return a GENERATOR of bytes (Shiny expects an iterable)
        def byte_chunks(buf: bytes, size: int = 1024 * 1024):
            for i in range(0, len(buf), size):
                yield buf[i:i+size]

        return byte_chunks(data)

app = App(app_ui, server)
