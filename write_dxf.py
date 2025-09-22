#write_dxf.py


import io
import ezdxf
from ezdxf.enums import TextEntityAlignment
from typing import List, Dict, Optional
import re

def safe_filename(s: str, default="drawing"):
    s = (s or default).strip()
    return re.sub(r"[^\w\-_. ]+", "_", s) or default

def as_text(v, default=""):
    try:
        return default if v is None else str(v)
    except Exception:
        return default

def upper_or(v, default=""):
    return as_text(v, default).upper()

def qfmt(a, b, da):
    try:
        da = float(da)
    except Exception:
        da = 0.0
    return f"{a * (da ** b):,.0f} cfs"


def build_dxf_bytes(ctx: dict) -> bytes:
    """
    Build DXF in memory (R2000; $INSUNITS set) and return bytes.
    Adds all text positions with safe fallbacks when ctx fields are empty.
    """
    # --- DXF doc
    doc = ezdxf.new("R2000", setup=True)
    doc.header["$INSUNITS"] = 2  # 1=in, 2=ft, 4=mm, 6=m, etc.
    msp = doc.modelspace()

    # --- helpers
    def as_text(v, default=""):
        try:
            s = default if v is None else str(v)
        except Exception:
            s = default
        # strip to avoid accidental spaces that look like blanks
        return s.strip()

    def upper_or(v, default=""):
        return as_text(v, default).upper()

    def qfmt(a, b, da):
        try:
            da_val = float(da)
        except Exception:
            da_val = 0.0
        return f"{a * (da_val ** b):,.0f} cfs"

    # --- inputs with sensible fallbacks so you can see text even if ctx is sparse
    DA  = ctx.get("DA", 0.0)
    bs  = as_text(ctx.get("bridge_stationtxt"), "STA-10+00")
    cnty= upper_or(ctx.get("Countyname"), "COUNTY")
    basin=upper_or(ctx.get("basinnames"), "BASIN")
    bnum= upper_or(ctx.get("Bridgenum"), "B-####")
    stream = upper_or(ctx.get("Stream"), "STREAM")
    basin_char = upper_or(ctx.get("new_basin_character"), "CHAR")
    sclass = upper_or(ctx.get("StreamClassification"), "CLASS")

    placeholder79a    = "USGS SIR 2023-5006"
    placeholder79Q10  = qfmt(191, 0.810, DA)
    placeholder79Q25  = qfmt(275, 0.790, DA)
    placeholder79Q50  = qfmt(355, 0.778, DA)
    placeholder79Q100 = qfmt(437, 0.766, DA)
    placeholder79Q500 = qfmt(646, 0.747, DA)

    # --- ALL text placements (same coords you provided)
    text_positions = [
        ((170.27, 327.01), "L"),
        ((78.34, 234.17), "FLOW"),
        ((47.26, 622.63), as_text(ctx.get("Placeholder1"), "")),
        ((201.46, 621.88), as_text(ctx.get("Placeholder2"), "")),
        ((346.25, 622.25), bs),                 # bridge_stationtxt
        ((-35.48, 602.49), cnty),               # Countyname
        ((135.34, 602.23), basin),              # basinnames
        ((360.5, 601.5),  bnum),                # Bridgenum
        ((560.0, 644.69), stream),              # Stream
        ((782.27, 645.05), basin_char),         # new_basin_character
        ((798.25, 625.44), sclass),             # StreamClassification
        ((1123.5, 461.44), placeholder79a),
        ((1123.5, 445.0),  placeholder79Q10),
        ((1123.5, 426.0),  placeholder79Q25),
        ((1123.5, 405.0),  placeholder79Q50),
        ((1123.5, 386.0),  placeholder79Q100),
        ((1123.5, 366.6),  placeholder79Q500),
    ]

    # --- write text (style=Standard; height=5)
    for (x, y), txt in text_positions:
        # skip truly-empty strings, otherwise add text
        if as_text(txt, "") != "":
            msp.add_text(
                as_text(txt),
                dxfattribs={"height": 5, "style": "Standard"},
            ).set_placement((float(x), float(y)))

    # --- in-memory write (StringIO -> bytes)
    s_buf = io.StringIO()
    doc.write(s_buf)                         # writes str
    data = s_buf.getvalue().encode("utf-8") # convert to bytes
    if not data:
        raise RuntimeError("DXF buffer is empty (doc.write produced no content)")
    return data

