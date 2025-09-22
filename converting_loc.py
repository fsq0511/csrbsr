# converting_loc.py

from pyproj import Transformer

_to_wgs84 = Transformer.from_crs(6543, 4326, always_xy=True)  # x,y -> lon,lat
_from_wgs84 = Transformer.from_crs(4326, 6543, always_xy=True) # lon,lat -> x,y (ftUS)

def ncft_to_wgs84(easting_ft: float, northing_ft: float):
    """EPSG:6543 (ftUS) -> EPSG:4326 (lon, lat)."""
    return _to_wgs84.transform(easting_ft, northing_ft)

def wgs84_to_ncft(lon: float, lat: float):
    """EPSG:4326 (lon, lat) -> EPSG:6543 (ftUS)."""
    return _from_wgs84.transform(lon, lat)