import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
load_dotenv(root_dir / ".env")

import pystac_client
import rasterio
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import rasterio.features
import numpy as np
from PIL import Image
from shapely.geometry import shape

def run_test():
    lat = 28.17
    lon = 77.61
    radius_km = 5.0
    
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * np.cos(np.radians(lat)))
    bbox = [round(lon - delta_lon, 6), round(lat - delta_lat, 6), round(lon + delta_lon, 6), round(lat + delta_lat, 6)]
    print(f"Bounding Box: {bbox}")

    client = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

    # Search 2021
    s2021 = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2021-01-01T00:00:00Z/2021-03-31T23:59:59Z",
        query={"eo:cloud_cover": {"lt": 5}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=1,
    )
    item2021 = list(s2021.items())[0]
    print(f"2021 Scene: {item2021.id} on {item2021.datetime} (cloud: {item2021.properties.get('eo:cloud_cover')}%)")

    # Search 2025
    s2025 = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2025-01-01T00:00:00Z/2025-03-31T23:59:59Z",
        query={"eo:cloud_cover": {"lt": 5}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=1,
    )
    item2025 = list(s2025.items())[0]
    print(f"2025 Scene: {item2025.id} on {item2025.datetime} (cloud: {item2025.properties.get('eo:cloud_cover')}%)")

    out_dir = root_dir / "outputs" / "construction" / "jewar_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    def read_band(asset_url, target_shape=None, resampling=Resampling.bilinear):
        with rasterio.open(asset_url) as src:
            nb = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*nb, transform=src.transform)
            if target_shape:
                data = src.read(1, window=win, out_shape=target_shape, resampling=resampling, boundless=True)
            else:
                data = src.read(1, window=win, boundless=True)
            return data.astype(np.float32), src.crs, rasterio.windows.transform(win, src.transform)

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True):
        r1, crs, win_transform = read_band(item2021.assets["red"].href)
        target_shape = r1.shape
        g1, _, _ = read_band(item2021.assets["green"].href, target_shape=target_shape)
        b1, _, _ = read_band(item2021.assets["blue"].href, target_shape=target_shape)
        n1, _, _ = read_band(item2021.assets["nir"].href, target_shape=target_shape)
        scl1, _, _ = read_band(item2021.assets["scl"].href, target_shape=target_shape, resampling=Resampling.nearest)

        r2, _, _ = read_band(item2025.assets["red"].href, target_shape=target_shape)
        g2, _, _ = read_band(item2025.assets["green"].href, target_shape=target_shape)
        b2, _, _ = read_band(item2025.assets["blue"].href, target_shape=target_shape)
        n2, _, _ = read_band(item2025.assets["nir"].href, target_shape=target_shape)
        scl2, _, _ = read_band(item2025.assets["scl"].href, target_shape=target_shape, resampling=Resampling.nearest)

    print(f"Successfully aligned all 10 bands. Shape: {target_shape}")

    # Invalid mask
    invalid_classes = {0, 1, 3, 8, 9, 10}
    mask_invalid1 = np.isin(scl1.astype(int), list(invalid_classes))
    mask_invalid2 = np.isin(scl2.astype(int), list(invalid_classes))
    valid_mask = ~(mask_invalid1 | mask_invalid2)

    ndvi1 = (n1 - r1) / (n1 + r1 + 1e-7)
    ndvi2 = (n2 - r2) / (n2 + r2 + 1e-7)
    delta_ndvi = ndvi2 - ndvi1

    # Change mask
    change_mask = (delta_ndvi < -0.15) & valid_mask
    disturbed_pixels = int(np.sum(change_mask))
    disturbed_area_m2 = disturbed_pixels * 100.0
    print(f"Total Disturbed Area: {disturbed_area_m2:,.0f} m2 ({disturbed_area_m2/10000:,.1f} ha)")

    # Vectorize
    raw_shapes = list(rasterio.features.shapes(change_mask.astype(np.uint8), mask=change_mask, transform=win_transform))
    print(f"Extracted {len(raw_shapes)} raw candidate shapes")

    candidates = []
    min_area_m2 = 10000.0  # >= 1 hectare
    for geom_utm, val in raw_shapes:
        if val != 1:
            continue
        poly_utm = shape(geom_utm)
        area_m2 = poly_utm.area
        if area_m2 < min_area_m2:
            continue
        geom_wgs84 = transform_geom(str(crs), "EPSG:4326", geom_utm)
        poly_wgs84 = shape(geom_wgs84)
        c_lon, c_lat = poly_wgs84.centroid.x, poly_wgs84.centroid.y
        bounds = poly_wgs84.bounds
        candidates.append({
            "centroid": {"latitude": round(c_lat, 6), "longitude": round(c_lon, 6)},
            "bbox": [round(b, 6) for b in bounds],
            "area_m2": round(area_m2, 1),
            "area_hectares": round(area_m2 / 10000.0, 2),
            "geometry": geom_wgs84,
        })

    candidates.sort(key=lambda x: x["area_m2"], reverse=True)
    print(f"Filtered to {len(candidates)} candidate polygons (>= 1 ha)")
    for i, c in enumerate(candidates[:5]):
        print(f"  Candidate #{i+1}: Centroid ({c['centroid']['latitude']}, {c['centroid']['longitude']}) - Area: {c['area_hectares']} ha ({c['area_m2']:,.0f} m2)")

if __name__ == "__main__":
    run_test()
