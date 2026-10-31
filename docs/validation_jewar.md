# Ground-Truth Validation: Noida International Airport (Jewar)

- **Validation Timestamp**: `2026-10-31 11:22:43 UTC`
- **Target Bounding Box**: `[min_lon=77.56, min_lat=28.125, max_lon=77.66, max_lat=28.215]`
- **OSM Data Source**: `LIVE_OPENSTREETMAP_API (api.openstreetmap.org)` (Overpass API)
- **Discovery GeoJSON**: `D:\forge\outputs\construction\discovery\DISC-20261031-E0DFC6\candidates.geojson`
- **Total Candidates Evaluated**: `15`
- **Total OSM Features in ROI**: `470`

## Overall Spatial Agreement Summary

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Candidate Intersection Precision** | **80.0%** (12/15) | Fraction of satellite-detected change polygons overlapping verified OSM infrastructure |
| **OSM Feature Capture Rate (Recall)** | **33.2%** (156/470) | Fraction of known OSM airport infrastructure overlapping satellite candidates |

## OSM Infrastructure Category Breakdown

| Feature Category | Total Features in ROI | Captured by Satellite Candidates | Category Capture Rate |
| :--- | :--- | :--- | :--- |
| `aeroway` | 17 | 17 | **100.0%** |
| `building` | 13 | 9 | **69.2%** |
| `highway` | 440 | 130 | **29.5%** |

## Top Candidate Polygons vs OSM Ground-Truth

| Candidate ID | Area (ha) | Gemini Classification | Intersecting OSM Ground-Truth Features |
| :--- | :--- | :--- | :--- |
| **CANDIDATE-01** | 539.8 | `UNCLEAR` | highway (motorway) (Yamuna Expressway), highway (motorway) (Yamuna Expressway), highway (motorway) (Yamuna Expressway) |
| **CANDIDATE-02** | 109.39 | `UNCLEAR` | aeroway (aerodrome) (Noida International Airport), highway (unclassified) (Unnamed), highway (unclassified) (Unnamed) |
| **CANDIDATE-03** | 32.23 | `UNCLEAR` | aeroway (aerodrome) (Noida International Airport), highway (track) (Unnamed), highway (unclassified) (Unnamed) |
| **CANDIDATE-04** | 28.67 | `UNCLEAR` | highway (track) (Unnamed), highway (track) (Unnamed), highway (track) (Unnamed) |
| **CANDIDATE-05** | 25.42 | `UNCLEAR` | highway (residential) (Unnamed), highway (residential) (Unnamed), highway (residential) (Unnamed) |
| **CANDIDATE-06** | 17.73 | `UNCLEAR` | highway (unclassified) (Chauroli road), highway (unclassified) (Unnamed) |
| **CANDIDATE-07** | 15.33 | `UNCLEAR` | highway (unclassified) (Unnamed) |
| **CANDIDATE-08** | 14.79 | `UNCLEAR` | aeroway (aerodrome) (Noida International Airport), highway (residential) (Unnamed), highway (residential) (Unnamed) |
| **CANDIDATE-09** | 12.91 | `UNCLEAR` | aeroway (aerodrome) (Noida International Airport), highway (unclassified) (Unnamed) |
| **CANDIDATE-10** | 12.19 | `UNCLEAR` | highway (residential) (Unnamed) |

## Honest Methodology Notes & Limitations

1. **Sentinel-2 Spatial Resolution (10m)**: Individual narrow access tracks (<10m wide) or small utility buildings may not produce large contiguous spectral clusters exceeding the 5,000 m² minimum threshold.
2. **OpenStreetMap Recency**: OpenStreetMap is community-curated; ongoing Jewar airport works (runway earthworks, apron clearing) may be captured in Sentinel-2 imagery before volunteers digitize them into OSM.
3. **Zero Fabrication**: All coordinates, intersections, and percentages are derived deterministically by Shapely geometry intersection between real STAC Sentinel-2 rasters and real Overpass API data.
