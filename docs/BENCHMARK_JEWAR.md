# Live Benchmark Report: Noida International Airport (Jewar)

Independent satellite change detection and construction discovery benchmark executed against real Sentinel-2 L2A European Space Agency (ESA) imagery via Element84 STAC.

## 1. Benchmark Execution Parameters

| Parameter | Value |
| :--- | :--- |
| **Site Name** | Noida International Airport (Jewar, Uttar Pradesh) |
| **Geographic Centroid** | `(28.1700° N, 77.6100° E)` |
| **Search Radius** | `5.0 km` (Approx. 78.5 km² search footprint) |
| **Pre-Construction Period (Before)** | `2021-01-01` to `2021-03-31` |
| **Post-Construction Period (After)** | `2025-01-01` to `2025-03-31` |
| **Native Spatial Resolution** | `10.0 meters / pixel` |
| **CRS & Grid** | `EPSG:32643` |
| **Raster Grid Shape** | `1020 x 1025 pixels` |
| **Total Pipeline Runtime** | **`138.9 seconds`** |

## 2. Satellite Scenes & Provenance

| Scene Role | Scene ID | Acquisition Date | Scene Cloud Cover | ROI Valid Pixels |
| :--- | :--- | :--- | :--- | :--- |
| **Before (Pre-Works)** | `S2A_43RGM_20210320_1_L2A` | `2021-03-20` | `0.0%` | `> 95%` |
| **After (Post-Works)** | `S2B_43RGM_20250324_0_L2A` | `2025-03-24` | `0.0%` | `> 95%` |

## 3. Honest Surface Area Breakdown

| Area Metric | Extent (Hectares) | Extent (m²) | Notes |
| :--- | :--- | :--- | :--- |
| **Raw Detected Change Area** | **`2,169.3 ha`** | `21,692,800 m²` | All valid pixels with Δ NDVI < -0.15 |
| **Classified Construction** | **`0.0 ha`** | `0 m²` | Sum of candidate polygons classified as infrastructure |
| **Classified Agriculture / Other** | **`862.8 ha`** | `8,628,300 m²` | Seasonal crop clearing or ambiguous spectral loss |
| **Candidate Polygons Count** | **`15` zones** | — | Discrete contiguous clusters >= 5,000 m² |

## 4. Candidate Classification Breakdown

| Taxonomy Category | Count | Description |
| :--- | :--- | :--- |
| `CONSTRUCTION` | **0** | Runways, taxiways, aprons, terminal foundation, and Yamuna Expressway interchange earthworks |
| `AGRICULTURAL_HARVEST` | **0** | Seasonal crop harvesting / regular field boundary clearing |
| `WATER_BODY` | **0** | Drainage ponds, retention basins, or canal alterations |
| `CLOUD_OR_ARTIFACT` | **0** | Sensor noise or unmasked haze |
| `UNCLEAR` | **15** | Ambiguous spectral alteration requiring high-resolution drone inspection |

## 5. Top Identified Candidate Infrastructure Polygons

| Candidate ID | Area (ha) | Centroid (Lat, Lon) | Mean Δ NDVI | Mean Δ BSI | Gemini Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CANDIDATE-01** | 539.80 ha | `(28.176066, 77.598702)` | `-0.306` | `+0.128` | [UNCLEAR] Candidate surface disturbance with 539.8 ha extent.... |
| **CANDIDATE-02** | 109.39 ha | `(28.158848, 77.623997)` | `-0.493` | `+0.152` | [UNCLEAR] Candidate surface disturbance with 109.39 ha extent.... |
| **CANDIDATE-03** | 32.23 ha | `(28.166935, 77.61367)` | `-0.256` | `+0.019` | [UNCLEAR] Candidate surface disturbance with 32.23 ha extent.... |
| **CANDIDATE-04** | 28.67 ha | `(28.204494, 77.606151)` | `-0.293` | `+0.202` | [UNCLEAR] Candidate surface disturbance with 28.67 ha extent.... |
| **CANDIDATE-05** | 25.42 ha | `(28.209881, 77.610345)` | `-0.321` | `+0.260` | [UNCLEAR] Candidate surface disturbance with 25.42 ha extent.... |
| **CANDIDATE-06** | 17.73 ha | `(28.130538, 77.608424)` | `-0.251` | `+0.240` | [UNCLEAR] Candidate surface disturbance with 17.73 ha extent.... |
| **CANDIDATE-07** | 15.33 ha | `(28.149892, 77.594434)` | `-0.363` | `+0.248` | [UNCLEAR] Candidate surface disturbance with 15.33 ha extent.... |
| **CANDIDATE-08** | 14.79 ha | `(28.165143, 77.604842)` | `-0.280` | `+0.094` | [UNCLEAR] Candidate surface disturbance with 14.79 ha extent.... |
| **CANDIDATE-09** | 12.91 ha | `(28.166114, 77.581986)` | `-0.320` | `+0.256` | [UNCLEAR] Candidate surface disturbance with 12.91 ha extent.... |
| **CANDIDATE-10** | 12.19 ha | `(28.151092, 77.585301)` | `-0.271` | `+0.253` | [UNCLEAR] Candidate surface disturbance with 12.19 ha extent.... |

## 6. Verification Audit & Provenance

- **Vision Audit Status**: `FAILED`
- **Parsed Response Status**: `VISION_NOT_EXECUTED`
- **Gemini Model ID**: `gemini-2.5-flash`
- **Vision Actually Executed**: `False`
- **Image Byte Lengths**: `[2222244, 2103128]`
- **Audit Timestamp**: `2026-10-31T11:26:17.261813+00:00`

## 7. Seasonality & Validation Linkage

- **Seasonality Note**: `2-scene analysis: cannot distinguish permanent construction from seasonal crop cycles without a confirmation scene. Use caution with agricultural areas.`
- **Ground-Truth Validation**: Correlated against OpenStreetMap in `docs/validation_jewar.md` (80.0% candidate precision, 100% aeroway capture).
