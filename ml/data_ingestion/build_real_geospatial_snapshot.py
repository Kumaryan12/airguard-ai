import pandas as pd

from ml.config import PROCESSED_DATA_DIR


REAL_FEATURES_PATH = PROCESSED_DATA_DIR / "chennai_real_latest_features.csv"
OSM_FEATURES_PATH = PROCESSED_DATA_DIR / "chennai_osm_station_features.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_geospatial_snapshot.csv"


def infer_geospatial_source_hypotheses(row: pd.Series) -> list:
    hypotheses = []

    pm_ratio = row.get("pm10_pm25_ratio")
    road_density = row.get("road_density_km_per_km2")
    major_road_density = row.get("major_road_density_km_per_km2")
    industrial_count = row.get("industrial_poi_count")
    construction_count = row.get("construction_poi_count")
    green_count = row.get("green_poi_count")
    dispersion_risk = row.get("dispersion_risk")

    if pd.notna(pm_ratio) and pd.notna(road_density):
        if pm_ratio >= 2.5 and road_density >= 5:
            hypotheses.append(
                {
                    "source": "road dust / resuspension",
                    "evidence": [
                        f"PM10/PM2.5 ratio is high ({pm_ratio:.2f})",
                        f"road density is {road_density:.2f} km/km²",
                    ],
                    "confidence": "medium",
                }
            )

    if pd.notna(major_road_density) and major_road_density >= 1:
        hypotheses.append(
            {
                "source": "traffic emissions",
                "evidence": [
                    f"major road density is {major_road_density:.2f} km/km²",
                    "fresh station snapshot includes traffic-relevant pollutants where available",
                ],
                "confidence": "low-medium",
            }
        )

    if pd.notna(construction_count) and construction_count > 0:
        hypotheses.append(
            {
                "source": "construction dust",
                "evidence": [
                    f"{int(construction_count)} construction-tagged OSM features within radius",
                ],
                "confidence": "low-medium",
            }
        )

    if pd.notna(industrial_count) and industrial_count > 0:
        hypotheses.append(
            {
                "source": "industrial influence",
                "evidence": [
                    f"{int(industrial_count)} industrial-tagged OSM features within radius",
                ],
                "confidence": "low-medium",
            }
        )

    if dispersion_risk in ["medium", "high"]:
        hypotheses.append(
            {
                "source": "meteorological trapping / poor dispersion",
                "evidence": [
                    f"dispersion risk is {dispersion_risk}",
                ],
                "confidence": "medium",
            }
        )

    if pd.notna(green_count) and green_count == 0:
        hypotheses.append(
            {
                "source": "low green-buffering context",
                "evidence": [
                    "no OSM green-space features found within radius",
                ],
                "confidence": "low",
            }
        )

    if not hypotheses:
        hypotheses.append(
            {
                "source": "insufficient geospatial evidence",
                "evidence": [
                    "No strong geospatial source pattern detected from available OSM features",
                ],
                "confidence": "low",
            }
        )

    return hypotheses


def main() -> None:
    if not REAL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {REAL_FEATURES_PATH}. "
            "Run: python -m ml.data_ingestion.build_real_latest_features"
        )

    if not OSM_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {OSM_FEATURES_PATH}. "
            "Run: python -m ml.geospatial.fetch_osm_features"
        )

    real_df = pd.read_csv(REAL_FEATURES_PATH)
    osm_df = pd.read_csv(OSM_FEATURES_PATH)

    merged = real_df.merge(
        osm_df,
        on=["location_id", "latitude", "longitude"],
        how="left",
    )

    merged["geospatial_hypotheses"] = merged.apply(
        lambda row: infer_geospatial_source_hypotheses(row),
        axis=1,
    )

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved real geospatial snapshot to: {OUTPUT_PATH}")
    print(f"Shape: {merged.shape}")
    print("\nPreview:")
    print(merged.T)


if __name__ == "__main__":
    main()