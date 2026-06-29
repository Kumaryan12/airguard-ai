import json
import ast
from datetime import datetime

import pandas as pd

from ml.config import PROCESSED_DATA_DIR, PROJECT_ROOT


INPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_geospatial_snapshot.csv"
OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "real_geospatial_snapshot.json"


def parse_hypotheses(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return []


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. "
            "Run: python -m ml.data_ingestion.build_real_geospatial_snapshot"
        )

    df = pd.read_csv(INPUT_PATH)

    stations = []

    for _, row in df.iterrows():
        stations.append(
            {
                "location_id": int(row["location_id"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "latest_datetime_utc": row["latest_datetime_utc"],
                "max_age_hours": float(row["max_age_hours"]),

                "pollutants": {
                    "pm25": None if pd.isna(row.get("pm25")) else float(row["pm25"]),
                    "pm10": None if pd.isna(row.get("pm10")) else float(row["pm10"]),
                    "no2": None if pd.isna(row.get("no2")) else float(row["no2"]),
                    "so2": None if pd.isna(row.get("so2")) else float(row["so2"]),
                    "co": None if pd.isna(row.get("co")) else float(row["co"]),
                    "o3": None if pd.isna(row.get("o3")) else float(row["o3"]),
                },

                "weather": {
                    "temperature": None if pd.isna(row.get("temperature")) else float(row["temperature"]),
                    "humidity": None if pd.isna(row.get("humidity")) else float(row["humidity"]),
                    "wind_speed": None if pd.isna(row.get("wind_speed")) else float(row["wind_speed"]),
                    "wind_direction": None if pd.isna(row.get("wind_direction")) else float(row["wind_direction"]),
                },

                "aqi_estimate": {
                    "estimated_aqi": None if pd.isna(row.get("estimated_aqi")) else float(row["estimated_aqi"]),
                    "estimated_aqi_category": row.get("estimated_aqi_category"),
                    "note": "Approximate AQI estimate; not official CPCB breakpoint AQI yet.",
                },

                "dispersion": {
                    "pm10_pm25_ratio": None if pd.isna(row.get("pm10_pm25_ratio")) else float(row["pm10_pm25_ratio"]),
                    "dispersion_penalty": None if pd.isna(row.get("dispersion_penalty")) else float(row["dispersion_penalty"]),
                    "dispersion_risk": row.get("dispersion_risk"),
                },

                "geospatial_features": {
                    "search_radius_meters": int(row["search_radius_meters"]),
                    "total_road_length_km": float(row["total_road_length_km"]),
                    "major_road_length_km": float(row["major_road_length_km"]),
                    "road_density_km_per_km2": float(row["road_density_km_per_km2"]),
                    "major_road_density_km_per_km2": float(row["major_road_density_km_per_km2"]),
                    "nearest_major_road_m": None if pd.isna(row.get("nearest_major_road_m")) else float(row["nearest_major_road_m"]),
                    "industrial_poi_count": int(row["industrial_poi_count"]),
                    "construction_poi_count": int(row["construction_poi_count"]),
                    "green_poi_count": int(row["green_poi_count"]),
                    "school_poi_count": int(row["school_poi_count"]),
                    "hospital_poi_count": int(row["hospital_poi_count"]),
                    "vulnerability_poi_count": int(row["vulnerability_poi_count"]),
                },

                "geospatial_hypotheses": parse_hypotheses(row.get("geospatial_hypotheses")),

                "data_status": row.get("data_status"),
                "data_source": row.get("data_source"),
                "geospatial_source": row.get("geospatial_source"),
            }
        )

    payload = {
        "project": "AirGuard AI",
        "city": "Chennai",
        "generated_at": datetime.utcnow().isoformat(),
        "output_type": "real_geospatial_sensor_snapshot",
        "description": (
            "Fresh OpenAQ station snapshot enriched with OSM geospatial features "
            "and dispersion/source-evidence hypotheses."
        ),
        "stations": stations,
        "limitations": [
            "AQI estimate is approximate and not yet official CPCB breakpoint AQI.",
            "Geospatial hypotheses are evidence signals, not causal proof.",
            "Current snapshot uses latest available OpenAQ station data; historical model training still uses mechanism-driven synthetic data.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved real geospatial snapshot JSON to: {OUTPUT_PATH}")
    print(f"Stations exported: {len(stations)}")


if __name__ == "__main__":
    main()