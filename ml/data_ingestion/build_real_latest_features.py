import numpy as np
import pandas as pd

from ml.config import PROCESSED_DATA_DIR


INPUT_PATH = PROCESSED_DATA_DIR / "chennai_openaq_latest_clean.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_latest_features.csv"


def estimate_simple_aqi(row: pd.Series) -> float:
    """
    First approximation for a demo feature table.
    This is not the official Indian AQI formula.
    Later we should replace this with CPCB breakpoint-based AQI.
    """
    candidates = []

    if pd.notna(row.get("pm25")):
        candidates.append(float(row["pm25"]) * 2.2)

    if pd.notna(row.get("pm10")):
        candidates.append(float(row["pm10"]) * 1.35)

    if pd.notna(row.get("no2")):
        candidates.append(float(row["no2"]) * 1.8)

    if pd.notna(row.get("o3")):
        candidates.append(float(row["o3"]) * 1.2)

    if not candidates:
        return np.nan

    return max(candidates)


def classify_aqi(aqi: float) -> str:
    if pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def compute_dispersion_features(row: pd.Series) -> dict:
    wind_speed = row.get("wind_speed")

    if pd.isna(wind_speed):
        wind_speed = 1.0

    wind_speed = max(float(wind_speed), 0.1)

    dispersion_penalty = 1.0 / (wind_speed + 0.35)

    if wind_speed <= 1.0:
        dispersion_risk = "high"
    elif wind_speed <= 2.0:
        dispersion_risk = "medium"
    else:
        dispersion_risk = "low"

    return {
        "dispersion_penalty": round(float(dispersion_penalty), 4),
        "dispersion_risk": dispersion_risk,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. "
            "Run: python -m ml.data_ingestion.clean_openaq_latest"
        )

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        print("No clean real-time station rows available.")
        df.to_csv(OUTPUT_PATH, index=False)
        return

    rows = []

    for _, row in df.iterrows():
        estimated_aqi = estimate_simple_aqi(row)
        dispersion = compute_dispersion_features(row)

        pm_ratio = np.nan
        if pd.notna(row.get("pm10")) and pd.notna(row.get("pm25")) and row.get("pm25") != 0:
            pm_ratio = float(row["pm10"]) / max(float(row["pm25"]), 1e-6)

        feature_row = {
            "location_id": row["location_id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "latest_datetime_utc": row["latest_datetime_utc"],
            "max_age_hours": row["max_age_hours"],

            "pm25": row.get("pm25", np.nan),
            "pm10": row.get("pm10", np.nan),
            "no2": row.get("no2", np.nan),
            "so2": row.get("so2", np.nan),
            "co": row.get("co", np.nan),
            "o3": row.get("o3", np.nan),

            "temperature": row.get("temperature", np.nan),
            "humidity": row.get("humidity", np.nan),
            "wind_speed": row.get("wind_speed", np.nan),
            "wind_direction": row.get("wind_direction", np.nan),

            "estimated_aqi": estimated_aqi,
            "estimated_aqi_category": classify_aqi(estimated_aqi),
            "pm10_pm25_ratio": pm_ratio,

            "dispersion_penalty": dispersion["dispersion_penalty"],
            "dispersion_risk": dispersion["dispersion_risk"],

            "data_source": "OpenAQ latest measurements",
            "data_status": "fresh_realtime_sensor_snapshot",
        }

        rows.append(feature_row)

    features = pd.DataFrame(rows)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved real latest feature table to: {OUTPUT_PATH}")
    print(f"Shape: {features.shape}")
    print("\nPreview:")
    print(features.head())


if __name__ == "__main__":
    main()