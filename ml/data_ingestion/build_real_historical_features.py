import numpy as np
import pandas as pd

from ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


INPUT_PATH = RAW_DATA_DIR / "chennai_openaq_historical_hourly.csv"
OSM_PATH = PROCESSED_DATA_DIR / "chennai_osm_station_features.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_historical_features.csv"


def estimate_simple_aqi(row: pd.Series) -> float:
    """
    First real-data AQI approximation.
    This is not official CPCB breakpoint AQI yet.
    Later we should replace this with CPCB breakpoint calculation.
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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["night_stagnation"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    return df


def add_dispersion_features(df: pd.DataFrame) -> pd.DataFrame:
    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

    wind = df["wind_speed"].fillna(1.0).clip(lower=0.1)
    df["dispersion_penalty"] = 1 / (wind + 0.35)

    df["dispersion_risk"] = np.select(
        [
            wind <= 1.0,
            wind <= 2.0,
        ],
        [
            "high",
            "medium",
        ],
        default="low",
    )

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location_id", "timestamp"]).copy()

    lag_columns = [
        "pm25",
        "pm10",
        "no2",
        "o3",
        "co",
        "temperature",
        "humidity",
        "wind_speed",
        "estimated_aqi",
    ]

    for col in lag_columns:
        if col not in df.columns:
            continue

        for lag in [1, 3, 6, 12, 24]:
            df[f"{col}_lag_{lag}h"] = df.groupby("location_id")[col].shift(lag)

        for window in [3, 6, 12, 24]:
            df[f"{col}_rolling_mean_{window}h"] = (
                df.groupby("location_id")[col]
                .shift(1)
                .rolling(window=window, min_periods=max(2, window // 2))
                .mean()
                .reset_index(level=0, drop=True)
            )

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    target_cols = ["pm25", "pm10", "estimated_aqi"]

    for col in target_cols:
        if col in df.columns:
            df[f"{col}_target_24h"] = df.groupby("location_id")[col].shift(-24)

    df["high_pollution_event_24h"] = (
        df["estimated_aqi_target_24h"] >= 201
    ).astype(int)

    return df


def attach_osm_features(df: pd.DataFrame) -> pd.DataFrame:
    if not OSM_PATH.exists():
        return df

    osm = pd.read_csv(OSM_PATH)

    keep_cols = [
        "location_id",
        "road_density_km_per_km2",
        "major_road_density_km_per_km2",
        "nearest_major_road_m",
        "industrial_poi_count",
        "construction_poi_count",
        "green_poi_count",
        "vulnerability_poi_count",
    ]

    osm = osm[[col for col in keep_cols if col in osm.columns]]

    return df.merge(osm, on="location_id", how="left")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}. "
            "Run: python -m ml.data_ingestion.fetch_openaq_historical"
        )

    raw = pd.read_csv(INPUT_PATH)

    if raw.empty:
        raise ValueError("Historical OpenAQ file is empty.")

    raw["timestamp"] = pd.to_datetime(raw["datetime_utc"], utc=True)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    pivot = (
        raw.pivot_table(
            index=["location_id", "timestamp"],
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    pivot = pivot.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    pivot = add_time_features(pivot)
    pivot = add_dispersion_features(pivot)

    pivot["estimated_aqi"] = pivot.apply(estimate_simple_aqi, axis=1)
    pivot["estimated_aqi_category"] = pivot["estimated_aqi"].apply(classify_aqi)

    if "pm25" in pivot.columns and "pm10" in pivot.columns:
        pivot["pm10_pm25_ratio"] = pivot["pm10"] / pivot["pm25"].replace(0, np.nan)
    else:
        pivot["pm10_pm25_ratio"] = np.nan

    pivot = attach_osm_features(pivot)

    pivot = add_lag_features(pivot)
    pivot = add_targets(pivot)

    required_target_cols = [
        "estimated_aqi",
        "estimated_aqi_target_24h",
    ]

    pivot = pivot.dropna(subset=required_target_cols).reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved real historical feature table to: {OUTPUT_PATH}")
    print(f"Shape: {pivot.shape}")

    print("\nColumns:")
    print(pivot.columns.tolist())

    print("\nDate range:")
    print(pivot["timestamp"].min(), "→", pivot["timestamp"].max())

    print("\nParameter availability:")
    for col in ["pm25", "pm10", "no2", "o3", "co", "temperature", "humidity", "wind_speed"]:
        if col in pivot.columns:
            print(col, "non-null:", pivot[col].notna().sum(), "/", len(pivot))

    print("\nHigh pollution event rate:")
    print(pivot["high_pollution_event_24h"].mean())

    print("\nAQI summary:")
    print(pivot["estimated_aqi"].describe())

    print("\nPreview:")
    print(pivot.head())


if __name__ == "__main__":
    main()