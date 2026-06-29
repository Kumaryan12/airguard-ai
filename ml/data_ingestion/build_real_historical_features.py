import numpy as np
import pandas as pd

from ml.aqi.cpcb_aqi import calculate_cpcb_aqi
from ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


OPENMETEO_PATH = RAW_DATA_DIR / "chennai_openmeteo_weather_2026_june.csv"
INPUT_PATH = RAW_DATA_DIR / "chennai_openaq_historical_hourly.csv"
OSM_PATH = PROCESSED_DATA_DIR / "chennai_osm_station_features.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_historical_features.csv"


def estimate_simple_aqi(row: pd.Series) -> float:
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
        return "Moderately Polluted"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["night_stagnation"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    return df


def add_dispersion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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


def add_cpcb_aqi_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cpcb_results = []

    for _, row in df.iterrows():
        pollutants = {
            "pm25": row.get("pm25"),
            "pm10": row.get("pm10"),
            "no2": row.get("no2"),
            "so2": row.get("so2"),
            "co": row.get("co"),
            "o3": row.get("o3"),
        }

        result = calculate_cpcb_aqi(pollutants)
        cpcb_results.append(result)

    df["cpcb_aqi"] = [item.get("aqi") for item in cpcb_results]
    df["cpcb_aqi_category"] = [item.get("category") for item in cpcb_results]
    df["dominant_pollutant"] = [
        item.get("dominant_pollutant") for item in cpcb_results
    ]

    for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        df[f"{pollutant}_sub_index"] = [
            item.get("sub_indices", {}).get(pollutant, {}).get("sub_index")
            for item in cpcb_results
        ]

    return df

def add_cpcb_averaging_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pollutant averaging windows before CPCB AQI calculation.

    Prototype convention:
    - PM2.5, PM10, NO2, SO2 use 24-hour rolling average.
    - CO and O3 use 8-hour rolling average.
    - Rolling windows use current and past values only.
    """

    df = df.sort_values(["location_id", "timestamp"]).copy()

    averaging_config = {
        "pm25": 24,
        "pm10": 24,
        "no2": 24,
        "so2": 24,
        "co": 8,
        "o3": 8,
    }

    for pollutant, window in averaging_config.items():
        if pollutant not in df.columns:
            continue

        df[f"{pollutant}_cpcb_avg"] = (
            df.groupby("location_id")[pollutant]
            .transform(
                lambda s, window=window: s.rolling(
                    window=window,
                    min_periods=max(3, window // 2),
                ).mean()
            )
        )

    cpcb_results = []

    for _, row in df.iterrows():
        pollutants = {
            "pm25": row.get("pm25_cpcb_avg"),
            "pm10": row.get("pm10_cpcb_avg"),
            "no2": row.get("no2_cpcb_avg"),
            "so2": row.get("so2_cpcb_avg"),
            "co": row.get("co_cpcb_avg"),
            "o3": row.get("o3_cpcb_avg"),
        }

        result = calculate_cpcb_aqi(pollutants)
        cpcb_results.append(result)

    df["cpcb_window_aqi"] = [item.get("aqi") for item in cpcb_results]
    df["cpcb_window_aqi_category"] = [item.get("category") for item in cpcb_results]
    df["cpcb_window_dominant_pollutant"] = [
        item.get("dominant_pollutant") for item in cpcb_results
    ]

    for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        df[f"{pollutant}_window_sub_index"] = [
            item.get("sub_indices", {}).get(pollutant, {}).get("sub_index")
            for item in cpcb_results
        ]

    return df

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "dayofweek" in df.columns:
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    if "wind_speed" in df.columns and "wind_direction" in df.columns:
        direction_rad = np.deg2rad(df["wind_direction"])
        df["wind_u"] = df["wind_speed"] * np.sin(direction_rad)
        df["wind_v"] = df["wind_speed"] * np.cos(direction_rad)
        df["low_wind_flag"] = (df["wind_speed"] < 2.0).astype(int)

    if "humidity" in df.columns:
        df["high_humidity_flag"] = (df["humidity"] >= 80).astype(int)

    if "precipitation" in df.columns:
        df["rain_flag"] = (df["precipitation"] > 0).astype(int)
        df["post_rain_3h"] = (
            df.groupby("location_id")["rain_flag"]
            .transform(lambda s: s.rolling(3, min_periods=1).max().shift(1))
        )

    if "cpcb_aqi" in df.columns:
        df["cpcb_aqi_delta_1h"] = (
            df["cpcb_aqi"] - df.groupby("location_id")["cpcb_aqi"].shift(1)
        )
        df["cpcb_aqi_delta_3h"] = (
            df["cpcb_aqi"] - df.groupby("location_id")["cpcb_aqi"].shift(3)
        )

        for window in [6, 12, 24]:
            df[f"cpcb_aqi_rolling_max_{window}h"] = (
                df.groupby("location_id")["cpcb_aqi"]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=2).max())
            )

    if "pm10_pm25_ratio" in df.columns:
        df["pm10_pm25_ratio_lag_1h"] = (
            df.groupby("location_id")["pm10_pm25_ratio"].shift(1)
        )
        df["pm10_pm25_ratio_rolling_mean_6h"] = (
            df.groupby("location_id")["pm10_pm25_ratio"]
            .transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
        )

    if "road_density_km_per_km2" in df.columns and "pm10_pm25_ratio" in df.columns:
        df["road_dust_score"] = (
            df["road_density_km_per_km2"] * df["pm10_pm25_ratio"]
        )

    if "major_road_density_km_per_km2" in df.columns and "low_wind_flag" in df.columns:
        df["traffic_accumulation_score"] = (
            df["major_road_density_km_per_km2"] * (1 + df["low_wind_flag"])
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
        "cpcb_aqi",
        "pm25_sub_index",
        "pm10_sub_index",
        "no2_sub_index",
        "o3_sub_index",
        "co_sub_index",
        "cpcb_window_aqi",
"pm25_window_sub_index",
"pm10_window_sub_index",
"no2_window_sub_index",
"o3_window_sub_index",
"co_window_sub_index",
    ]

    for col in lag_columns:
        if col not in df.columns:
            continue

        for lag in [1, 3, 6, 12, 24]:
            df[f"{col}_lag_{lag}h"] = df.groupby("location_id")[col].shift(lag)

        for window in [3, 6, 12, 24]:
            df[f"{col}_rolling_mean_{window}h"] = (
                df.groupby("location_id")[col]
                .transform(
                    lambda s, window=window: s.shift(1).rolling(
                        window=window,
                        min_periods=max(2, window // 2),
                    ).mean()
                )
            )

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location_id", "timestamp"]).copy()

    target_cols = [
    "pm25",
    "pm10",
    "estimated_aqi",
    "cpcb_aqi",
    "cpcb_window_aqi",
    "pm10_sub_index",
    "pm25_sub_index",
    "pm10_window_sub_index",
    "pm25_window_sub_index",
]

    for col in target_cols:
        if col in df.columns:
            df[f"{col}_target_24h"] = df.groupby("location_id")[col].shift(-24)

    if "estimated_aqi_target_24h" in df.columns:
        df["high_pollution_event_24h"] = (
            df["estimated_aqi_target_24h"] >= 201
        ).astype(int)

    if "cpcb_aqi_target_24h" in df.columns:
        df["cpcb_high_pollution_event_24h"] = (
            df["cpcb_aqi_target_24h"] >= 101
        ).astype(int)
    
    if "cpcb_window_aqi_target_24h" in df.columns:
        df["cpcb_window_high_pollution_event_24h"] = (
            df["cpcb_window_aqi_target_24h"] >= 101
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


def merge_openmeteo_weather(df: pd.DataFrame) -> pd.DataFrame:
    if not OPENMETEO_PATH.exists():
        print("Open-Meteo weather file not found. Skipping weather merge.")
        return df

    weather = pd.read_csv(OPENMETEO_PATH)

    if weather.empty:
        print("Open-Meteo weather file is empty. Skipping weather merge.")
        return df

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        utc=True,
        errors="coerce",
    )
    weather["timestamp_hour"] = weather["timestamp"].dt.floor("h")

    df = df.copy()
    df["timestamp_hour"] = df["timestamp"].dt.floor("h")

    weather_keep = [
        "timestamp_hour",
        "temperature",
        "humidity",
        "precipitation",
        "wind_speed",
        "wind_direction",
        "surface_pressure",
    ]

    weather = weather[[col for col in weather_keep if col in weather.columns]].copy()

    merged = df.merge(
        weather,
        on="timestamp_hour",
        how="left",
        suffixes=("", "_openmeteo"),
    )

    for col in ["temperature", "humidity", "wind_speed", "wind_direction"]:
        fallback_col = f"{col}_openmeteo"

        if fallback_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[fallback_col]
            else:
                merged[col] = merged[col].fillna(merged[fallback_col])

    for col in ["precipitation", "surface_pressure"]:
        fallback_col = f"{col}_openmeteo"

        if fallback_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[fallback_col]
            else:
                merged[col] = merged[col].fillna(merged[fallback_col])

    drop_cols = [
        col
        for col in merged.columns
        if col.endswith("_openmeteo") or col == "timestamp_hour"
    ]

    merged = merged.drop(columns=drop_cols)

    return merged


def apply_physical_sanity_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sanity_ranges = {
        "pm25": (0, 1000),
        "pm10": (0, 1500),
        "no2": (0, 1000),
        "o3": (0, 1000),
        "so2": (0, 2000),
        "co": (0, 50000),
        "temperature": (-10, 60),
        "humidity": (0, 100),
        "wind_speed": (0, 40),
        "wind_direction": (0, 360),
        "precipitation": (0, 500),
        "surface_pressure": (800, 1100),
    }

    for col, (low, high) in sanity_ranges.items():
        if col not in df.columns:
            continue

        invalid_mask = (df[col] < low) | (df[col] > high)
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            print(f"Sanity filter: setting {invalid_count} invalid {col} values to NaN")

        df.loc[invalid_mask, col] = np.nan

    return df


def cap_extreme_aqi_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)

    df = df[
        (df["estimated_aqi"].notna())
        & (df["estimated_aqi"] >= 0)
        & (df["estimated_aqi"] <= 500)
    ].copy()

    after = len(df)

    print(f"AQI quality filter applied: {before} -> {after}")

    return df


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

    pivot = merge_openmeteo_weather(pivot)
    pivot = apply_physical_sanity_filters(pivot)

    pivot = add_time_features(pivot)
    pivot = add_dispersion_features(pivot)

    pivot["estimated_aqi"] = pivot.apply(estimate_simple_aqi, axis=1)
    pivot["estimated_aqi_category"] = pivot["estimated_aqi"].apply(classify_aqi)
    pivot = cap_extreme_aqi_values(pivot)

    if "pm25" in pivot.columns and "pm10" in pivot.columns:
        pivot["pm10_pm25_ratio"] = pivot["pm10"] / pivot["pm25"].replace(0, np.nan)
    else:
        pivot["pm10_pm25_ratio"] = np.nan

    pivot = add_cpcb_aqi_features(pivot)
    pivot = add_cpcb_averaging_window_features(pivot)
    pivot = attach_osm_features(pivot)
    pivot = add_advanced_features(pivot)
    pivot = add_lag_features(pivot)
    pivot = add_targets(pivot)

    required_target_cols = [
    "estimated_aqi",
    "estimated_aqi_target_24h",
    "cpcb_aqi",
    "cpcb_aqi_target_24h",
    "cpcb_window_aqi",
    "cpcb_window_aqi_target_24h",
]

    existing_required_cols = [
        col for col in required_target_cols if col in pivot.columns
    ]

    pivot = pivot.dropna(subset=existing_required_cols).reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved real historical feature table to: {OUTPUT_PATH}")
    print(f"Shape: {pivot.shape}")

    print("\nColumns:")
    print(pivot.columns.tolist())

    print("\nDate range:")
    print(pivot["timestamp"].min(), "→", pivot["timestamp"].max())

    print("\nParameter availability:")
    for col in [
        "pm25",
        "pm10",
        "no2",
        "o3",
        "co",
        "temperature",
        "humidity",
        "wind_speed",
        "cpcb_aqi",
    ]:
        if col in pivot.columns:
            print(col, "non-null:", pivot[col].notna().sum(), "/", len(pivot))

    print("\nHigh pollution event rate:")
    if "high_pollution_event_24h" in pivot.columns:
        print("estimated:", pivot["high_pollution_event_24h"].mean())
    if "cpcb_high_pollution_event_24h" in pivot.columns:
        print("cpcb:", pivot["cpcb_high_pollution_event_24h"].mean())

    print("\nAQI summary:")
    print("estimated_aqi:")
    print(pivot["estimated_aqi"].describe())

    if "cpcb_aqi" in pivot.columns:
        print("\ncpcb_aqi:")
        print(pivot["cpcb_aqi"].describe())

    print("\nPreview:")
    preview_cols = [
        "location_id",
        "timestamp",
        "pm25",
        "pm10",
        "estimated_aqi",
        "cpcb_aqi",
        "cpcb_aqi_target_24h",
        "dominant_pollutant",
        "wind_u",
        "wind_v",
        "road_dust_score",
        "traffic_accumulation_score",
    ]

    preview_cols = [col for col in preview_cols if col in pivot.columns]

    print(pivot[preview_cols].head())


if __name__ == "__main__":
    main()