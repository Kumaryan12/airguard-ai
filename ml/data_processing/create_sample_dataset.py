import numpy as np
import pandas as pd
from pathlib import Path

from ml.config import PROCESSED_DATA_DIR, RANDOM_STATE


def create_sample_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    wards = [
        ("W01", "T Nagar", 13.0418, 80.2341),
        ("W02", "Guindy", 13.0108, 80.2206),
        ("W03", "Manali", 13.1667, 80.2667),
        ("W04", "Velachery", 12.9755, 80.2207),
        ("W05", "Anna Nagar", 13.0850, 80.2101),
        ("W06", "Adyar", 13.0067, 80.2578),
        ("W07", "Porur", 13.0382, 80.1565),
        ("W08", "Perambur", 13.1210, 80.2326),
        ("W09", "Tambaram", 12.9249, 80.1000),
        ("W10", "Royapuram", 13.1137, 80.2954),
    ]

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=24 * 90,
        freq="h",
    )

    rows = []

    for ward_id, ward_name, lat, lon in wards:
        road_density = rng.uniform(0.3, 0.95)
        construction_score = rng.uniform(0.05, 0.75)
        industrial_score = rng.uniform(0.05, 0.85)
        green_cover = rng.uniform(0.05, 0.55)

        for ts in timestamps:
            hour = ts.hour
            dayofweek = ts.dayofweek

            rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
            night_stagnation = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
            weekend = 1 if dayofweek >= 5 else 0

            temperature = 28 + 5 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.2)
            humidity = 68 - 8 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 4)
            wind_speed = max(0.2, rng.normal(2.2, 0.8))
            wind_direction = rng.uniform(0, 360)

            traffic_proxy = (
                0.45 * road_density
                + 0.35 * rush_hour
                - 0.15 * weekend
                + rng.normal(0, 0.08)
            )
            traffic_proxy = float(np.clip(traffic_proxy, 0, 1))

            dispersion_penalty = 1 / (wind_speed + 0.4)

            pm25 = (
                28
                + 32 * traffic_proxy
                + 18 * industrial_score
                + 13 * construction_score
                + 10 * night_stagnation
                + 18 * dispersion_penalty
                - 12 * green_cover
                + rng.normal(0, 7)
            )

            pm10 = (
                45
                + 25 * traffic_proxy
                + 35 * construction_score
                + 18 * road_density
                + 12 * dispersion_penalty
                - 10 * green_cover
                + rng.normal(0, 9)
            )

            no2 = (
                14
                + 26 * traffic_proxy
                + 8 * industrial_score
                + 5 * rush_hour
                + rng.normal(0, 4)
            )

            pm25 = max(5, pm25)
            pm10 = max(10, pm10)
            no2 = max(2, no2)

            aqi = max(pm25 * 2.2, pm10 * 1.4, no2 * 1.8)

            rows.append(
                {
                    "timestamp": ts,
                    "ward_id": ward_id,
                    "ward_name": ward_name,
                    "lat": lat,
                    "lon": lon,
                    "hour": hour,
                    "dayofweek": dayofweek,
                    "rush_hour": rush_hour,
                    "night_stagnation": night_stagnation,
                    "weekend": weekend,
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction,
                    "road_density": road_density,
                    "construction_score": construction_score,
                    "industrial_score": industrial_score,
                    "green_cover": green_cover,
                    "traffic_proxy": traffic_proxy,
                    "pm25": pm25,
                    "pm10": pm10,
                    "no2": no2,
                    "aqi": aqi,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["ward_id", "timestamp"]).reset_index(drop=True)

    df["pm25_target_24h"] = df.groupby("ward_id")["pm25"].shift(-24)
    df["pm10_target_24h"] = df.groupby("ward_id")["pm10"].shift(-24)
    df["aqi_target_24h"] = df.groupby("ward_id")["aqi"].shift(-24)

    df = df.dropna().reset_index(drop=True)

    return df


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = create_sample_dataset()
    output_path = PROCESSED_DATA_DIR / "sample_chennai_features.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved sample dataset to: {output_path}")
    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()