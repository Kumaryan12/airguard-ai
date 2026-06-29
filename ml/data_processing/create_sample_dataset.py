import numpy as np
import pandas as pd

from ml.config import PROCESSED_DATA_DIR, RANDOM_STATE


def create_sample_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    wards = [
        ("W01", "T Nagar", 13.0418, 80.2341, "commercial"),
        ("W02", "Guindy", 13.0108, 80.2206, "traffic_industrial"),
        ("W03", "Manali", 13.1667, 80.2667, "industrial"),
        ("W04", "Velachery", 12.9755, 80.2207, "residential"),
        ("W05", "Anna Nagar", 13.0850, 80.2101, "residential_traffic"),
        ("W06", "Adyar", 13.0067, 80.2578, "coastal_residential"),
        ("W07", "Porur", 13.0382, 80.1565, "construction_growth"),
        ("W08", "Perambur", 13.1210, 80.2326, "dense_traffic"),
        ("W09", "Tambaram", 12.9249, 80.1000, "suburban"),
        ("W10", "Royapuram", 13.1137, 80.2954, "port_industrial"),
    ]

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=24 * 120,
        freq="h",
    )

    citywide_stagnation_days = set(rng.choice(np.arange(5, 115), size=14, replace=False))
    citywide_waste_burning_days = set(rng.choice(np.arange(5, 115), size=10, replace=False))

    rows = []

    for ward_id, ward_name, lat, lon, ward_type in wards:
        road_density = rng.uniform(0.3, 0.95)
        construction_score = rng.uniform(0.05, 0.75)
        industrial_score = rng.uniform(0.05, 0.85)
        green_cover = rng.uniform(0.05, 0.55)
        population_vulnerability = rng.uniform(0.25, 0.9)

        if "industrial" in ward_type:
            industrial_score = max(industrial_score, rng.uniform(0.75, 0.95))

        if "traffic" in ward_type or "commercial" in ward_type:
            road_density = max(road_density, rng.uniform(0.75, 0.98))

        if "construction" in ward_type:
            construction_score = max(construction_score, rng.uniform(0.75, 0.95))

        ward_event_days = set(rng.choice(np.arange(7, 115), size=12, replace=False))

        for ts in timestamps:
            day_index = (ts - timestamps[0]).days
            hour = ts.hour
            dayofweek = ts.dayofweek

            rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
            night_stagnation = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
            weekend = 1 if dayofweek >= 5 else 0

            seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * day_index / 120)

            temperature = 28 + 5 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.2)
            humidity = 68 - 8 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 4)

            base_wind = 2.4 + 0.7 * np.sin(2 * np.pi * hour / 24)
            wind_speed = max(0.15, rng.normal(base_wind, 0.75))
            wind_direction = rng.uniform(0, 360)

            city_stagnation_event = int(day_index in citywide_stagnation_days and hour in range(5, 13))
            waste_burning_event = int(day_index in citywide_waste_burning_days and hour in [20, 21, 22, 23, 0, 1])
            local_event = int(day_index in ward_event_days and hour in range(7, 12))

            if city_stagnation_event:
                wind_speed = max(0.15, rng.normal(0.55, 0.18))
                humidity += rng.uniform(5, 12)

            traffic_proxy = (
                0.48 * road_density
                + 0.38 * rush_hour
                - 0.12 * weekend
                + 0.08 * seasonal_factor
                + rng.normal(0, 0.07)
            )
            traffic_proxy = float(np.clip(traffic_proxy, 0, 1))

            dispersion_penalty = 1 / (wind_speed + 0.35)

            traffic_component = 28 * traffic_proxy + 8 * rush_hour
            road_dust_component = 18 * road_density + 12 * traffic_proxy + 10 * (1 - green_cover)
            construction_component = 28 * construction_score + 35 * local_event * construction_score
            industrial_component = 26 * industrial_score + 20 * local_event * industrial_score
            waste_burning_component = 42 * waste_burning_event
            meteorology_component = 22 * dispersion_penalty + 35 * city_stagnation_event

            random_noise_pm25 = rng.normal(0, 6)
            random_noise_pm10 = rng.normal(0, 8)
            random_noise_no2 = rng.normal(0, 3)

            pm25 = (
                16
                + 0.85 * traffic_component
                + 0.35 * road_dust_component
                + 0.45 * construction_component
                + 0.75 * industrial_component
                + 0.90 * waste_burning_component
                + 0.95 * meteorology_component
                - 13 * green_cover
                + random_noise_pm25
            )

            pm10 = (
                28
                + 0.55 * traffic_component
                + 1.15 * road_dust_component
                + 1.05 * construction_component
                + 0.45 * industrial_component
                + 0.65 * waste_burning_component
                + 0.75 * meteorology_component
                - 10 * green_cover
                + random_noise_pm10
            )

            no2 = (
                10
                + 0.95 * traffic_component
                + 0.35 * industrial_component
                + 6 * rush_hour
                + random_noise_no2
            )

            pm25 = max(5, pm25)
            pm10 = max(10, pm10)
            no2 = max(2, no2)

            aqi = max(pm25 * 2.2, pm10 * 1.35, no2 * 1.8)

            source_components = {
                "traffic": traffic_component,
                "road_dust": road_dust_component,
                "construction": construction_component,
                "industrial": industrial_component,
                "waste_burning": waste_burning_component,
                "meteorology": meteorology_component,
            }

            dominant_source_label = max(source_components, key=source_components.get)

            rows.append(
                {
                    "timestamp": ts,
                    "ward_id": ward_id,
                    "ward_name": ward_name,
                    "ward_type": ward_type,
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
                    "population_vulnerability": population_vulnerability,
                    "traffic_proxy": traffic_proxy,
                    "city_stagnation_event": city_stagnation_event,
                    "waste_burning_event": waste_burning_event,
                    "local_event": local_event,
                    "traffic_component": traffic_component,
                    "road_dust_component": road_dust_component,
                    "construction_component": construction_component,
                    "industrial_component": industrial_component,
                    "waste_burning_component": waste_burning_component,
                    "meteorology_component": meteorology_component,
                    "dominant_source_label": dominant_source_label,
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

    df["dominant_source_target_24h"] = df.groupby("ward_id")[
        "dominant_source_label"
    ].shift(-24)

    df["high_pollution_event_24h"] = (df["aqi_target_24h"] >= 201).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = create_sample_dataset()
    output_path = PROCESSED_DATA_DIR / "sample_chennai_features.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved sample dataset to: {output_path}")
    print(f"Shape: {df.shape}")

    print("\nHigh pollution event rate:")
    print(df["high_pollution_event_24h"].mean())

    print("\nDominant source distribution:")
    print(df["dominant_source_label"].value_counts(normalize=True).round(3))

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()