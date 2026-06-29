import argparse
from pathlib import Path

import pandas as pd
import requests

from ml.config import RAW_DATA_DIR


CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]


def fetch_openmeteo_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone,
    }

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(f"No hourly data returned: {payload}")

    hourly = payload["hourly"]

    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])

    df = df.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
        }
    )

    df["latitude"] = latitude
    df["longitude"] = longitude
    df["source"] = "open_meteo_archive"

    return df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-04-30")
    parser.add_argument("--latitude", type=float, default=CHENNAI_LAT)
    parser.add_argument("--longitude", type=float, default=CHENNAI_LON)
    parser.add_argument(
        "--output",
        default="chennai_openmeteo_weather.csv",
    )

    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_openmeteo_weather(
        latitude=args.latitude,
        longitude=args.longitude,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    output_path = RAW_DATA_DIR / args.output
    df.to_csv(output_path, index=False)

    print(f"Saved weather data to: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()