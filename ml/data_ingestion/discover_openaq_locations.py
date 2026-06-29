import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

from ml.config import RAW_DATA_DIR, PROJECT_ROOT


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

CORE_PARAMETERS = {
    "pm25",
    "pm10",
    "no2",
    "o3",
    "co",
    "so2",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
}


def request_openaq(endpoint: str, params: dict) -> dict:
    url = f"{OPENAQ_BASE_URL}/{endpoint.lstrip('/')}"
    headers = {}

    api_key = os.getenv("OPENAQ_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=90,
    )

    if response.status_code == 401:
        raise RuntimeError("OpenAQ 401 Unauthorized. Check OPENAQ_API_KEY in .env.")

    if response.status_code == 422:
        raise RuntimeError(
            "OpenAQ 422 Unprocessable Entity.\n"
            f"URL: {response.url}\n"
            f"Response: {response.text}"
        )

    response.raise_for_status()
    return response.json()


def normalize_parameter_name(name):
    if name is None:
        return None

    text = str(name).lower().strip()

    if "pm25" in text or "pm2.5" in text:
        return "pm25"
    if "pm10" in text:
        return "pm10"
    if "no2" in text:
        return "no2"
    if "so2" in text:
        return "so2"
    if text == "co" or "co " in text:
        return "co"
    if "o3" in text:
        return "o3"
    if "temperature" in text:
        return "temperature"
    if "humidity" in text or "relativehumidity" in text:
        return "humidity"
    if "wind direction" in text or "wind_direction" in text:
        return "wind_direction"
    if "wind speed" in text or "wind_speed" in text:
        return "wind_speed"

    return text.replace(" ", "_")


def extract_coordinates(location: Dict[str, Any]) -> tuple[float | None, float | None]:
    coordinates = location.get("coordinates") or {}

    lat = coordinates.get("latitude")
    lon = coordinates.get("longitude")

    if lat is None:
        lat = location.get("latitude")

    if lon is None:
        lon = location.get("longitude")

    return lat, lon


def get_location_sensors(location_id: int) -> pd.DataFrame:
    payload = request_openaq(
        f"locations/{location_id}/sensors",
        params={
            "limit": 1000,
            "page": 1,
        },
    )

    rows = []

    for item in payload.get("results", []):
        parameter = item.get("parameter") or {}
        raw_name = parameter.get("name") or parameter.get("displayName") or item.get("name")
        normalized = normalize_parameter_name(raw_name)

        rows.append(
            {
                "location_id": location_id,
                "sensor_id": item.get("id"),
                "sensor_name": item.get("name"),
                "parameter": normalized,
                "raw_parameter": raw_name,
                "unit": item.get("unit") or parameter.get("units"),
            }
        )

    return pd.DataFrame(rows)


def score_location(parameters: set[str]) -> int:
    score = 0

    if "pm25" in parameters:
        score += 5
    if "pm10" in parameters:
        score += 5
    if "no2" in parameters:
        score += 2
    if "so2" in parameters:
        score += 1
    if "co" in parameters:
        score += 1
    if "o3" in parameters:
        score += 1
    if "temperature" in parameters:
        score += 1
    if "humidity" in parameters:
        score += 1
    if "wind_speed" in parameters:
        score += 2
    if "wind_direction" in parameters:
        score += 1

    return score


def discover_locations(
    latitude: float,
    longitude: float,
    radius_meters: int,
    max_pages: int,
    limit: int,
) -> pd.DataFrame:
    rows = []

    for page in range(1, max_pages + 1):
        print(f"Fetching locations page {page}...")

        payload = request_openaq(
            "locations",
            params={
                "coordinates": f"{latitude},{longitude}",
                "radius": min(radius_meters, 25000),
                "limit": limit,
                "page": page,
                "order_by": "id",
            },
        )

        results = payload.get("results", [])

        if not results:
            break

        for item in results:
            location_id = item.get("id")
            lat, lon = extract_coordinates(item)

            if location_id is None:
                continue

            try:
                sensors_df = get_location_sensors(int(location_id))
            except Exception as exc:
                print(f"Failed sensors for location {location_id}: {exc}")
                sensors_df = pd.DataFrame()

            parameters = set()

            if not sensors_df.empty and "parameter" in sensors_df.columns:
                parameters = set(
                    sensors_df["parameter"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

            core_parameters = sorted(parameters.intersection(CORE_PARAMETERS))
            usable_score = score_location(set(core_parameters))

            rows.append(
                {
                    "location_id": location_id,
                    "name": item.get("name"),
                    "locality": item.get("locality"),
                    "timezone": item.get("timezone"),
                    "latitude": lat,
                    "longitude": lon,
                    "last_updated": item.get("datetimeLast"),
                    "provider": (item.get("provider") or {}).get("name"),
                    "owner": (item.get("owner") or {}).get("name"),
                    "sensor_count": len(sensors_df),
                    "parameters": ",".join(core_parameters),
                    "has_pm25": "pm25" in core_parameters,
                    "has_pm10": "pm10" in core_parameters,
                    "has_pm25_and_pm10": (
                        "pm25" in core_parameters and "pm10" in core_parameters
                    ),
                    "has_weather": (
                        "temperature" in core_parameters
                        or "humidity" in core_parameters
                        or "wind_speed" in core_parameters
                    ),
                    "usable_score": usable_score,
                }
            )

        if len(results) < limit:
            break

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--latitude", type=float, default=13.0827)
    parser.add_argument("--longitude", type=float, default=80.2707)
    parser.add_argument("--radius-meters", type=int, default=250000)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="openaq_locations_chennai_tn_discovery.csv")

    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    locations_df = discover_locations(
        latitude=args.latitude,
        longitude=args.longitude,
        radius_meters=args.radius_meters,
        max_pages=args.max_pages,
        limit=args.limit,
    )

    if locations_df.empty:
        print("No locations found.")
        return

    locations_df = locations_df.sort_values(
        ["usable_score", "has_pm25_and_pm10", "has_pm25", "has_pm10"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    output_path = RAW_DATA_DIR / args.output
    locations_df.to_csv(output_path, index=False)

    print(f"\nSaved location discovery to: {output_path}")
    print(f"Shape: {locations_df.shape}")

    display_cols = [
        "location_id",
        "name",
        "latitude",
        "longitude",
        "last_updated",
        "parameters",
        "has_pm25",
        "has_pm10",
        "has_pm25_and_pm10",
        "has_weather",
        "usable_score",
    ]

    display_cols = [col for col in display_cols if col in locations_df.columns]

    print("\nTop usable locations:")
    print(locations_df[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()