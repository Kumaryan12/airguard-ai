import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from ml.config import RAW_DATA_DIR, PROJECT_ROOT
from datetime import datetime, timezone

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

OPENAQ_BASE_URL = "https://api.openaq.org/v3"


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
        timeout=60,
    )

    if response.status_code == 401:
        raise RuntimeError(
            "OpenAQ returned 401 Unauthorized. "
            "Check OPENAQ_API_KEY in .env."
        )

    if response.status_code == 422:
        raise RuntimeError(
            "OpenAQ returned 422 Unprocessable Entity.\n"
            f"URL: {response.url}\n"
            f"Response: {response.text}"
        )

    response.raise_for_status()

    payload = response.json()

    if payload is None:
        raise RuntimeError(
            "OpenAQ returned an empty JSON response.\n"
            f"URL: {response.url}"
        )

    return payload

def compute_freshness_status(datetime_utc: str) -> dict:
    if not datetime_utc:
        return {
            "age_hours": None,
            "freshness_status": "missing_timestamp",
            "usable_for_realtime": False,
        }

    try:
        measured_at = pd.to_datetime(datetime_utc, utc=True)
        now = pd.Timestamp.now(tz="UTC")
        age_hours = (now - measured_at).total_seconds() / 3600

        if age_hours <= 6:
            status = "fresh"
            usable = True
        elif age_hours <= 24:
            status = "recent"
            usable = True
        elif age_hours <= 168:
            status = "stale"
            usable = False
        else:
            status = "archival"
            usable = False

        return {
            "age_hours": round(float(age_hours), 2),
            "freshness_status": status,
            "usable_for_realtime": usable,
        }

    except Exception:
        return {
            "age_hours": None,
            "freshness_status": "invalid_timestamp",
            "usable_for_realtime": False,
        }
def search_locations(city: str, country: str = "IN", limit: int = 100) -> pd.DataFrame:
    payload = request_openaq(
    "locations",
    params={
        "iso": country,
        "limit": limit,
        "page": 1,
    },
)

    results = payload.get("results", [])

    rows = []

    for item in results:
        name = item.get("name", "")
        locality = item.get("locality", "")
        timezone = item.get("timezone", "")

        text = f"{name} {locality}".lower()

        if city.lower() not in text:
            continue

        coords = item.get("coordinates") or {}

        rows.append(
            {
                "location_id": item.get("id"),
                "name": name,
                "locality": locality,
                "timezone": timezone,
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "instruments": item.get("instruments"),
                "sensors": item.get("sensors"),
            }
        )

    return pd.DataFrame(rows)


def safe_get_name(obj):
    if isinstance(obj, dict):
        return obj.get("name") or obj.get("displayName") or obj.get("id")
    return obj


def safe_get_id(obj):
    if isinstance(obj, dict):
        return obj.get("id")
    return None


def safe_get_value(obj):
    if isinstance(obj, dict):
        return obj.get("value")
    return obj


def safe_get_unit(obj):
    if isinstance(obj, dict):
        return obj.get("unit")
    return None


def safe_get_datetime(obj, key: str):
    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, dict):
            return value.get("utc") or value.get("local")
        return value
    return None


def safe_get_coordinate(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return None

def build_sensor_lookup(locations_df: pd.DataFrame) -> dict:
    sensor_lookup = {}

    if "sensors" not in locations_df.columns:
        return sensor_lookup

    for _, row in locations_df.iterrows():
        sensors = row.get("sensors")

        if not isinstance(sensors, list):
            continue

        for sensor in sensors:
            if not isinstance(sensor, dict):
                continue

            sensor_id = sensor.get("id")
            if sensor_id is None:
                continue

            parameter = sensor.get("parameter") or {}

            sensor_lookup[int(sensor_id)] = {
                "sensor_id": int(sensor_id),
                "sensor_name": sensor.get("name"),
                "parameter": (
                    parameter.get("name")
                    or parameter.get("displayName")
                    or sensor.get("name")
                ),
                "parameter_id": parameter.get("id"),
                "unit": sensor.get("unit") or parameter.get("units"),
            }

    return sensor_lookup


def normalize_parameter_name(name):
    if name is None:
        return None

    text = str(name).lower()

    if "pm25" in text or "pm2.5" in text:
        return "pm25"
    if "pm10" in text:
        return "pm10"
    if "no2" in text:
        return "no2"
    if "so2" in text:
        return "so2"
    if "co" in text:
        return "co"
    if "o3" in text:
        return "o3"
    if "temperature" in text:
        return "temperature"
    if "humidity" in text:
        return "humidity"

    return text.replace(" ", "_")


def fetch_latest_for_location(location_id: int, sensor_lookup: dict) -> pd.DataFrame:
    payload = request_openaq(
        f"locations/{location_id}/latest",
        params={},
    )

    results = payload.get("results", [])

    rows = []

    for item in results:
        sensor_id = item.get("sensorsId")
        sensor_meta = sensor_lookup.get(int(sensor_id), {}) if sensor_id is not None else {}

        datetime_obj = item.get("datetime") or {}
        datetime_utc = datetime_obj.get("utc")
        freshness = compute_freshness_status(datetime_utc)
        coordinates = item.get("coordinates") or {}

        raw_parameter = sensor_meta.get("parameter") or sensor_meta.get("sensor_name")
        parameter = normalize_parameter_name(raw_parameter)

        rows.append(
            {
                "location_id": location_id,
                "sensor_id": sensor_id,
                "sensor_name": sensor_meta.get("sensor_name"),
                "parameter": parameter,
                "raw_parameter": raw_parameter,
                "parameter_id": sensor_meta.get("parameter_id"),
                "value": item.get("value"),
                "unit": sensor_meta.get("unit"),
                "datetime_utc": datetime_obj.get("utc"),
                "datetime_local": datetime_obj.get("local"),
                "latitude": coordinates.get("latitude"),
                "longitude": coordinates.get("longitude"),
                "datetime_utc": datetime_utc,
                "datetime_local": datetime_obj.get("local"),
                "age_hours": freshness["age_hours"],
                "freshness_status": freshness["freshness_status"],
                "usable_for_realtime": freshness["usable_for_realtime"],
            }
        )

    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--city", default="Chennai")
    parser.add_argument("--country", default="IN")
    parser.add_argument("--output-prefix", default="chennai_openaq")

    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    locations_df = search_locations(
        city=args.city,
        country=args.country,
    )

    locations_path = RAW_DATA_DIR / f"{args.output_prefix}_locations.csv"
    locations_df.to_csv(locations_path, index=False)

    print(f"Saved locations to: {locations_path}")
    print(f"Locations found: {len(locations_df)}")
    print(locations_df.head())

    if locations_df.empty:
        print(
            "\nNo OpenAQ locations found for this city. "
            "Try another city or use CPCB manual export."
        )
        return

    sensor_lookup = build_sensor_lookup(locations_df)

    print(f"\nSensors mapped: {len(sensor_lookup)}")

    latest_frames = []

    for location_id in locations_df["location_id"].dropna().astype(int).tolist():
        try:
            latest_df = fetch_latest_for_location(location_id, sensor_lookup=sensor_lookup)
            if not latest_df.empty:
                latest_frames.append(latest_df)
        except Exception as exc:
            print(f"Failed latest fetch for location {location_id}: {exc}")

    if latest_frames:
        latest_all = pd.concat(latest_frames, ignore_index=True)
    else:
        latest_all = pd.DataFrame()

    latest_path = RAW_DATA_DIR / f"{args.output_prefix}_latest_measurements.csv"
    latest_all.to_csv(latest_path, index=False)

    print(f"\nSaved latest measurements to: {latest_path}")
    print(f"Shape: {latest_all.shape}")
    print(latest_all.head())


if __name__ == "__main__":
    main()