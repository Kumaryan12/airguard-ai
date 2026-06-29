import argparse
import os
import time
from typing import Dict, List

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


PREFERRED_UNIT_PRIORITY = {
    "pm25": ["µg/m³", "ug/m3"],
    "pm10": ["µg/m³", "ug/m3"],
    "no2": ["µg/m³", "ug/m3", "ppb"],
    "so2": ["µg/m³", "ug/m3", "ppb"],
    "o3": ["µg/m³", "ug/m3", "ppb"],
    "co": ["µg/m³", "ug/m3", "ppm", "ppb"],
    "temperature": ["c"],
    "humidity": ["%"],
    "wind_speed": ["m/s"],
    "wind_direction": ["deg"],
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


def unit_rank(parameter: str, unit: str | None) -> int:
    if unit is None:
        return 999

    unit_text = str(unit).strip().lower()
    preferred_units = PREFERRED_UNIT_PRIORITY.get(parameter, [])

    for index, preferred in enumerate(preferred_units):
        if unit_text == preferred.lower():
            return index

    return 999


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


def choose_best_sensors(sensors_df: pd.DataFrame) -> pd.DataFrame:
    if sensors_df.empty:
        return sensors_df

    usable = sensors_df[sensors_df["parameter"].isin(CORE_PARAMETERS)].copy()

    if usable.empty:
        return usable

    usable["unit_rank"] = usable.apply(
        lambda row: unit_rank(row["parameter"], row.get("unit")),
        axis=1,
    )

    usable = usable.sort_values(
        ["parameter", "unit_rank", "sensor_id"],
        ascending=[True, True, True],
    )

    best = usable.drop_duplicates(subset=["location_id", "parameter"], keep="first")

    return best.drop(columns=["unit_rank"], errors="ignore").reset_index(drop=True)


def parse_measurement_row(item: dict, sensor_id: int, sensor_meta: dict) -> dict:
    period = item.get("period") or {}
    value_obj = item.get("value")

    if isinstance(value_obj, dict):
        value = value_obj.get("value")
        unit = value_obj.get("unit") or sensor_meta.get("unit")
    else:
        value = value_obj
        unit = sensor_meta.get("unit")

    datetime_from = period.get("datetimeFrom") or {}
    datetime_to = period.get("datetimeTo") or {}

    datetime_utc = (
        datetime_from.get("utc")
        if isinstance(datetime_from, dict)
        else None
    )

    if datetime_utc is None:
        datetime_utc = (
            datetime_to.get("utc")
            if isinstance(datetime_to, dict)
            else None
        )

    if datetime_utc is None:
        datetime_obj = item.get("datetime") or {}
        datetime_utc = datetime_obj.get("utc") if isinstance(datetime_obj, dict) else None

    return {
        "location_id": sensor_meta["location_id"],
        "sensor_id": sensor_id,
        "parameter": sensor_meta["parameter"],
        "raw_parameter": sensor_meta["raw_parameter"],
        "value": value,
        "unit": unit,
        "datetime_utc": datetime_utc,
    }


def fetch_sensor_hourly_measurements(
    sensor_id: int,
    sensor_meta: dict,
    date_from: str,
    date_to: str,
    max_pages: int,
    limit: int = 1000,
) -> pd.DataFrame:
    rows = []

    for page in range(1, max_pages + 1):
        payload = request_openaq(
            f"sensors/{sensor_id}/measurements/hourly",
            params={
                "date_from": date_from,
                "date_to": date_to,
                "limit": limit,
                "page": page,
            },
        )

        results = payload.get("results", [])

        if not results:
            break

        for item in results:
            rows.append(
                parse_measurement_row(
                    item=item,
                    sensor_id=sensor_id,
                    sensor_meta=sensor_meta,
                )
            )

        if len(results) < limit:
            break

        time.sleep(0.2)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--location-ids",
        default="2586,5655,10780,11578,11579,12046,11581",
        help="Comma-separated OpenAQ location IDs.",
    )
    parser.add_argument("--date-from", default="2026-04-01")
    parser.add_argument("--date-to", default="2026-06-29")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument(
        "--output",
        default="chennai_multi_node_openaq_historical_hourly.csv",
    )
    parser.add_argument(
        "--sensors-output",
        default="chennai_multi_node_openaq_sensors.csv",
    )

    args = parser.parse_args()

    location_ids = [
        int(item.strip())
        for item in args.location_ids.split(",")
        if item.strip()
    ]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_sensor_frames = []
    all_history_frames = []

    for location_id in location_ids:
        print(f"\n=== Location {location_id} ===")

        try:
            sensors_df = get_location_sensors(location_id)
        except Exception as exc:
            print(f"Failed to fetch sensors for location {location_id}: {exc}")
            continue

        if sensors_df.empty:
            print("No sensors found.")
            continue

        all_sensor_frames.append(sensors_df)

        best_sensors = choose_best_sensors(sensors_df)

        print("Selected sensors:")
        if best_sensors.empty:
            print("No usable core sensors.")
            continue

        print(best_sensors[["sensor_id", "parameter", "raw_parameter", "unit"]])

        for _, sensor_row in best_sensors.iterrows():
            sensor_id = int(sensor_row["sensor_id"])
            sensor_meta = sensor_row.to_dict()

            print(
                f"Fetching {sensor_meta['parameter']} "
                f"sensor {sensor_id} for location {location_id}"
            )

            try:
                sensor_df = fetch_sensor_hourly_measurements(
                    sensor_id=sensor_id,
                    sensor_meta=sensor_meta,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    max_pages=args.max_pages,
                )

                print(f"Rows fetched: {len(sensor_df)}")

                if not sensor_df.empty:
                    all_history_frames.append(sensor_df)

            except Exception as exc:
                print(f"Failed sensor {sensor_id}: {exc}")

            time.sleep(0.3)

    if all_sensor_frames:
        sensors_all = pd.concat(all_sensor_frames, ignore_index=True)
    else:
        sensors_all = pd.DataFrame()

    sensors_output_path = RAW_DATA_DIR / args.sensors_output
    sensors_all.to_csv(sensors_output_path, index=False)

    if all_history_frames:
        history = pd.concat(all_history_frames, ignore_index=True)
    else:
        history = pd.DataFrame(
            columns=[
                "location_id",
                "sensor_id",
                "parameter",
                "raw_parameter",
                "value",
                "unit",
                "datetime_utc",
            ]
        )

    if not history.empty:
        history["datetime_utc"] = pd.to_datetime(
            history["datetime_utc"],
            utc=True,
            errors="coerce",
        )

        date_from = pd.Timestamp(args.date_from, tz="UTC")
        date_to = pd.Timestamp(args.date_to, tz="UTC") + pd.Timedelta(days=1)

        before = history.shape
        history = history[
            (history["datetime_utc"] >= date_from)
            & (history["datetime_utc"] < date_to)
        ].copy()
        after = history.shape

        print(f"\nLocal date filter applied: {before} -> {after}")

        history["datetime_utc"] = history["datetime_utc"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    output_path = RAW_DATA_DIR / args.output
    history.to_csv(output_path, index=False)

    print(f"\nSaved sensors to: {sensors_output_path}")
    print(f"Saved multi-node historical data to: {output_path}")
    print(f"Shape: {history.shape}")

    if not history.empty:
        print("\nRows by location:")
        print(history["location_id"].value_counts())

        print("\nParameter counts:")
        print(history["parameter"].value_counts())

        print("\nDate range:")
        print(history["datetime_utc"].min(), "→", history["datetime_utc"].max())

        print("\nPreview:")
        print(history.head())


if __name__ == "__main__":
    main()