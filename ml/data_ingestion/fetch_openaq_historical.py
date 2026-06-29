import argparse
import os
from typing import Optional

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

    if response.status_code == 408:
        raise RuntimeError(
            "OpenAQ 408 Request Timeout.\n"
            f"URL: {response.url}\n"
            "Try reducing date range, reducing limit, or rerunning."
        )

    if response.status_code == 422:
        raise RuntimeError(
            "OpenAQ 422 Unprocessable Entity.\n"
            f"URL: {response.url}\n"
            f"Response: {response.text}"
        )

    if response.status_code >= 500:
        raise RuntimeError(
            f"OpenAQ server error {response.status_code}.\n"
            f"URL: {response.url}\n"
            f"Response: {response.text[:500]}"
        )

    response.raise_for_status()

    payload = response.json()
    if payload is None:
        raise RuntimeError(f"OpenAQ returned empty JSON. URL: {response.url}")

    return payload


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
    if text == "co" or text.startswith("co "):
        return "co"
    if "o3" in text:
        return "o3"
    if "temperature" in text:
        return "temperature"
    if "relativehumidity" in text or "humidity" in text:
        return "humidity"
    if "wind_direction" in text or "wind direction" in text:
        return "wind_direction"
    if "wind_speed" in text or "wind speed" in text:
        return "wind_speed"

    return text.replace(" ", "_")


def get_location_sensors(location_id: int) -> pd.DataFrame:
    payload = request_openaq(
        f"locations/{location_id}/sensors",
        params={
            "limit": 1000,
            "page": 1,
        },
    )

    results = payload.get("results", [])

    rows = []

    for item in results:
        parameter = item.get("parameter") or {}
        raw_name = (
            parameter.get("name")
            or parameter.get("displayName")
            or item.get("name")
        )
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


def parse_measurement_row(item: dict, sensor_id: int, sensor_meta: dict) -> dict:
    period = item.get("period") or {}
    value_obj = item.get("value")

    if isinstance(value_obj, dict):
        value = value_obj.get("value")
        unit = value_obj.get("unit") or sensor_meta.get("unit")
    else:
        value = value_obj
        unit = sensor_meta.get("unit")

    datetime_utc = None

    datetime_from = period.get("datetimeFrom") or {}
    datetime_to = period.get("datetimeTo") or {}

    if isinstance(datetime_from, dict):
        datetime_utc = datetime_from.get("utc") or datetime_from.get("local")

    if datetime_utc is None and isinstance(datetime_to, dict):
        datetime_utc = datetime_to.get("utc") or datetime_to.get("local")

    if datetime_utc is None:
        datetime_obj = item.get("datetime") or {}
        if isinstance(datetime_obj, dict):
            datetime_utc = datetime_obj.get("utc") or datetime_obj.get("local")

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
    max_pages: int = 20,
    limit: int = 1000,
) -> pd.DataFrame:
    rows = []

    datetime_from = f"{date_from}T00:00:00Z"
    datetime_to = f"{date_to}T23:59:59Z"

    for page in range(1, max_pages + 1):
        payload = request_openaq(
            f"sensors/{sensor_id}/measurements/hourly",
            params={
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
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

    return pd.DataFrame(rows)


def apply_local_date_filter(
    history: pd.DataFrame,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    if history.empty:
        return history

    history = history.copy()
    history["datetime_utc"] = pd.to_datetime(
        history["datetime_utc"],
        utc=True,
        errors="coerce",
    )

    requested_start = pd.to_datetime(f"{date_from}T00:00:00Z", utc=True)
    requested_end = pd.to_datetime(f"{date_to}T23:59:59Z", utc=True)

    before_shape = history.shape

    history = history[
        (history["datetime_utc"] >= requested_start)
        & (history["datetime_utc"] <= requested_end)
    ].copy()

    print(f"\nLocal date filter applied: {before_shape} -> {history.shape}")

    history["datetime_utc"] = history["datetime_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    return history


def choose_best_sensor_per_parameter(sensors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Some locations contain old and new sensors for the same pollutant.
    First version rule:
    keep the largest sensor_id per parameter, because newer OpenAQ sensor IDs
    are usually larger and more likely to represent the active feed.
    """
    if sensors_df.empty:
        return sensors_df

    sensors_df = sensors_df.copy()
    sensors_df["sensor_id_numeric"] = pd.to_numeric(
        sensors_df["sensor_id"],
        errors="coerce",
    )

    best = (
        sensors_df.sort_values("sensor_id_numeric")
        .groupby("parameter", as_index=False)
        .tail(1)
        .drop(columns=["sensor_id_numeric"])
        .reset_index(drop=True)
    )

    return best


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--location-id", type=int, default=2586)
    parser.add_argument("--date-from", default="2026-06-01")
    parser.add_argument("--date-to", default="2026-06-29")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", default="chennai_openaq_historical_hourly.csv")

    parser.add_argument(
        "--fresh-sensor-prefix",
        default="122356",
        help=(
            "Only use sensors whose ID starts with this prefix. "
            "Use empty string to disable."
        ),
    )

    parser.add_argument(
        "--one-sensor-per-parameter",
        action="store_true",
        help="Keep only one sensor per parameter after filtering.",
    )

    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sensors_df = get_location_sensors(args.location_id)

    sensors_path = RAW_DATA_DIR / f"openaq_location_{args.location_id}_sensors.csv"
    sensors_df.to_csv(sensors_path, index=False)

    print(f"Saved sensor metadata to: {sensors_path}")
    print("Sensors:")
    print(sensors_df)

    usable_sensors = sensors_df[sensors_df["parameter"].isin(CORE_PARAMETERS)].copy()

    if args.fresh_sensor_prefix:
        usable_sensors = usable_sensors[
            usable_sensors["sensor_id"]
            .astype(str)
            .str.startswith(args.fresh_sensor_prefix)
        ].copy()

    if args.one_sensor_per_parameter:
        usable_sensors = choose_best_sensor_per_parameter(usable_sensors)

    print("\nUsable core sensors after filtering:")
    print(usable_sensors[["sensor_id", "parameter", "raw_parameter", "unit"]])

    frames = []

    for _, sensor_row in usable_sensors.iterrows():
        sensor_id = int(sensor_row["sensor_id"])
        sensor_meta = sensor_row.to_dict()

        print(
            f"\nFetching hourly history for sensor {sensor_id} "
            f"({sensor_meta['parameter']})"
        )

        try:
            sensor_df = fetch_sensor_hourly_measurements(
                sensor_id=sensor_id,
                sensor_meta=sensor_meta,
                date_from=args.date_from,
                date_to=args.date_to,
                max_pages=args.max_pages,
                limit=args.limit,
            )

            print(f"Rows fetched before local filtering: {len(sensor_df)}")

            if not sensor_df.empty:
                frames.append(sensor_df)

        except Exception as exc:
            print(f"Failed sensor {sensor_id}: {exc}")

    if frames:
        history = pd.concat(frames, ignore_index=True)
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

    history = apply_local_date_filter(
        history=history,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    output_path = RAW_DATA_DIR / args.output
    history.to_csv(output_path, index=False)

    print(f"\nSaved historical OpenAQ data to: {output_path}")
    print(f"Shape: {history.shape}")

    if not history.empty:
        print("\nParameter counts:")
        print(history["parameter"].value_counts())

        print("\nDate range:")
        print(history["datetime_utc"].min(), "→", history["datetime_utc"].max())

        print("\nPreview:")
        print(history.head())


if __name__ == "__main__":
    main()