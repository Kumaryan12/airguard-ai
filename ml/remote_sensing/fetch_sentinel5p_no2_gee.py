import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import ee
from dotenv import load_dotenv

from ml.config import PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


DATASET_ID = "COPERNICUS/S5P/OFFL/L3_NO2"
BAND_NAME = "tropospheric_NO2_column_number_density"

RAW_OUTPUT_PATH = RAW_DATA_DIR / "chennai_sentinel5p_no2_stats.json"
PROCESSED_OUTPUT_PATH = PROCESSED_DATA_DIR / "chennai_remote_sensing_evidence.json"


def initialize_earth_engine() -> None:
    project = os.getenv("GEE_PROJECT")

    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def build_geometry(latitude: float, longitude: float, buffer_km: float):
    point = ee.Geometry.Point([longitude, latitude])
    station_buffer = point.buffer(buffer_km * 1000)

    city_bbox = ee.Geometry.Rectangle(
        [
            longitude - 0.35,
            latitude - 0.35,
            longitude + 0.35,
            latitude + 0.35,
        ]
    )

    return point, station_buffer, city_bbox


def reduce_no2_stats(image, geometry, scale: int = 1000) -> Dict[str, Any]:
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.percentile([90, 95]), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
    )

    stats = image.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    )

    return stats.getInfo()


def safe_get(stats: Dict[str, Any], suffix: str):
    key = f"{BAND_NAME}_{suffix}"
    value = stats.get(key)
    if value is None:
        return None
    return float(value)


def classify_relative_signal(station_mean, city_mean, city_p90):
    if station_mean is None or city_mean is None or city_p90 is None:
        return "unknown"

    if station_mean >= city_p90:
        return "high_relative_no2"
    if station_mean >= city_mean:
        return "moderate_relative_no2"
    return "low_relative_no2"


def build_evidence_payload(
    raw_payload: Dict[str, Any],
    station_stats: Dict[str, Any],
    city_stats: Dict[str, Any],
) -> Dict[str, Any]:
    station_mean = safe_get(station_stats, "mean")
    station_median = safe_get(station_stats, "median")
    station_p90 = safe_get(station_stats, "p90")
    station_p95 = safe_get(station_stats, "p95")

    city_mean = safe_get(city_stats, "mean")
    city_median = safe_get(city_stats, "median")
    city_p90 = safe_get(city_stats, "p90")
    city_p95 = safe_get(city_stats, "p95")

    relative_signal = classify_relative_signal(
        station_mean=station_mean,
        city_mean=city_mean,
        city_p90=city_p90,
    )

    supports = []
    caveats = [
        "Sentinel-5P NO2 is a column measurement, not direct ground-level AQI.",
        "Satellite evidence supports regional combustion-pattern context, not causal source proof.",
        "Clouds, retrieval quality, and satellite overpass timing can affect coverage.",
    ]

    if relative_signal in ["moderate_relative_no2", "high_relative_no2"]:
        supports.append("regional combustion-related pollution context")
        supports.append("traffic corridor exposure hypothesis")
        supports.append("industrial influence screening, not confirmation")

    if relative_signal == "low_relative_no2":
        supports.append("no strong regional NO2 enhancement detected in this window")

    return {
        "project": "AirGuard AI",
        "city": raw_payload["city"],
        "output_type": "remote_sensing_evidence",
        "satellite_layer": "Sentinel-5P OFFL NO2",
        "dataset_id": DATASET_ID,
        "band": BAND_NAME,
        "date_from": raw_payload["date_from"],
        "date_to": raw_payload["date_to"],
        "location": raw_payload["location"],
        "collection_image_count": raw_payload["collection_image_count"],
        "station_buffer_km": raw_payload["station_buffer_km"],
        "station_no2_stats": {
            "mean": station_mean,
            "median": station_median,
            "p90": station_p90,
            "p95": station_p95,
        },
        "city_no2_stats": {
            "mean": city_mean,
            "median": city_median,
            "p90": city_p90,
            "p95": city_p95,
        },
        "relative_no2_signal": relative_signal,
        "evidence_role": "regional combustion pollution context",
        "supports": supports,
        "does_not_prove": [
            "exact ground-level AQI",
            "confirmed source attribution",
            "specific factory or road segment responsibility",
        ],
        "caveats": caveats,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--city", default="Chennai")
    parser.add_argument("--latitude", type=float, default=13.164544)
    parser.add_argument("--longitude", type=float, default=80.26285)
    parser.add_argument("--date-from", default="2026-06-01")
    parser.add_argument("--date-to", default="2026-06-29")
    parser.add_argument("--station-buffer-km", type=float, default=10.0)

    args = parser.parse_args()

    initialize_earth_engine()

    _, station_buffer, city_bbox = build_geometry(
        latitude=args.latitude,
        longitude=args.longitude,
        buffer_km=args.station_buffer_km,
    )

    collection = (
        ee.ImageCollection(DATASET_ID)
        .filterDate(args.date_from, args.date_to)
        .filterBounds(city_bbox)
        .select(BAND_NAME)
    )

    image_count = int(collection.size().getInfo())

    if image_count == 0:
        raise RuntimeError(
            f"No Sentinel-5P NO2 images found for {args.date_from} to {args.date_to}."
        )

    no2_image = collection.median()

    station_stats = reduce_no2_stats(no2_image, station_buffer)
    city_stats = reduce_no2_stats(no2_image, city_bbox)

    raw_payload = {
        "project": "AirGuard AI",
        "city": args.city,
        "dataset_id": DATASET_ID,
        "band": BAND_NAME,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "location": {
            "latitude": args.latitude,
            "longitude": args.longitude,
        },
        "station_buffer_km": args.station_buffer_km,
        "collection_image_count": image_count,
        "station_stats_raw": station_stats,
        "city_stats_raw": city_stats,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    evidence_payload = build_evidence_payload(
        raw_payload=raw_payload,
        station_stats=station_stats,
        city_stats=city_stats,
    )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, indent=2)

    with open(PROCESSED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(evidence_payload, f, indent=2)

    print(f"Saved raw Sentinel-5P NO2 stats to: {RAW_OUTPUT_PATH}")
    print(f"Saved remote sensing evidence to: {PROCESSED_OUTPUT_PATH}")
    print(json.dumps(evidence_payload, indent=2))


if __name__ == "__main__":
    main()