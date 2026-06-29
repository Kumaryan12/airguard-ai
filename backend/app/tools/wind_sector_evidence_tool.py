import json
from pathlib import Path
from typing import Any, Dict, Optional

from ml.config import PROJECT_ROOT


REAL_SNAPSHOT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "real_geospatial_snapshot.json"
)


def compass_sector_from_degrees(degrees: Optional[float]) -> Optional[str]:
    if degrees is None:
        return None

    try:
        degrees = float(degrees) % 360
    except (TypeError, ValueError):
        return None

    sectors = [
        "north",
        "north-east",
        "east",
        "south-east",
        "south",
        "south-west",
        "west",
        "north-west",
    ]

    index = round(degrees / 45) % 8
    return sectors[index]


def classify_wind_speed(wind_speed: Optional[float]) -> str:
    if wind_speed is None:
        return "unknown"

    try:
        wind_speed = float(wind_speed)
    except (TypeError, ValueError):
        return "unknown"

    if wind_speed < 1.0:
        return "very_low"
    if wind_speed < 2.0:
        return "low"
    if wind_speed < 4.0:
        return "moderate"
    return "good_dispersion"


def confidence_delta_from_context(
    source: str,
    wind_speed_class: str,
    pm10_pm25_ratio: Optional[float],
    road_density: Optional[float],
    major_road_density: Optional[float],
    industrial_poi_count: Optional[int],
) -> str:
    if source == "road dust / resuspension":
        if (
            pm10_pm25_ratio is not None
            and pm10_pm25_ratio >= 2.5
            and road_density is not None
            and road_density >= 15
            and wind_speed_class in ["very_low", "low"]
        ):
            return "increase"
        return "neutral"

    if source == "traffic corridor exposure":
        if (
            major_road_density is not None
            and major_road_density >= 3
            and wind_speed_class in ["very_low", "low", "moderate"]
        ):
            return "slight_increase"
        return "neutral"

    if source == "industrial influence screening":
        if (
            industrial_poi_count is not None
            and industrial_poi_count >= 10
            and wind_speed_class in ["very_low", "low"]
        ):
            return "slight_increase_requires_directional_geometry"
        return "neutral"

    if source == "meteorological trapping / poor dispersion":
        if wind_speed_class in ["very_low", "low"]:
            return "increase"
        return "neutral"

    return "neutral"


class WindSectorEvidenceTool:
    """
    Wind Sector Evidence Tool.

    Role:
    - Uses wind direction and wind speed to add meteorological context.
    - Strengthens or weakens source hypotheses only at a screening level.
    - Does not claim exact upwind source attribution unless source coordinates are available.
    """

    def __init__(self, snapshot_path: Path = REAL_SNAPSHOT_PATH) -> None:
        self.snapshot_path = snapshot_path

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> Dict[str, Any]:
        snapshot = self._load_json(self.snapshot_path)
        stations = snapshot.get("stations", [])

        outputs = []

        for station in stations:
            weather = station.get("weather", {})
            dispersion = station.get("dispersion", {})
            geospatial = station.get("geospatial_features", {})

            wind_direction = weather.get("wind_direction")
            wind_speed = weather.get("wind_speed")

            wind_from_sector = compass_sector_from_degrees(wind_direction)
            wind_speed_class = classify_wind_speed(wind_speed)

            pm10_pm25_ratio = dispersion.get("pm10_pm25_ratio")
            road_density = geospatial.get("road_density_km_per_km2")
            major_road_density = geospatial.get("major_road_density_km_per_km2")
            industrial_poi_count = geospatial.get("industrial_poi_count")

            source_alignment = [
                {
                    "source": "road dust / resuspension",
                    "alignment": "meteorologically_plausible_screening",
                    "confidence_delta": confidence_delta_from_context(
                        source="road dust / resuspension",
                        wind_speed_class=wind_speed_class,
                        pm10_pm25_ratio=pm10_pm25_ratio,
                        road_density=road_density,
                        major_road_density=major_road_density,
                        industrial_poi_count=industrial_poi_count,
                    ),
                    "reason": [
                        f"Wind speed class is {wind_speed_class}.",
                        f"PM10/PM2.5 ratio is {round(pm10_pm25_ratio, 2) if pm10_pm25_ratio is not None else 'unknown'}.",
                        f"Road density is {road_density} km/km².",
                    ],
                },
                {
                    "source": "traffic corridor exposure",
                    "alignment": "meteorologically_plausible_screening",
                    "confidence_delta": confidence_delta_from_context(
                        source="traffic corridor exposure",
                        wind_speed_class=wind_speed_class,
                        pm10_pm25_ratio=pm10_pm25_ratio,
                        road_density=road_density,
                        major_road_density=major_road_density,
                        industrial_poi_count=industrial_poi_count,
                    ),
                    "reason": [
                        f"Wind is reported from the {wind_from_sector} sector.",
                        f"Major road density is {major_road_density} km/km².",
                        "Exact upwind road-segment geometry is not yet computed.",
                    ],
                },
                {
                    "source": "industrial influence screening",
                    "alignment": "screening_only",
                    "confidence_delta": confidence_delta_from_context(
                        source="industrial influence screening",
                        wind_speed_class=wind_speed_class,
                        pm10_pm25_ratio=pm10_pm25_ratio,
                        road_density=road_density,
                        major_road_density=major_road_density,
                        industrial_poi_count=industrial_poi_count,
                    ),
                    "reason": [
                        f"Industrial POI count is {industrial_poi_count}.",
                        f"Wind is reported from the {wind_from_sector} sector.",
                        "Industrial POI count alone does not prove emissions impact.",
                    ],
                },
                {
                    "source": "meteorological trapping / poor dispersion",
                    "alignment": "dispersion_context",
                    "confidence_delta": confidence_delta_from_context(
                        source="meteorological trapping / poor dispersion",
                        wind_speed_class=wind_speed_class,
                        pm10_pm25_ratio=pm10_pm25_ratio,
                        road_density=road_density,
                        major_road_density=major_road_density,
                        industrial_poi_count=industrial_poi_count,
                    ),
                    "reason": [
                        f"Wind speed is {wind_speed} m/s.",
                        f"Wind speed class is {wind_speed_class}.",
                        "Low wind can allow local pollutant accumulation.",
                    ],
                },
            ]

            outputs.append(
                {
                    "location_id": station.get("location_id"),
                    "latest_datetime_utc": station.get("latest_datetime_utc"),
                    "wind_direction_degrees": wind_direction,
                    "interpreted_wind_from_sector": wind_from_sector,
                    "wind_speed_mps": wind_speed,
                    "wind_speed_class": wind_speed_class,
                    "source_alignment": source_alignment,
                    "does_not_prove": [
                        "causal source attribution",
                        "specific road responsibility",
                        "specific industrial unit responsibility",
                        "exact upwind source contribution without source-coordinate geometry",
                    ],
                    "recommended_next_upgrade": (
                        "Add exact road-segment and industrial-POI bearings from the station, "
                        "then compare source bearings against wind direction sectors."
                    ),
                }
            )

        return {
            "tool_name": "Wind Sector Evidence Tool",
            "tool_type": "meteorology_source_screening_tool",
            "city": snapshot.get("city", "Chennai"),
            "stations": outputs,
            "warning": (
                "This tool provides screening-level wind-sector evidence. "
                "It does not prove source attribution."
            ),
        }


def main() -> None:
    tool = WindSectorEvidenceTool()
    result = tool.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()