import json
from pathlib import Path
from typing import Any, Dict

from ml.aqi.cpcb_aqi import calculate_cpcb_aqi
from ml.config import PROJECT_ROOT


REAL_SNAPSHOT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "real_geospatial_snapshot.json"
)


class CpcbAqiTool:
    """
    CPCB AQI Tool.

    Computes CPCB breakpoint AQI from available pollutant values.
    Missing pollutants are not fabricated.
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

        results = []

        for station in stations:
            pollutants = station.get("pollutants", {})
            aqi_result = calculate_cpcb_aqi(pollutants)

            results.append(
                {
                    "location_id": station.get("location_id"),
                    "latest_datetime_utc": station.get("latest_datetime_utc"),
                    "data_status": station.get("data_status"),
                    "cpcb_aqi": aqi_result,
                    "note": (
                        "CPCB breakpoint AQI computed from available latest pollutant values. "
                        "Missing pollutants are not fabricated. This is not a final regulatory AQI "
                        "unless correct averaging windows are used."
                    ),
                }
            )

        return {
            "tool_name": "CPCB AQI Tool",
            "tool_type": "official_breakpoint_aqi_calculation_tool",
            "city": "Chennai",
            "stations": results,
        }


def main() -> None:
    tool = CpcbAqiTool()
    result = tool.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()