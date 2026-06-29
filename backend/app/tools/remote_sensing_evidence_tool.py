import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REMOTE_SENSING_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "remote_sensing_evidence.json"
)


class RemoteSensingEvidenceTool:
    """
    Remote sensing evidence tool.

    Role:
    - Reads Sentinel-5P NO2 evidence.
    - Summarizes regional combustion-related satellite context.
    - Does not claim causal source attribution.
    """

    def __init__(self, evidence_path: Path = REMOTE_SENSING_PATH) -> None:
        self.evidence_path = evidence_path

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {
                "available": False,
                "reason": f"Missing file: {path}",
            }

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["available"] = True
        return data

    def run(self) -> Dict[str, Any]:
        evidence = self._load_json(self.evidence_path)

        if not evidence.get("available"):
            return {
                "tool_name": "Remote Sensing Evidence Tool",
                "tool_type": "satellite_context_tool",
                "available": False,
                "summary": "Remote sensing evidence is not available yet.",
                "reason": evidence.get("reason"),
            }

        relative_signal = evidence.get("relative_no2_signal")

        if relative_signal == "high_relative_no2":
            interpretation = (
                "Satellite NO2 shows a high relative combustion-related signal near the station buffer."
            )
        elif relative_signal == "moderate_relative_no2":
            interpretation = (
                "Satellite NO2 shows a moderate relative combustion-related signal near the station buffer."
            )
        elif relative_signal == "low_relative_no2":
            interpretation = (
                "Satellite NO2 does not show a strong regional enhancement near the station buffer."
            )
        else:
            interpretation = "Satellite NO2 signal is unavailable or inconclusive."

        return {
            "tool_name": "Remote Sensing Evidence Tool",
            "tool_type": "satellite_context_tool",
            "available": True,
            "satellite_layer": evidence.get("satellite_layer"),
            "dataset_id": evidence.get("dataset_id"),
            "date_from": evidence.get("date_from"),
            "date_to": evidence.get("date_to"),
            "collection_image_count": evidence.get("collection_image_count"),
            "relative_no2_signal": relative_signal,
            "station_no2_stats": evidence.get("station_no2_stats"),
            "city_no2_stats": evidence.get("city_no2_stats"),
            "evidence_role": evidence.get("evidence_role"),
            "supports": evidence.get("supports"),
            "does_not_prove": evidence.get("does_not_prove"),
            "interpretation": interpretation,
            "caveats": evidence.get("caveats"),
        }


def main() -> None:
    tool = RemoteSensingEvidenceTool()
    result = tool.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()