import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REAL_SNAPSHOT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "real_geospatial_snapshot.json"
)


class InterventionRankingTool:
    """
    Intervention ranking tool.

    Role:
    - Reads current real geospatial/sensor snapshot.
    - Converts evidence-backed source hypotheses into candidate interventions.
    - This is a deterministic recommendation tool, not an autonomous agent.
    """

    def __init__(self, snapshot_path: Path = REAL_SNAPSHOT_PATH) -> None:
        self.snapshot_path = snapshot_path

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _rank_interventions_for_station(self, station: Dict[str, Any]) -> List[Dict[str, Any]]:
        interventions = []

        aqi = station.get("aqi_estimate", {}).get("estimated_aqi")
        aqi_category = station.get("aqi_estimate", {}).get("estimated_aqi_category")
        dispersion_risk = station.get("dispersion", {}).get("dispersion_risk")
        geo = station.get("geospatial_features", {})
        hypotheses = station.get("geospatial_hypotheses", [])

        hypothesis_sources = {
            item.get("source", "").lower(): item for item in hypotheses
        }

        road_density = geo.get("road_density_km_per_km2")
        major_road_density = geo.get("major_road_density_km_per_km2")
        industrial_count = geo.get("industrial_poi_count")
        vulnerability_count = geo.get("vulnerability_poi_count")

        if any("road dust" in src or "resuspension" in src for src in hypothesis_sources):
            interventions.append(
                {
                    "intervention": "Targeted mechanical road sweeping and water misting",
                    "source_addressed": "road dust / resuspension",
                    "priority": "medium",
                    "why": [
                        f"PM10-heavy signal and road density {road_density} km/km²",
                        "Action is low-regret and operationally deployable.",
                    ],
                    "required_evidence_before_enforcement": [
                        "field inspection photos",
                        "road dust accumulation confirmation",
                        "ward engineer confirmation",
                    ],
                    "intervention_type": "municipal_operations",
                }
            )

        if major_road_density is not None and major_road_density >= 2:
            interventions.append(
                {
                    "intervention": "Traffic flow and idling control near major road corridor",
                    "source_addressed": "traffic emissions / road corridor exposure",
                    "priority": "medium",
                    "why": [
                        f"Major road density {major_road_density} km/km²",
                        "Major road lies close to the monitoring point.",
                    ],
                    "required_evidence_before_enforcement": [
                        "traffic police confirmation",
                        "peak-hour congestion observation",
                        "NO2/CO corroboration if available",
                    ],
                    "intervention_type": "traffic_management",
                }
            )

        if industrial_count is not None and industrial_count > 0:
            interventions.append(
                {
                    "intervention": "Screen nearby industrial units for compliance risk",
                    "source_addressed": "possible industrial influence",
                    "priority": "low-medium",
                    "why": [
                        f"{industrial_count} industrial-tagged OSM features within radius",
                        "POI proximity is not causal proof, so action should begin with screening.",
                    ],
                    "required_evidence_before_enforcement": [
                        "official industry inventory",
                        "stack/emission permit records",
                        "wind-sector analysis",
                        "field inspection",
                    ],
                    "intervention_type": "inspection_screening",
                }
            )

        if dispersion_risk in ["medium", "high"]:
            interventions.append(
                {
                    "intervention": "Schedule dust and traffic interventions during poor-dispersion window",
                    "source_addressed": "meteorological trapping / poor dispersion",
                    "priority": "medium" if dispersion_risk == "medium" else "high",
                    "why": [
                        f"Dispersion risk is {dispersion_risk}",
                        "Low wind can allow pollutants to accumulate locally.",
                    ],
                    "required_evidence_before_enforcement": [
                        "weather forecast confirmation",
                        "wind speed and direction trend",
                    ],
                    "intervention_type": "timing_optimisation",
                }
            )

        if vulnerability_count is not None and vulnerability_count > 0:
            interventions.append(
                {
                    "intervention": "Prepare targeted advisory for nearby sensitive receptors",
                    "source_addressed": "public health exposure",
                    "priority": "low" if aqi_category in ["Good", "Satisfactory"] else "medium",
                    "why": [
                        f"{vulnerability_count} school/hospital POIs within radius",
                        f"Current estimated AQI category is {aqi_category}",
                    ],
                    "required_evidence_before_enforcement": [
                        "school/hospital list validation",
                        "local authority approval for advisory",
                    ],
                    "intervention_type": "public_health_advisory",
                }
            )

        if not interventions:
            interventions.append(
                {
                    "intervention": "Continue monitoring; no intervention recommended from current evidence",
                    "source_addressed": "none",
                    "priority": "low",
                    "why": [
                        "No sufficiently strong evidence pattern detected.",
                    ],
                    "required_evidence_before_enforcement": [],
                    "intervention_type": "monitoring",
                }
            )

        priority_order = {"high": 3, "medium": 2, "low-medium": 1.5, "low": 1}

        interventions = sorted(
            interventions,
            key=lambda x: priority_order.get(x["priority"], 0),
            reverse=True,
        )

        return interventions

    def run(self) -> Dict[str, Any]:
        snapshot = self._load_json(self.snapshot_path)

        stations_output = []

        for station in snapshot.get("stations", []):
            stations_output.append(
                {
                    "location_id": station.get("location_id"),
                    "latest_datetime_utc": station.get("latest_datetime_utc"),
                    "aqi_context": station.get("aqi_estimate"),
                    "data_status": station.get("data_status"),
                    "ranked_interventions": self._rank_interventions_for_station(station),
                }
            )

        return {
            "tool_name": "Intervention Ranking Tool",
            "tool_type": "deterministic_intervention_mapping_tool",
            "city": snapshot.get("city", "Chennai"),
            "stations": stations_output,
            "warning": (
                "Interventions are candidate operational actions. Enforcement requires "
                "human review and additional official evidence."
            ),
        }


def main() -> None:
    tool = InterventionRankingTool()
    result = tool.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()