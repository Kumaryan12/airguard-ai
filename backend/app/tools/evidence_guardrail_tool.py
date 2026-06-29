import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REAL_SNAPSHOT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "real_geospatial_snapshot.json"
)

REAL_FORECAST_AGENT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "real_forecast_agent_output.json"
)


class EvidenceGuardrailTool:
    """
Evidence guardrail tool.

Role:
- Reads real sensor/geospatial evidence.
- Reads forecast validation output.
- Checks which claims are supported.
- Flags weak claims and overclaims.
- This is a deterministic validation tool, not an autonomous agent.
"""

    def __init__(
        self,
        snapshot_path: Path = REAL_SNAPSHOT_PATH,
        forecast_agent_path: Path = REAL_FORECAST_AGENT_PATH,
    ) -> None:
        self.snapshot_path = snapshot_path
        self.forecast_agent_path = forecast_agent_path

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def verify_forecast_claims(self, forecast_agent: Dict[str, Any]) -> List[Dict[str, Any]]:
        claims = []

        selected_method = forecast_agent["selected_forecast_method"]
        selected_metrics = forecast_agent["selected_method_metrics"]
        best_learned = forecast_agent["best_learned_model"]

        claims.append(
            {
                "claim": "AirGuard has a real-data forecast validation pipeline.",
                "status": "supported",
                "evidence": [
                    "Real forecast benchmark metrics exist.",
                    f"Selected forecast method: {selected_method}",
                    f"Test RMSE: {selected_metrics['rmse']:.2f}",
                    f"RMSE improvement vs persistence: {selected_metrics['rmse_improvement_vs_persistence'] * 100:.2f}%",
                ],
            }
        )

        claims.append(
            {
                "claim": "The learned ML model is currently the best operational forecast method.",
                "status": "weak_or_not_supported",
                "evidence": [
                    f"Best learned model: {best_learned['name']}",
                    f"Selected operational method: {selected_method}",
                    "The selected method is a baseline, not the learned model.",
                ],
                "safe_rewording": (
                    "The best learned model is competitive, but the current validated operational "
                    "method is the 24-hour rolling mean baseline until more data is available."
                ),
            }
        )

        claims.append(
            {
                "claim": "The operational forecast method improves over persistence.",
                "status": "supported",
                "evidence": [
                    f"RMSE improvement vs persistence: {selected_metrics['rmse_improvement_vs_persistence'] * 100:.2f}%",
                    f"Persistence RMSE: {forecast_agent['persistence_baseline']['rmse']:.2f}",
                    f"Selected method RMSE: {selected_metrics['rmse']:.2f}",
                ],
            }
        )

        return claims

    def verify_source_claims(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        claims = []

        if not snapshot.get("stations"):
            return [
                {
                    "claim": "Real geospatial source evidence is available.",
                    "status": "weak_or_not_supported",
                    "evidence": ["No station snapshot found."],
                }
            ]

        station = snapshot["stations"][0]
        pollutants = station.get("pollutants", {})
        dispersion = station.get("dispersion", {})
        geo = station.get("geospatial_features", {})
        hypotheses = station.get("geospatial_hypotheses", [])

        claims.append(
            {
                "claim": "Current station reading is fresh enough for real-time context.",
                "status": "supported"
                if station.get("data_status") == "fresh_realtime_sensor_snapshot"
                else "partially_supported",
                "evidence": [
                    f"Data status: {station.get('data_status')}",
                    f"Sensor age: {station.get('max_age_hours')} hours",
                    f"Latest timestamp: {station.get('latest_datetime_utc')}",
                ],
            }
        )

        pm_ratio = dispersion.get("pm10_pm25_ratio")
        road_density = geo.get("road_density_km_per_km2")
        nearest_major = geo.get("nearest_major_road_m")

        road_dust_status = "partially_supported"
        road_dust_evidence = []

        if pm_ratio is not None:
            road_dust_evidence.append(f"PM10/PM2.5 ratio: {pm_ratio:.2f}")
        if road_density is not None:
            road_dust_evidence.append(f"Road density: {road_density:.2f} km/km²")
        if nearest_major is not None:
            road_dust_evidence.append(f"Nearest major road: {nearest_major:.2f} m")

        if pm_ratio and pm_ratio >= 2.5 and road_density and road_density >= 5:
            road_dust_status = "supported"

        claims.append(
            {
                "claim": "Road dust / resuspension is a plausible source hypothesis.",
                "status": road_dust_status,
                "evidence": road_dust_evidence,
                "safe_rewording": (
                    "The evidence supports road dust/resuspension as a plausible hypothesis, "
                    "not a confirmed causal source."
                ),
            }
        )

        industrial_count = geo.get("industrial_poi_count")

        claims.append(
            {
                "claim": "Industrial influence is confirmed as a pollution source.",
                "status": "do_not_claim",
                "evidence": [
                    f"Industrial OSM POIs within radius: {industrial_count}",
                    "POI proximity alone does not prove emissions impact.",
                ],
                "safe_rewording": (
                    "Industrial influence is a low-to-medium confidence hypothesis because "
                    "industrial-tagged OSM features exist nearby. Confirmation requires emissions "
                    "inventory, wind-sector analysis, or source apportionment."
                ),
            }
        )

        claims.append(
            {
                "claim": "The system has real geospatial evidence for source hypotheses.",
                "status": "supported",
                "evidence": [
                    f"Road density: {geo.get('road_density_km_per_km2')} km/km²",
                    f"Major road density: {geo.get('major_road_density_km_per_km2')} km/km²",
                    f"Industrial POIs: {geo.get('industrial_poi_count')}",
                    f"Vulnerability POIs: {geo.get('vulnerability_poi_count')}",
                    f"Hypotheses generated: {len(hypotheses)}",
                ],
            }
        )

        claims.append(
            {
                "claim": "The source attribution is causal proof.",
                "status": "do_not_claim",
                "evidence": [
                    "Current evidence uses pollutant ratios, OSM proximity, and dispersion proxies.",
                    "No receptor model, emissions inventory, or chemical source apportionment is used yet.",
                ],
                "safe_rewording": (
                    "The current system generates evidence-backed source hypotheses, not causal proof."
                ),
            }
        )

        return claims

    def run(self) -> Dict[str, Any]:
        snapshot = self._load_json(self.snapshot_path)
        forecast_agent = self._load_json(self.forecast_agent_path)

        forecast_claims = self.verify_forecast_claims(forecast_agent)
        source_claims = self.verify_source_claims(snapshot)

        all_claims = forecast_claims + source_claims

        return {
            "tool_name": "Evidence Guardrail Tool",
            "agent_type": "claim_verification_and_overclaim_prevention_agent",
            "city": "Chennai",
            "station_location_id": forecast_agent["station_location_id"],
            "claims_checked": len(all_claims),
            "claims": all_claims,
            "summary": {
                "supported": sum(c["status"] == "supported" for c in all_claims),
                "partially_supported": sum(c["status"] == "partially_supported" for c in all_claims),
                "weak_or_not_supported": sum(c["status"] == "weak_or_not_supported" for c in all_claims),
                "do_not_claim": sum(c["status"] == "do_not_claim" for c in all_claims),
            },
            "recommended_demo_framing": (
                "AirGuard AI has validated real-data ingestion, forecasting benchmarks, "
                "sensor freshness checks, and geospatial evidence generation. Source outputs "
                "should be presented as evidence-backed hypotheses, not causal proof."
            ),
        }


def main() -> None:
    tool = EvidenceGuardrailTool()
    result = tool.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()