import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REAL_BENCHMARK_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "real_forecast_benchmark_metrics.json"
)

REAL_SNAPSHOT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "real_geospatial_snapshot.json"
)


class RealForecastAgent:
    """
    Forecast validation agent.

    Role:
    - Reads real historical forecast benchmark results.
    - Selects the best validated operational forecast method.
    - Separates best overall method from best learned ML model.
    - Produces an honest decision summary for dashboard/decision memo use.
    """

    def __init__(
        self,
        benchmark_path: Path = REAL_BENCHMARK_PATH,
        snapshot_path: Path = REAL_SNAPSHOT_PATH,
    ) -> None:
        self.benchmark_path = benchmark_path
        self.snapshot_path = snapshot_path

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> Dict[str, Any]:
        benchmark = self._load_json(self.benchmark_path)

        snapshot = None
        if self.snapshot_path.exists():
            snapshot = self._load_json(self.snapshot_path)

        results = benchmark["results"]

        best_overall_name = benchmark["best_overall"]
        best_learned_name = benchmark["best_learned_model"]
        best_baseline_name = benchmark["best_baseline"]

        best_overall = results[best_overall_name]
        best_learned = results[best_learned_name]
        persistence = results["persistence_current_aqi"]

        learned_beats_best_baseline = (
            best_learned["rmse"] < results[best_baseline_name]["rmse"]
        )

        if learned_beats_best_baseline:
            deployment_mode = "learned_model_operational"
            selected_method = best_learned_name
            selected_metrics = best_learned
            rationale = (
                "The best learned model outperforms the strongest operational baseline, "
                "so it can be used as the primary real-data forecast method for this station."
            )
        else:
            deployment_mode = "baseline_operational_ml_experimental"
            selected_method = best_overall_name
            selected_metrics = best_overall
            rationale = (
    "The strongest operational baseline outperforms the learned models on the current "
    "one-station real-data validation. AirGuard should use the validated baseline "
    "for operational forecast display and keep the learned model as experimental "
    "until more stations, longer history, and official AQI labels are available."
)

        latest_station_context = None

        if snapshot and snapshot.get("stations"):
            station = snapshot["stations"][0]
            latest_station_context = {
                "location_id": station.get("location_id"),
                "latest_datetime_utc": station.get("latest_datetime_utc"),
                "max_age_hours": station.get("max_age_hours"),
                "estimated_aqi": station.get("aqi_estimate", {}).get("estimated_aqi"),
                "estimated_aqi_category": station.get("aqi_estimate", {}).get(
                    "estimated_aqi_category"
                ),
                "dispersion_risk": station.get("dispersion", {}).get("dispersion_risk"),
                "data_status": station.get("data_status"),
            }

        return {
            "agent_name": "Real Forecast Agent",
            "agent_type": "validation_and_model_selection_agent",
            "city": benchmark["city"],
            "station_location_id": benchmark["station_location_id"],
            "deployment_mode": deployment_mode,
            "selected_forecast_method": selected_method,
            "selected_method_type": selected_metrics["type"],
            "selected_method_metrics": {
                "mae": selected_metrics["mae"],
                "rmse": selected_metrics["rmse"],
                "r2": selected_metrics["r2"],
                "aqi_category_accuracy": selected_metrics["aqi_category_accuracy"],
                "rmse_improvement_vs_persistence": selected_metrics[
                    "rmse_improvement_vs_persistence"
                ],
            },
            "best_learned_model": {
                "name": best_learned_name,
                "metrics": best_learned,
            },
            "best_baseline": {
                "name": best_baseline_name,
                "metrics": results[best_baseline_name],
            },
            "persistence_baseline": persistence,
            "latest_station_context": latest_station_context,
            "rationale": rationale,
            "warnings": [
                benchmark["important_warning"],
                "Do not claim the learned model beats all baselines unless benchmark results show it.",
                "Current AQI target is approximate and must later be replaced with official CPCB breakpoint AQI.",
            ],
        }


def main() -> None:
    agent = RealForecastAgent()
    result = agent.run()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()