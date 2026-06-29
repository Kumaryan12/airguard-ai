import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ml.config import PROJECT_ROOT

def clean_text(value):
    if isinstance(value, str):
        replacements = {
            "Thecurrent": "The current",
            "thecurrent": "the current",
            "currentAQI": "current AQI",
            "AQItarget": "AQI target",
            "CPCBAQI": "CPCB AQI",
            "CPCBbreakpoint": "CPCB breakpoint",
            "Medium-PriorityPreventive": "Medium-Priority Preventive",
            "ActionsRecommended": "Actions Recommended",
            "rollingmean": "rolling mean",
            "meanbaseline": "mean baseline",
            "selectedforecast": "selected forecast",
            "operationalforecast": "operational forecast",
            "neardusty": "near dusty",
            "nearhigh-traffic": "near high-traffic",
            "signaland": "signal and",
            "roaddensity": "road density",
            "asa pollution": "as a pollution",
            "asSatisfactory": "as Satisfactory",
            "columnmeasurement": "column measurement",
            "canaffect": "can affect",
            "source": "source.",
            "fromavailable": "from available",
            "not official CPCBbreakpoint": "not official CPCB breakpoint",
            "not direct ground-level AQI": "not direct ground-level AQI",
        }

        for bad, good in replacements.items():
            value = value.replace(bad, good)

        return " ".join(value.split())

    if isinstance(value, list):
        return [clean_text(item) for item in value]

    if isinstance(value, dict):
        return {key: clean_text(item) for key, item in value.items()}

    return value


DATA_DIR = PROJECT_ROOT / "backend" / "data" / "sample"
OUTPUT_PATH = DATA_DIR / "airguard_demo_output.json"




def load_json(name: str) -> Dict[str, Any]:
    path = DATA_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Missing required demo file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    snapshot = load_json("real_geospatial_snapshot.json")
    forecast_benchmark = load_json("real_forecast_benchmark_metrics.json")
    remote_sensing = load_json("remote_sensing_evidence.json")
    supervisor = load_json("groq_supervisor_agent_output.json")
    citizen = load_json("citizen_advisory_agent_output.json")
    cpcb_aqi = load_json("cpcb_aqi_output.json")

    station = snapshot["stations"][0] if snapshot.get("stations") else {}
    decision = supervisor.get("decision", {})
    advisory = citizen.get("advisory", {})
    cpcb_station = cpcb_aqi["stations"][0] if cpcb_aqi.get("stations") else {}
    cpcb_result = cpcb_station.get("cpcb_aqi", {})

    payload = {
        "project": "AirGuard AI",
        "output_type": "integrated_demo_payload",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "city": "Chennai",
        "station_location_id": station.get("location_id"),
        "executive_summary": {
            "headline": decision.get("decision_headline"),
            "monitoring_priority": decision.get("monitoring_priority"),
            "intervention_required_now": decision.get("intervention_required_now"),
            "current_estimated_aqi": station.get("aqi_estimate", {}).get("estimated_aqi"),
            "current_estimated_aqi_category": station.get("aqi_estimate", {}).get("estimated_aqi_category"),
            "data_status": station.get("data_status"),
            "sensor_age_hours": station.get("max_age_hours"),
            "selected_forecast_method": decision.get("selected_forecast_method"),
            "remote_sensing_signal": remote_sensing.get("relative_no2_signal"),
            "sentinel5p_image_count": remote_sensing.get("collection_image_count"),
            "citizen_advisory_level": advisory.get("advisory_level"),
            "panic_level": advisory.get("panic_level"),
            "cpcb_aqi": cpcb_result.get("aqi"),
            "cpcb_aqi_category": cpcb_result.get("category"),
            "dominant_pollutant": cpcb_result.get("dominant_pollutant"),
        },
        "evidence_stack": {
            "ground_sensor": {
                "latest_datetime_utc": station.get("latest_datetime_utc"),
                "max_age_hours": station.get("max_age_hours"),
                "pollutants": station.get("pollutants"),
                "weather": station.get("weather"),
                "aqi_estimate": station.get("aqi_estimate"),
                "dispersion": station.get("dispersion"),
                "cpcb_aqi_calculation": cpcb_aqi,
            },
            "forecast_validation": {
                "best_overall": forecast_benchmark.get("best_overall"),
                "best_learned_model": forecast_benchmark.get("best_learned_model"),
                "important_warning": forecast_benchmark.get("important_warning"),
                "results": forecast_benchmark.get("results"),
            },
            "geospatial_context": station.get("geospatial_features"),
            "source_hypotheses": station.get("geospatial_hypotheses"),
            "remote_sensing": {
                "satellite_layer": remote_sensing.get("satellite_layer"),
                "dataset_id": remote_sensing.get("dataset_id"),
                "collection_image_count": remote_sensing.get("collection_image_count"),
                "relative_no2_signal": remote_sensing.get("relative_no2_signal"),
                "station_no2_stats": remote_sensing.get("station_no2_stats"),
                "city_no2_stats": remote_sensing.get("city_no2_stats"),
                "interpretation": remote_sensing.get("interpretation"),
                "does_not_prove": remote_sensing.get("does_not_prove"),
                "caveats": remote_sensing.get("caveats"),
            },
        },
        "agent_outputs": {
            "groq_supervisor_decision": decision,
            "citizen_advisory": advisory,
        },
        "recommended_actions": decision.get("recommended_actions", []),
        "safe_claims": decision.get("safe_claims", []),
        "claims_to_avoid": decision.get("claims_to_avoid", []),
        "limitations": decision.get("limitations", []),
    }
    payload = clean_text(payload)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved integrated AirGuard demo output to: {OUTPUT_PATH}")
    print("Headline:", payload["executive_summary"]["headline"])
    print("Monitoring priority:", payload["executive_summary"]["monitoring_priority"])
    print("Current AQI:", payload["executive_summary"]["current_estimated_aqi"])
    print("Remote sensing signal:", payload["executive_summary"]["remote_sensing_signal"])
    print("Citizen advisory level:", payload["executive_summary"]["citizen_advisory_level"])


if __name__ == "__main__":
    main()