import json
from datetime import datetime
from pathlib import Path

from ml.config import PROJECT_ROOT


SAMPLE_DIR = PROJECT_ROOT / "backend" / "data" / "sample"

FORECAST_PATH = SAMPLE_DIR / "forecast_output_model.json"
EVENT_RISK_PATH = SAMPLE_DIR / "event_risk_output.json"
ATTRIBUTION_PATH = SAMPLE_DIR / "attribution_output_model.json"
SUMMARY_PATH = SAMPLE_DIR / "ml_summary.json"

OUTPUT_PATH = SAMPLE_DIR / "airguard_intelligence_output.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_risk_priority_score(
    forecast_aqi: float,
    event_probability: float,
    model_confidence: float,
    vulnerability: float = 0.6,
) -> float:
    """
    First version of a priority score.

    This is not the intervention recommendation yet.
    This score helps Person 2 rank hotspots in the agent/dashboard layer.
    """
    aqi_factor = min(max((forecast_aqi - 50) / 300, 0), 1)
    event_factor = min(max(event_probability, 0), 1)
    confidence_factor = min(max(model_confidence, 0), 1)
    vulnerability_factor = min(max(vulnerability, 0), 1)

    score = (
        0.35 * aqi_factor
        + 0.35 * event_factor
        + 0.15 * confidence_factor
        + 0.15 * vulnerability_factor
    )

    return round(float(score), 4)


def get_priority_band(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    if score >= 0.55:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def main() -> None:
    forecast_payload = load_json(FORECAST_PATH)
    event_payload = load_json(EVENT_RISK_PATH)
    attribution_payload = load_json(ATTRIBUTION_PATH)
    summary_payload = load_json(SUMMARY_PATH)

    forecast_by_ward = {
        item["ward_id"]: item for item in forecast_payload["wards"]
    }

    event_by_ward = {
        item["ward_id"]: item for item in event_payload["wards"]
    }

    attribution_by_ward = {
        item["ward_id"]: item for item in attribution_payload["wards"]
    }

    ward_ids = sorted(
        set(forecast_by_ward)
        & set(event_by_ward)
        & set(attribution_by_ward)
    )

    wards = []

    for ward_id in ward_ids:
        forecast = forecast_by_ward[ward_id]
        event = event_by_ward[ward_id]
        attribution = attribution_by_ward[ward_id]

        forecast_aqi = float(forecast["forecast_aqi"])
        event_probability = float(event["high_pollution_probability_24h"])
        model_confidence = float(forecast["model_confidence"])

        # Placeholder vulnerability until real school/hospital/population layer is added.
        # Person 2 can later replace this from vulnerability geospatial data.
        vulnerability = 0.6

        priority_score = get_risk_priority_score(
            forecast_aqi=forecast_aqi,
            event_probability=event_probability,
            model_confidence=model_confidence,
            vulnerability=vulnerability,
        )

        ward_output = {
            "ward_id": ward_id,
            "ward_name": forecast["ward_name"],
            "lat": forecast["lat"],
            "lon": forecast["lon"],

            "forecast": {
                "horizon_hours": forecast_payload["horizon_hours"],
                "current_pm25": forecast["current_pm25"],
                "forecast_pm25": forecast["forecast_pm25"],
                "forecast_pm25_p10": forecast["forecast_pm25_p10"],
                "forecast_pm25_p90": forecast["forecast_pm25_p90"],
                "current_aqi": forecast["current_aqi"],
                "forecast_aqi": forecast["forecast_aqi"],
                "risk_level": forecast["risk_level"],
                "model_confidence": forecast["model_confidence"],
            },

            "event_risk": {
                "event_definition": event_payload["event_definition"],
                "event_threshold_aqi": event["event_threshold_aqi"],
                "high_pollution_probability_24h": event[
                    "high_pollution_probability_24h"
                ],
                "high_pollution_event_predicted": event[
                    "high_pollution_event_predicted"
                ],
            },

            "source_attribution": {
                "dominant_sources": attribution["dominant_sources"],
                "attribution_confidence": attribution["attribution_confidence"],
                "causal_warning": attribution["causal_warning"],
            },

            "priority": {
                "priority_score": priority_score,
                "priority_band": get_priority_band(priority_score),
                "priority_reason": (
                    "Computed from forecast AQI, high-pollution event probability, "
                    "model confidence, and placeholder population vulnerability."
                ),
            },
        }

        wards.append(ward_output)

    wards = sorted(
        wards,
        key=lambda item: item["priority"]["priority_score"],
        reverse=True,
    )

    payload = {
        "project": "AirGuard AI",
        "city": forecast_payload["city"],
        "generated_at": datetime.utcnow().isoformat(),
        "output_type": "combined_ml_intelligence",
        "description": (
            "Integration-ready ML intelligence output combining forecasting, "
            "event-risk detection, and probabilistic source attribution."
        ),
        "models_summary": summary_payload["models"],
        "wards": wards,
    }

    save_json(payload, OUTPUT_PATH)

    print(f"Saved combined intelligence output to: {OUTPUT_PATH}")
    print(f"Wards exported: {len(wards)}")
    print("\nTop priority wards:")
    for ward in wards[:5]:
        print(
            ward["ward_id"],
            ward["ward_name"],
            ward["priority"]["priority_band"],
            ward["priority"]["priority_score"],
            "AQI:",
            round(ward["forecast"]["forecast_aqi"], 2),
            "Event probability:",
            round(ward["event_risk"]["high_pollution_probability_24h"], 3),
        )


if __name__ == "__main__":
    main()