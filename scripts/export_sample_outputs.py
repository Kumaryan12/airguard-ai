import json
import shutil
from pathlib import Path

from ml.config import ARTIFACTS_DIR, PROJECT_ROOT


SAMPLE_OUTPUT_DIR = PROJECT_ROOT / "backend" / "data" / "sample"


FILES_TO_EXPORT = [
    "forecast_output_model.json",
    "event_risk_output.json",
    "attribution_output_model.json",
    "forecast_model_metrics.json",
    "event_risk_metrics.json",
    "source_attribution_metrics.json",
]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_ml_summary() -> dict:
    forecast_metrics_path = ARTIFACTS_DIR / "forecast_model_metrics.json"
    event_metrics_path = ARTIFACTS_DIR / "event_risk_metrics.json"
    attribution_metrics_path = ARTIFACTS_DIR / "source_attribution_metrics.json"

    forecast_metrics = load_json(forecast_metrics_path)
    event_metrics = load_json(event_metrics_path)
    attribution_metrics = load_json(attribution_metrics_path)

    return {
        "project": "AirGuard AI",
        "city": "Chennai",
        "person_1_owner": "AI/ML + Geospatial Intelligence",
        "models": {
            "forecasting": {
                "model_type": "LightGBM regression",
                "targets": ["PM2.5", "PM10", "AQI"],
                "aqi_rmse": forecast_metrics["aqi"]["rmse"],
                "baseline_aqi_rmse": forecast_metrics["aqi"]["baseline_rmse"],
                "aqi_rmse_improvement_vs_persistence": forecast_metrics["aqi"][
                    "rmse_improvement_vs_persistence"
                ],
                "aqi_category_accuracy": forecast_metrics["aqi_category_accuracy"],
            },
            "event_risk": {
                "model_type": "LightGBM binary classifier",
                "event_definition": "AQI >= 201 within 24h",
                "precision": event_metrics["precision"],
                "recall": event_metrics["recall"],
                "f1": event_metrics["f1"],
                "roc_auc": event_metrics["roc_auc"],
                "chosen_threshold": event_metrics["threshold"],
            },
            "source_attribution": {
                "model_type": "LightGBM multiclass classifier",
                "classes": attribution_metrics["class_labels"],
                "accuracy": attribution_metrics["accuracy"],
                "macro_f1": attribution_metrics["macro_f1"],
                "weighted_f1": attribution_metrics["weighted_f1"],
            },
        },
        "technical_notes": [
            "Forecasting model improves AQI RMSE versus persistence baseline.",
            "Event-risk model is used as hotspot trigger because regression alone misses rare spikes.",
            "Source attribution is probabilistic and not direct causal proof.",
            "Meteorology is treated as a confounding/dispersion driver and remains harder to classify.",
        ],
    }


def main() -> None:
    SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = []

    for filename in FILES_TO_EXPORT:
        src = ARTIFACTS_DIR / filename
        if not src.exists():
            missing.append(str(src))
            continue

        dst = SAMPLE_OUTPUT_DIR / filename
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    if missing:
        raise FileNotFoundError(
            "Missing artifact files. Run all ML training scripts first:\n"
            + "\n".join(missing)
        )

    summary = build_ml_summary()
    summary_path = SAMPLE_OUTPUT_DIR / "ml_summary.json"
    save_json(summary, summary_path)
    print(f"Saved ML summary -> {summary_path}")


if __name__ == "__main__":
    main()