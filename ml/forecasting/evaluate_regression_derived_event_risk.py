import json
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.config import ARTIFACTS_DIR, PROJECT_ROOT


PREDICTIONS_PATH = ARTIFACTS_DIR / "real_cpcb_window_forecast_test_predictions.csv"
METRICS_PATH = ARTIFACTS_DIR / "regression_derived_event_risk_metrics.json"
BACKEND_OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "sample"
    / "regression_derived_event_risk_metrics.json"
)

TARGET_AQI_COL = "cpcb_window_aqi_target_24h"
CURRENT_AQI_COL = "cpcb_window_aqi"

PREVENTIVE_AQI_THRESHOLD = 76.0
PREVENTIVE_DELTA_THRESHOLD = 15.0


def get_preventive_risk(
    predicted_aqi: np.ndarray,
    current_aqi: np.ndarray,
) -> np.ndarray:
    return (
        (predicted_aqi >= PREVENTIVE_AQI_THRESHOLD)
        | ((predicted_aqi - current_aqi) >= PREVENTIVE_DELTA_THRESHOLD)
    ).astype(int)


def get_true_preventive_risk(df: pd.DataFrame) -> np.ndarray:
    return get_preventive_risk(
        predicted_aqi=df[TARGET_AQI_COL].values,
        current_aqi=df[CURRENT_AQI_COL].values,
    )


def safe_auc(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    return float(average_precision_score(y_true, y_score))


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_score),
        "average_precision": safe_average_precision(y_true, y_score),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "positive_rate_predicted": float(np.mean(y_pred)),
    }


def normalise_score(predicted_aqi: np.ndarray) -> np.ndarray:
    return np.clip(predicted_aqi / 150.0, 0, 1)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}. "
            "Run: python -m ml.forecasting.train_real_cpcb_window_forecast_benchmark"
        )

    df = pd.read_csv(PREDICTIONS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    required_cols = [
        CURRENT_AQI_COL,
        TARGET_AQI_COL,
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required_cols).copy()

    y_true = get_true_preventive_risk(df)

    results = {}

    current_aqi = df[CURRENT_AQI_COL].values

    if CURRENT_AQI_COL in df.columns:
        persistence_predicted_aqi = df[CURRENT_AQI_COL].values
        y_pred = get_preventive_risk(persistence_predicted_aqi, current_aqi)
        y_score = normalise_score(persistence_predicted_aqi)

        results["persistence_current_aqi_risk"] = {
            **evaluate_binary_predictions(y_true, y_pred, y_score),
            "type": "baseline",
            "source_forecast": CURRENT_AQI_COL,
        }

    if "rolling_mean_24h_prediction" in df.columns:
        rolling_predicted_aqi = df["rolling_mean_24h_prediction"].values
        y_pred = get_preventive_risk(rolling_predicted_aqi, current_aqi)
        y_score = normalise_score(rolling_predicted_aqi)

        results["rolling_mean_24h_regression_risk"] = {
            **evaluate_binary_predictions(y_true, y_pred, y_score),
            "type": "baseline",
            "source_forecast": "rolling_mean_24h_prediction",
        }

    learned_prediction_cols = [
        col
        for col in df.columns
        if col.endswith("_prediction")
        and col not in {"rolling_mean_24h_prediction"}
    ]

    for col in learned_prediction_cols:
        predicted_aqi = df[col].values
        y_pred = get_preventive_risk(predicted_aqi, current_aqi)
        y_score = normalise_score(predicted_aqi)

        results[f"{col}_derived_risk"] = {
            **evaluate_binary_predictions(y_true, y_pred, y_score),
            "type": "learned_model_derived_risk",
            "source_forecast": col,
        }

    if not results:
        raise ValueError("No forecast columns found for risk evaluation.")

    best_overall = max(results, key=lambda name: results[name]["f1"])

    baseline_names = [
        name for name, metrics in results.items()
        if metrics.get("type") == "baseline"
    ]

    learned_names = [
        name for name, metrics in results.items()
        if metrics.get("type") == "learned_model_derived_risk"
    ]

    best_baseline = (
        max(baseline_names, key=lambda name: results[name]["f1"])
        if baseline_names
        else None
    )

    best_learned = (
        max(learned_names, key=lambda name: results[name]["f1"])
        if learned_names
        else None
    )

    output = {
        "city": "Chennai",
        "station_location_id": int(df["location_id"].iloc[0])
        if "location_id" in df.columns
        else None,
        "model_type": "regression_derived_preventive_event_risk",
        "risk_definition": (
            f"predicted_aqi_24h >= {PREVENTIVE_AQI_THRESHOLD} "
            f"OR predicted_aqi_24h - current_aqi >= {PREVENTIVE_DELTA_THRESHOLD}"
        ),
        "true_event_rate": float(np.mean(y_true)),
        "test_rows": int(len(df)),
        "date_range": {
            "test_start": str(df["timestamp"].min()),
            "test_end": str(df["timestamp"].max()),
        },
        "results": results,
        "best_baseline": best_baseline,
        "best_learned": best_learned,
        "best_overall": best_overall,
        "selection_metric": "f1",
        "important_warning": (
            "Preventive event risk is derived from 24h AQI forecasts using transparent "
            "thresholds. This is designed for early-warning screening, not automatic enforcement."
        ),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    BACKEND_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BACKEND_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nRegression-derived preventive event risk")
    print(json.dumps(output, indent=2))

    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved backend metrics to: {BACKEND_OUTPUT_PATH}")


if __name__ == "__main__":
    main()