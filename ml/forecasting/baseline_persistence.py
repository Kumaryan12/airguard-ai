import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR, FORECAST_HORIZON_HOURS


RISK_LEVELS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 10_000, "Severe"),
]


def classify_aqi(aqi: float) -> str:
    for low, high, label in RISK_LEVELS:
        if low <= aqi <= high:
            return label
    return "Unknown"


def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "sample_chennai_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python -m ml.data_processing.create_sample_dataset"
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def create_persistence_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Persistence baseline:
    predicted PM2.5 24h ahead = current PM2.5
    predicted PM10 24h ahead = current PM10
    predicted AQI 24h ahead = current AQI
    """
    pred_df = df.copy()

    pred_df["pm25_pred_persistence"] = pred_df["pm25"]
    pred_df["pm10_pred_persistence"] = pred_df["pm10"]
    pred_df["aqi_pred_persistence"] = pred_df["aqi"]

    return pred_df


def evaluate_predictions(pred_df: pd.DataFrame) -> dict:
    metrics = {}

    targets = [
        ("pm25", "pm25_target_24h", "pm25_pred_persistence"),
        ("pm10", "pm10_target_24h", "pm10_pred_persistence"),
        ("aqi", "aqi_target_24h", "aqi_pred_persistence"),
    ]

    for name, target_col, pred_col in targets:
        y_true = pred_df[target_col]
        y_pred = pred_df[pred_col]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        metrics[name] = {
            "mae": float(mae),
            "rmse": float(rmse),
        }

    pred_df["actual_risk_level_24h"] = pred_df["aqi_target_24h"].apply(classify_aqi)
    pred_df["predicted_risk_level"] = pred_df["aqi_pred_persistence"].apply(classify_aqi)

    category_accuracy = (
        pred_df["actual_risk_level_24h"] == pred_df["predicted_risk_level"]
    ).mean()

    high_pollution_actual = pred_df["aqi_target_24h"] >= 201
    high_pollution_pred = pred_df["aqi_pred_persistence"] >= 201

    true_positives = (high_pollution_actual & high_pollution_pred).sum()
    false_negatives = (high_pollution_actual & ~high_pollution_pred).sum()

    recall = true_positives / max(true_positives + false_negatives, 1)

    metrics["aqi_category_accuracy"] = float(category_accuracy)
    metrics["high_pollution_event_recall"] = float(recall)

    return metrics


def build_forecast_output(pred_df: pd.DataFrame) -> dict:
    latest_ts = pred_df["timestamp"].max()
    latest_df = pred_df[pred_df["timestamp"] == latest_ts].copy()

    wards = []

    for _, row in latest_df.iterrows():
        forecast_aqi = float(row["aqi_pred_persistence"])
        current_aqi = float(row["aqi"])

        ward_payload = {
            "ward_id": row["ward_id"],
            "ward_name": row["ward_name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "current_pm25": float(row["pm25"]),
            "forecast_pm25": float(row["pm25_pred_persistence"]),
            "forecast_pm25_p10": float(row["pm25_pred_persistence"] * 0.9),
            "forecast_pm25_p90": float(row["pm25_pred_persistence"] * 1.1),
            "current_aqi": current_aqi,
            "forecast_aqi": forecast_aqi,
            "risk_level": classify_aqi(forecast_aqi),
            "threshold_crossing_probability": float(min(max((forecast_aqi - 100) / 250, 0), 1)),
            "model_confidence": 0.55,
        }

        wards.append(ward_payload)

    return {
        "city": "Chennai",
        "generated_at": datetime.utcnow().isoformat(),
        "model_type": "persistence_baseline",
        "horizon_hours": FORECAST_HORIZON_HOURS,
        "wards": wards,
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    pred_df = create_persistence_predictions(df)
    metrics = evaluate_predictions(pred_df)
    forecast_output = build_forecast_output(pred_df)

    metrics_path = ARTIFACTS_DIR / "baseline_persistence_metrics.json"
    forecast_path = ARTIFACTS_DIR / "forecast_output_baseline.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump(forecast_output, f, indent=2)

    print("Baseline persistence evaluation")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved forecast output to: {forecast_path}")


if __name__ == "__main__":
    main()