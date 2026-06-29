import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.config import (
    ARTIFACTS_DIR,
    FORECAST_HORIZON_HOURS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


RISK_LEVELS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 10_000, "Severe"),
]


FEATURE_COLUMNS = [
    "ward_id",
    "hour",
    "dayofweek",
    "rush_hour",
    "night_stagnation",
    "weekend",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "road_density",
    "construction_score",
    "industrial_score",
    "green_cover",
    "traffic_proxy",
    "pm25",
    "pm10",
    "no2",
    "aqi",
]


TARGET_COLUMNS = {
    "pm25": "pm25_target_24h",
    "pm10": "pm10_target_24h",
    "aqi": "aqi_target_24h",
}


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
    return df.sort_values(["timestamp", "ward_id"]).reset_index(drop=True)


def time_based_split(df: pd.DataFrame, train_fraction: float = 0.8):
    unique_times = np.array(sorted(df["timestamp"].unique()))
    split_idx = int(len(unique_times) * train_fraction)

    train_times = set(unique_times[:split_idx])
    test_times = set(unique_times[split_idx:])

    train_df = df[df["timestamp"].isin(train_times)].copy()
    test_df = df[df["timestamp"].isin(test_times)].copy()

    return train_df, test_df


def build_model() -> Pipeline:
    categorical_features = ["ward_id"]
    numeric_features = [col for col in FEATURE_COLUMNS if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.035,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        objective="regression",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_single_target(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def train_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    models = {}
    predictions = test_df.copy()
    metrics = {}

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    for target_name, target_col in TARGET_COLUMNS.items():
        print(f"\nTraining model for: {target_name}")

        y_train = train_df[target_col]
        y_test = test_df[target_col]

        model = build_model()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        models[target_name] = model
        predictions[f"{target_name}_pred_model"] = pred

        metrics[target_name] = evaluate_single_target(y_test, pred)

    predictions["actual_risk_level_24h"] = predictions["aqi_target_24h"].apply(classify_aqi)
    predictions["predicted_risk_level"] = predictions["aqi_pred_model"].apply(classify_aqi)

    category_accuracy = (
        predictions["actual_risk_level_24h"] == predictions["predicted_risk_level"]
    ).mean()

    high_pollution_actual = predictions["aqi_target_24h"] >= 201
    high_pollution_pred = predictions["aqi_pred_model"] >= 201

    true_positives = (high_pollution_actual & high_pollution_pred).sum()
    false_negatives = (high_pollution_actual & ~high_pollution_pred).sum()

    recall = true_positives / max(true_positives + false_negatives, 1)

    metrics["aqi_category_accuracy"] = float(category_accuracy)
    metrics["high_pollution_event_recall"] = float(recall)

    return models, predictions, metrics


def load_baseline_metrics() -> dict:
    path = ARTIFACTS_DIR / "baseline_persistence_metrics.json"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_baseline_comparison(model_metrics: dict, baseline_metrics: dict) -> dict:
    if not baseline_metrics:
        return model_metrics

    for target in ["pm25", "pm10", "aqi"]:
        if target in model_metrics and target in baseline_metrics:
            baseline_rmse = baseline_metrics[target]["rmse"]
            model_rmse = model_metrics[target]["rmse"]

            improvement = (baseline_rmse - model_rmse) / baseline_rmse

            model_metrics[target]["baseline_rmse"] = float(baseline_rmse)
            model_metrics[target]["rmse_improvement_vs_persistence"] = float(improvement)

    if "high_pollution_event_recall" in baseline_metrics:
        model_metrics["baseline_high_pollution_event_recall"] = baseline_metrics[
            "high_pollution_event_recall"
        ]

    return model_metrics


def estimate_uncertainty(predictions: pd.DataFrame, target_name: str) -> pd.Series:
    """
    Simple residual-based uncertainty estimate for first version.
    Later we can replace this with quantile models or conformal prediction.
    """
    residuals = (
        predictions[f"{target_name}_target_24h"]
        - predictions[f"{target_name}_pred_model"]
    ).abs()

    residual_scale = residuals.groupby(predictions["ward_id"]).transform("median")
    residual_scale = residual_scale.fillna(residuals.median())

    return residual_scale


def build_forecast_output(predictions: pd.DataFrame, metrics: dict) -> dict:
    latest_ts = predictions["timestamp"].max()
    latest_df = predictions[predictions["timestamp"] == latest_ts].copy()

    latest_df["pm25_uncertainty"] = estimate_uncertainty(predictions, "pm25")
    latest_df["aqi_uncertainty"] = estimate_uncertainty(predictions, "aqi")

    wards = []

    for _, row in latest_df.iterrows():
        forecast_pm25 = float(row["pm25_pred_model"])
        forecast_aqi = float(row["aqi_pred_model"])

        pm25_uncertainty = float(max(row["pm25_uncertainty"], 1.0))

        threshold_probability = float(min(max((forecast_aqi - 100) / 250, 0), 1))

        ward_payload = {
            "ward_id": row["ward_id"],
            "ward_name": row["ward_name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "current_pm25": float(row["pm25"]),
            "forecast_pm25": forecast_pm25,
            "forecast_pm25_p10": float(max(forecast_pm25 - 1.28 * pm25_uncertainty, 0)),
            "forecast_pm25_p90": float(forecast_pm25 + 1.28 * pm25_uncertainty),
            "current_aqi": float(row["aqi"]),
            "forecast_aqi": forecast_aqi,
            "risk_level": classify_aqi(forecast_aqi),
            "threshold_crossing_probability": threshold_probability,
            "model_confidence": float(min(max(1 - pm25_uncertainty / 40, 0.25), 0.95)),
        }

        wards.append(ward_payload)

    return {
        "city": "Chennai",
        "generated_at": datetime.utcnow().isoformat(),
        "model_type": "lightgbm_forecasting_model",
        "horizon_hours": FORECAST_HORIZON_HOURS,
        "metrics_summary": metrics,
        "wards": wards,
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_df, test_df = time_based_split(df)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    models, predictions, metrics = train_models(train_df, test_df)

    baseline_metrics = load_baseline_metrics()
    metrics = add_baseline_comparison(metrics, baseline_metrics)

    forecast_output = build_forecast_output(predictions, metrics)

    model_path = ARTIFACTS_DIR / "forecast_models_lightgbm.joblib"
    metrics_path = ARTIFACTS_DIR / "forecast_model_metrics.json"
    forecast_path = ARTIFACTS_DIR / "forecast_output_model.json"

    joblib.dump(models, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump(forecast_output, f, indent=2)

    print("\nForecast model evaluation")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved forecast output to: {forecast_path}")


if __name__ == "__main__":
    main()