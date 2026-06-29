import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE


INPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_historical_features.csv"


BASE_FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "rush_hour",
    "weekend",
    "night_stagnation",
    "pm25",
    "pm10",
    "no2",
    "o3",
    "co",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "dispersion_penalty",
    "pm10_pm25_ratio",
    "road_density_km_per_km2",
    "major_road_density_km_per_km2",
    "nearest_major_road_m",
    "industrial_poi_count",
    "construction_poi_count",
    "green_poi_count",
    "vulnerability_poi_count",
]


TARGET_COLUMN = "estimated_aqi_target_24h"
CURRENT_AQI_COLUMN = "estimated_aqi"


def classify_aqi(aqi: float) -> str:
    if pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing real feature table: {INPUT_PATH}. "
            "Run: python -m ml.data_ingestion.build_real_historical_features"
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    return df.sort_values(["timestamp", "location_id"]).reset_index(drop=True)


def select_features(df: pd.DataFrame) -> list:
    lag_features = [
        col for col in df.columns
        if "_lag_" in col or "_rolling_mean_" in col
    ]

    features = [
        col for col in BASE_FEATURE_COLUMNS + lag_features
        if col in df.columns
    ]

    usable = []

    for col in features:
        missing_rate = df[col].isna().mean()
        if missing_rate <= 0.4:
            usable.append(col)

    return usable


def time_split(df: pd.DataFrame, train_fraction: float = 0.75):
    unique_times = np.array(sorted(df["timestamp"].unique()))
    split_idx = int(len(unique_times) * train_fraction)

    train_times = set(unique_times[:split_idx])
    test_times = set(unique_times[split_idx:])

    train_df = df[df["timestamp"].isin(train_times)].copy()
    test_df = df[df["timestamp"].isin(test_times)].copy()

    return train_df, test_df


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    feature_columns = select_features(df)

    model_df = df.dropna(subset=feature_columns + [TARGET_COLUMN, CURRENT_AQI_COLUMN]).copy()

    print(f"Original shape: {df.shape}")
    print(f"Model shape after dropping missing features: {model_df.shape}")
    print(f"Selected features ({len(feature_columns)}):")
    print(feature_columns)

    if len(model_df) < 200:
        raise ValueError(
            f"Not enough rows for real training after cleaning: {len(model_df)}. "
            "Need more historical data or fewer required features."
        )

    train_df, test_df = time_split(model_df)

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    baseline_pred = test_df[CURRENT_AQI_COLUMN].values

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    model_pred = model.predict(X_test)

    baseline_metrics = evaluate(y_test, baseline_pred)
    model_metrics = evaluate(y_test, model_pred)

    improvement = (
        baseline_metrics["rmse"] - model_metrics["rmse"]
    ) / baseline_metrics["rmse"]

    test_result = test_df[
        [
            "timestamp",
            "location_id",
            CURRENT_AQI_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()

    test_result["baseline_pred"] = baseline_pred
    test_result["model_pred"] = model_pred
    test_result["actual_category"] = test_result[TARGET_COLUMN].apply(classify_aqi)
    test_result["model_category"] = test_result["model_pred"].apply(classify_aqi)

    category_accuracy = (
        test_result["actual_category"] == test_result["model_category"]
    ).mean()

    metrics = {
        "city": "Chennai",
        "station_location_id": int(model_df["location_id"].iloc[0]),
        "model_type": "real_historical_lightgbm_regression",
        "target": TARGET_COLUMN,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "date_range": {
            "train_start": str(train_df["timestamp"].min()),
            "train_end": str(train_df["timestamp"].max()),
            "test_start": str(test_df["timestamp"].min()),
            "test_end": str(test_df["timestamp"].max()),
        },
        "baseline_persistence": baseline_metrics,
        "model": model_metrics,
        "rmse_improvement_vs_persistence": float(improvement),
        "aqi_category_accuracy": float(category_accuracy),
        "important_warning": (
            "First real model is trained on one OpenAQ station and approximate AQI. "
            "This validates the real-data training pipeline, not full citywide deployment."
        ),
    }

    model_path = ARTIFACTS_DIR / "real_forecast_model_lightgbm.joblib"
    metrics_path = ARTIFACTS_DIR / "real_forecast_model_metrics.json"
    preds_path = ARTIFACTS_DIR / "real_forecast_test_predictions.csv"

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        model_path,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    test_result.to_csv(preds_path, index=False)

    print("\nReal historical forecast validation")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved test predictions to: {preds_path}")


if __name__ == "__main__":
    main()