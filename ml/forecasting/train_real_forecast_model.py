import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE


INPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_historical_features.csv"

TARGET_COLUMN = "estimated_aqi_target_24h"
CURRENT_AQI_COLUMN = "estimated_aqi"


CORE_FEATURES = [
    "hour",
    "dayofweek",
    "rush_hour",
    "weekend",
    "night_stagnation",
    "pm25",
    "pm10",
    "no2",
    "co",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "surface_pressure",
    "dispersion_penalty",
    "pm10_pm25_ratio",
]


COMPACT_LAG_FEATURES = [
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",
    "pm25_rolling_mean_3h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_mean_12h",
    "pm25_rolling_mean_24h",
    "pm10_lag_1h",
    "pm10_lag_3h",
    "pm10_lag_6h",
    "pm10_lag_12h",
    "pm10_lag_24h",
    "pm10_rolling_mean_3h",
    "pm10_rolling_mean_6h",
    "pm10_rolling_mean_12h",
    "pm10_rolling_mean_24h",
    "estimated_aqi_lag_1h",
    "estimated_aqi_lag_3h",
    "estimated_aqi_lag_6h",
    "estimated_aqi_lag_12h",
    "estimated_aqi_lag_24h",
    "estimated_aqi_rolling_mean_3h",
    "estimated_aqi_rolling_mean_6h",
    "estimated_aqi_rolling_mean_12h",
    "estimated_aqi_rolling_mean_24h",
]


OPTIONAL_GEOSPATIAL_FEATURES = [
    "road_density_km_per_km2",
    "major_road_density_km_per_km2",
    "nearest_major_road_m",
    "industrial_poi_count",
    "construction_poi_count",
    "green_poi_count",
    "vulnerability_poi_count",
]


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


def select_features(df: pd.DataFrame, include_geospatial: bool = False) -> list:
    candidate_features = CORE_FEATURES + COMPACT_LAG_FEATURES

    if include_geospatial:
        candidate_features += OPTIONAL_GEOSPATIAL_FEATURES

    features = [col for col in candidate_features if col in df.columns]

    usable = []

    for col in features:
        missing_rate = df[col].isna().mean()
        unique_count = df[col].nunique(dropna=True)

        if missing_rate <= 0.5 and unique_count > 1:
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


def category_accuracy(y_true, y_pred) -> float:
    true_cat = pd.Series(y_true).apply(classify_aqi)
    pred_cat = pd.Series(y_pred).apply(classify_aqi)
    return float((true_cat.values == pred_cat.values).mean())


def make_models() -> dict:
    return {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lightgbm_conservative": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=160,
                        learning_rate=0.03,
                        num_leaves=7,
                        max_depth=4,
                        min_child_samples=30,
                        subsample=0.8,
                        colsample_bytree=0.75,
                        reg_alpha=0.4,
                        reg_lambda=2.0,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    feature_columns = select_features(df, include_geospatial=False)

    required = [TARGET_COLUMN, CURRENT_AQI_COLUMN]
    model_df = df.dropna(subset=required).copy()

    print(f"Original shape: {df.shape}")
    print(f"Model shape before feature imputation: {model_df.shape}")
    print(f"Selected compact features ({len(feature_columns)}):")
    print(feature_columns)

    if len(model_df) < 500:
        raise ValueError(
            f"Not enough rows for real training: {len(model_df)}. "
            "Need more historical data."
        )

    train_df, test_df = time_split(model_df)

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    baselines = {}

    baselines["persistence_current_aqi"] = test_df[CURRENT_AQI_COLUMN].values

    if "estimated_aqi_lag_24h" in test_df.columns:
        baselines["seasonal_24h_lag"] = test_df["estimated_aqi_lag_24h"].fillna(
            test_df[CURRENT_AQI_COLUMN]
        ).values

    if "estimated_aqi_rolling_mean_24h" in test_df.columns:
        baselines["rolling_mean_24h"] = test_df["estimated_aqi_rolling_mean_24h"].fillna(
            test_df[CURRENT_AQI_COLUMN]
        ).values

    results = {}

    for baseline_name, pred in baselines.items():
        results[baseline_name] = {
            **evaluate(y_test, pred),
            "aqi_category_accuracy": category_accuracy(y_test, pred),
            "type": "baseline",
        }

    trained_models = make_models()
    fitted_models = {}

    for model_name, model in trained_models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        fitted_models[model_name] = model

        results[model_name] = {
            **evaluate(y_test, pred),
            "aqi_category_accuracy": category_accuracy(y_test, pred),
            "type": "learned_model",
        }

    reference_baseline = results["persistence_current_aqi"]["rmse"]

    for name, metrics in results.items():
        metrics["rmse_improvement_vs_persistence"] = float(
            (reference_baseline - metrics["rmse"]) / reference_baseline
        )

    best_model_name = min(
        [name for name, metrics in results.items() if metrics["type"] == "learned_model"],
        key=lambda name: results[name]["rmse"],
    )

    best_baseline_name = min(
        [name for name, metrics in results.items() if metrics["type"] == "baseline"],
        key=lambda name: results[name]["rmse"],
    )

    best_overall_name = min(results.keys(), key=lambda name: results[name]["rmse"])

    metrics_payload = {
        "city": "Chennai",
        "station_location_id": int(model_df["location_id"].iloc[0]),
        "model_type": "real_historical_forecast_benchmark",
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
        "results": results,
        "best_learned_model": best_model_name,
        "best_baseline": best_baseline_name,
        "best_overall": best_overall_name,
        "important_warning": (
            "This is a first real-data benchmark on one OpenAQ station using approximate AQI. "
            "It validates the training/evaluation pipeline, but citywide deployment requires more stations, "
            "official AQI calculation, and longer historical coverage."
        ),
    }

    model_path = ARTIFACTS_DIR / "real_forecast_best_model.joblib"
    metrics_path = ARTIFACTS_DIR / "real_forecast_benchmark_metrics.json"

    if best_model_name in fitted_models:
        joblib.dump(
            {
                "model": fitted_models[best_model_name],
                "feature_columns": feature_columns,
                "best_model_name": best_model_name,
            },
            model_path,
        )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nReal historical forecast benchmark")
    print(json.dumps(metrics_payload, indent=2))

    print(f"\nSaved benchmark metrics to: {metrics_path}")
    print(f"Saved best learned model to: {model_path}")


if __name__ == "__main__":
    main()