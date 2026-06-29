import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

from ml.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT


INPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_historical_features.csv"
METRICS_PATH = ARTIFACTS_DIR / "real_cpcb_forecast_benchmark_metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "real_cpcb_forecast_test_predictions.csv"
MODEL_PATH = ARTIFACTS_DIR / "real_cpcb_forecast_best_model.joblib"

BACKEND_OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "sample"
    / "real_cpcb_forecast_benchmark_metrics.json"
)

TARGET_COL = "cpcb_aqi_target_24h"


EXCLUDE_COLUMNS = {
    "timestamp",
    "location_id",
    "estimated_aqi_category",
    "cpcb_aqi_category",
    "dominant_pollutant",
    "dispersion_risk",
    "pm25_target_24h",
    "pm10_target_24h",
    "estimated_aqi_target_24h",
    "high_pollution_event_24h",
    "cpcb_aqi_target_24h",
    "pm10_sub_index_target_24h",
    "pm25_sub_index_target_24h",
    "cpcb_high_pollution_event_24h",
}


def get_aqi_category(aqi: float) -> str:
    if pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderately Polluted"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    true_cat = [get_aqi_category(x) for x in y_true]
    pred_cat = [get_aqi_category(x) for x in y_pred]
    category_accuracy = float(np.mean(np.array(true_cat) == np.array(pred_cat)))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "aqi_category_accuracy": category_accuracy,
    }


def select_numeric_features(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    features = [
        col
        for col in numeric_cols
        if col not in EXCLUDE_COLUMNS
    ]

    features = [
        col
        for col in features
        if not col.endswith("_target_24h")
    ]

    return features


def temporal_train_test_split(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(df) * (1 - test_fraction))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def build_models(random_state: int = 42) -> Dict[str, Any]:
    models = {
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
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=8,
                        random_state=random_state,
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
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if LGBMRegressor is not None:
        models["lightgbm_conservative"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.03,
                        max_depth=4,
                        num_leaves=16,
                        min_child_samples=30,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    return models


def evaluate_baselines(test_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    y_true = test_df[TARGET_COL].values

    baselines = {}

    if "cpcb_aqi" in test_df.columns:
        pred = test_df["cpcb_aqi"].values
        baselines["persistence_current_cpcb_aqi"] = {
            **regression_metrics(y_true, pred),
            "type": "baseline",
        }

    if "cpcb_aqi_lag_24h" in test_df.columns:
        pred = test_df["cpcb_aqi_lag_24h"].values
        mask = ~pd.isna(pred)

        if mask.sum() > 0:
            baselines["seasonal_24h_lag"] = {
                **regression_metrics(y_true[mask], pred[mask]),
                "type": "baseline",
            }

    if "cpcb_aqi_rolling_mean_24h" in test_df.columns:
        pred = test_df["cpcb_aqi_rolling_mean_24h"].values
        mask = ~pd.isna(pred)

        if mask.sum() > 0:
            baselines["rolling_mean_24h"] = {
                **regression_metrics(y_true[mask], pred[mask]),
                "type": "baseline",
            }

    return baselines


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input feature table: {INPUT_PATH}. "
            "Run: python -m ml.data_ingestion.build_real_historical_features"
        )

    df = pd.read_csv(INPUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    print("Original shape:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df = df.dropna(subset=[TARGET_COL, "cpcb_aqi"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = select_numeric_features(df)

    print("Model shape:", df.shape)
    print(f"Selected features ({len(feature_cols)}):")
    print(feature_cols)

    train_df, test_df = temporal_train_test_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    results = evaluate_baselines(test_df)

    persistence_rmse = results.get(
        "persistence_current_cpcb_aqi",
        {},
    ).get("rmse")

    models = build_models()
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        pred = np.clip(pred, 0, 500)

        metrics = regression_metrics(y_test, pred)
        metrics["type"] = "learned_model"

        if persistence_rmse:
            metrics["rmse_improvement_vs_persistence"] = float(
                (persistence_rmse - metrics["rmse"]) / persistence_rmse
            )

        results[name] = metrics
        fitted_models[name] = model

    if persistence_rmse:
        for name, metrics in results.items():
            if "rmse_improvement_vs_persistence" not in metrics:
                metrics["rmse_improvement_vs_persistence"] = float(
                    (persistence_rmse - metrics["rmse"]) / persistence_rmse
                )

    best_overall = min(results, key=lambda name: results[name]["rmse"])

    learned_names = [
        name
        for name, metrics in results.items()
        if metrics.get("type") == "learned_model"
    ]

    baseline_names = [
        name
        for name, metrics in results.items()
        if metrics.get("type") == "baseline"
    ]

    best_learned_model = (
        min(learned_names, key=lambda name: results[name]["rmse"])
        if learned_names
        else None
    )

    best_baseline = (
        min(baseline_names, key=lambda name: results[name]["rmse"])
        if baseline_names
        else None
    )

    output = {
        "city": "Chennai",
        "station_location_id": int(df["location_id"].iloc[0])
        if "location_id" in df.columns
        else None,
        "model_type": "real_cpcb_forecast_benchmark",
        "target": TARGET_COL,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_cols)),
        "date_range": {
            "train_start": str(train_df["timestamp"].min()),
            "train_end": str(train_df["timestamp"].max()),
            "test_start": str(test_df["timestamp"].min()),
            "test_end": str(test_df["timestamp"].max()),
        },
        "results": results,
        "best_learned_model": best_learned_model,
        "best_baseline": best_baseline,
        "best_overall": best_overall,
        "important_warning": (
            "CPCB AQI target is computed from available latest pollutant values. "
            "For final regulatory AQI, pollutant-specific CPCB averaging windows should be used."
        ),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    BACKEND_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BACKEND_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    prediction_output = test_df[
        [
            "timestamp",
            "location_id",
            "cpcb_aqi",
            TARGET_COL,
        ]
    ].copy()

    if best_learned_model is not None:
        best_model = fitted_models[best_learned_model]
        best_pred = best_model.predict(X_test)
        prediction_output[f"{best_learned_model}_prediction"] = np.clip(
            best_pred,
            0,
            500,
        )

        joblib.dump(best_model, MODEL_PATH)

    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    print("\nReal CPCB forecast benchmark")
    print(json.dumps(output, indent=2))

    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved backend metrics to: {BACKEND_OUTPUT_PATH}")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")

    if best_learned_model is not None:
        print(f"Saved best learned model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()