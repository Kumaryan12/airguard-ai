import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from ml.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT


INPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_historical_features.csv"

METRICS_PATH = ARTIFACTS_DIR / "cpcb_event_classifier_metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "cpcb_event_classifier_test_predictions.csv"
MODEL_PATH = ARTIFACTS_DIR / "cpcb_event_classifier_best_model.joblib"

BACKEND_OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "sample"
    / "cpcb_event_classifier_metrics.json"
)

TARGET_COL = "cpcb_window_preventive_risk_event_24h"


EXCLUDE_COLUMNS = {
    "timestamp",
    "location_id",
    "estimated_aqi_category",
    "cpcb_aqi_category",
    "cpcb_window_aqi_category",
    "dominant_pollutant",
    "cpcb_window_dominant_pollutant",
    "dispersion_risk",
    "pm25_target_24h",
    "pm10_target_24h",
    "estimated_aqi_target_24h",
    "high_pollution_event_24h",
    "cpcb_aqi_target_24h",
    "pm10_sub_index_target_24h",
    "pm25_sub_index_target_24h",
    "cpcb_high_pollution_event_24h",
    "cpcb_window_aqi_target_24h",
    "pm10_window_sub_index_target_24h",
    "pm25_window_sub_index_target_24h",
    "cpcb_window_high_pollution_event_24h",
    "cpcb_window_preventive_risk_event_24h",
"cpcb_window_aqi_delta_target_24h",
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

    features = [
        col
        for col in features
        if df[col].notna().sum() > 0
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


def safe_auc(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    return float(average_precision_score(y_true, y_score))


def classifier_metrics(y_true, y_pred, y_score) -> Dict[str, Any]:
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


def build_models(random_state: int = 42) -> Dict[str, Any]:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees_classifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest_classifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if LGBMClassifier is not None:
        models["lightgbm_classifier"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=250,
                        learning_rate=0.03,
                        max_depth=4,
                        num_leaves=16,
                        min_child_samples=25,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        class_weight="balanced",
                        random_state=random_state,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    return models


def get_positive_score(model, X_test) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    decision = model.decision_function(X_test)
    return 1 / (1 + np.exp(-decision))


def evaluate_baselines(test_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    y_true = test_df[TARGET_COL].astype(int).values

    baselines = {}

    if "cpcb_window_aqi" in test_df.columns:
        pred = (test_df["cpcb_window_aqi"].values >= 76).astype(int)
        score = test_df["cpcb_window_aqi"].values / 500.0

        baselines["persistence_current_event"] = {
            **classifier_metrics(y_true, pred, score),
            "type": "baseline",
        }

    if "cpcb_window_aqi_lag_24h" in test_df.columns:
        pred = (test_df["cpcb_window_aqi"].values >= 76).astype(int)
        score = test_df["cpcb_window_aqi_lag_24h"].fillna(0).values / 500.0

        baselines["seasonal_24h_lag_event"] = {
            **classifier_metrics(y_true, pred, score),
            "type": "baseline",
        }

    if "cpcb_window_aqi_rolling_mean_24h" in test_df.columns:
        pred = (test_df["cpcb_window_aqi"].values >= 76).astype(int)
        score = test_df["cpcb_window_aqi_rolling_mean_24h"].fillna(0).values / 500.0

        baselines["rolling_mean_24h_event"] = {
            **classifier_metrics(y_true, pred, score),
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

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = select_numeric_features(df)

    print("Model shape:", df.shape)
    print("Target:", TARGET_COL)
    print("Event rate:", float(df[TARGET_COL].mean()))
    print(f"Selected features ({len(feature_cols)}):")
    print(feature_cols)

    train_df, test_df = temporal_train_test_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int).values

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int).values

    print("\nTrain event rate:", float(np.mean(y_train)))
    print("Test event rate:", float(np.mean(y_test)))

    results = evaluate_baselines(test_df)

    models = build_models()
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_score = get_positive_score(model, X_test)
        y_pred = (y_score >= 0.5).astype(int)

        metrics = classifier_metrics(y_test, y_pred, y_score)
        metrics["type"] = "learned_model"

        results[name] = metrics
        fitted_models[name] = model

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
        max(learned_names, key=lambda name: results[name]["f1"])
        if learned_names
        else None
    )

    best_baseline = (
        max(baseline_names, key=lambda name: results[name]["f1"])
        if baseline_names
        else None
    )

    best_overall = max(results, key=lambda name: results[name]["f1"])

    output = {
        "city": "Chennai",
        "station_location_id": int(df["location_id"].iloc[0])
        if "location_id" in df.columns
        else None,
        "model_type": "cpcb_window_event_classifier",
        "target": TARGET_COL,
        "event_definition": "cpcb_window_aqi_target_24h >= 76 OR 24h AQI increase >= 15",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_cols)),
        "event_rate_overall": float(df[TARGET_COL].mean()),
        "event_rate_train": float(np.mean(y_train)),
        "event_rate_test": float(np.mean(y_test)),
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
        "selection_metric": "f1",
        "important_warning": (
            "This is an event-risk classifier for CPCB-window AQI crossing "
            "Moderately Polluted threshold in 24 hours. It is trained on one station, "
            "so results validate the prototype pipeline rather than citywide deployment."
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
            "cpcb_window_aqi",
            "cpcb_window_aqi_target_24h",
            TARGET_COL,
        ]
    ].copy()

    if best_learned_model is not None:
        best_model = fitted_models[best_learned_model]
        best_score = get_positive_score(best_model, X_test)
        best_pred = (best_score >= 0.5).astype(int)

        prediction_output[f"{best_learned_model}_risk_score"] = best_score
        prediction_output[f"{best_learned_model}_prediction"] = best_pred

        joblib.dump(best_model, MODEL_PATH)

    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    print("\nCPCB event classifier benchmark")
    print(json.dumps(output, indent=2))

    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved backend metrics to: {BACKEND_OUTPUT_PATH}")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")

    if best_learned_model is not None:
        print(f"Saved best learned model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()