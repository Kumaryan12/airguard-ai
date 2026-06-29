import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.config import ARTIFACTS_DIR, FORECAST_HORIZON_HOURS, PROCESSED_DATA_DIR, RANDOM_STATE


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
    "population_vulnerability",
    "traffic_proxy",
    "city_stagnation_event",
    "waste_burning_event",
    "local_event",
    "pm25",
    "pm10",
    "no2",
    "aqi",
]
EVENT_THRESHOLD_AQI = 201


def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "sample_chennai_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python -m ml.data_processing.create_sample_dataset"
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "ward_id"]).reset_index(drop=True)

    df["high_pollution_event_24h"] = (
        df["aqi_target_24h"] >= EVENT_THRESHOLD_AQI
    ).astype(int)

    return df


def time_based_split(df: pd.DataFrame, train_fraction: float = 0.8):
    unique_times = np.array(sorted(df["timestamp"].unique()))
    split_idx = int(len(unique_times) * train_fraction)

    train_times = set(unique_times[:split_idx])
    test_times = set(unique_times[split_idx:])

    train_df = df[df["timestamp"].isin(train_times)].copy()
    test_df = df[df["timestamp"].isin(test_times)].copy()

    return train_df, test_df


def build_event_model(pos_weight: float) -> Pipeline:
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

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        class_weight={0: 1.0, 1: pos_weight},
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def choose_operating_threshold(y_true, y_prob):
    """
    We prefer recall for intervention planning, but still avoid absurd false positives.
    This searches thresholds and picks the best F1 with recall >= 0.60 if possible.
    """
    candidates = np.linspace(0.05, 0.95, 91)

    best = None

    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        if recall >= 0.60:
            if best is None or candidate["f1"] > best["f1"]:
                best = candidate

    if best is not None:
        return best

    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        if best is None or candidate["f1"] > best["f1"]:
            best = candidate

    return best


def evaluate_classifier(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics


def build_event_risk_output(test_df, y_prob, threshold, metrics):
    latest_ts = test_df["timestamp"].max()
    latest_df = test_df[test_df["timestamp"] == latest_ts].copy()
    latest_indices = latest_df.index.to_numpy()

    prob_series = pd.Series(y_prob, index=test_df.index)
    latest_df["event_probability_24h"] = prob_series.loc[latest_indices].values
    latest_df["event_predicted"] = (
        latest_df["event_probability_24h"] >= threshold
    ).astype(int)

    wards = []

    for _, row in latest_df.iterrows():
        wards.append(
            {
                "ward_id": row["ward_id"],
                "ward_name": row["ward_name"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "event_threshold_aqi": EVENT_THRESHOLD_AQI,
                "high_pollution_probability_24h": float(row["event_probability_24h"]),
                "high_pollution_event_predicted": bool(row["event_predicted"]),
                "current_aqi": float(row["aqi"]),
                "target_horizon_hours": FORECAST_HORIZON_HOURS,
            }
        )

    return {
        "city": "Chennai",
        "generated_at": datetime.utcnow().isoformat(),
        "model_type": "lightgbm_event_risk_classifier",
        "event_definition": f"AQI >= {EVENT_THRESHOLD_AQI} within 24h",
        "metrics_summary": metrics,
        "wards": wards,
    }


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_df, test_df = time_based_split(df)

    train_positive_rate = train_df["high_pollution_event_24h"].mean()
    test_positive_rate = test_df["high_pollution_event_24h"].mean()

    positives = train_df["high_pollution_event_24h"].sum()
    negatives = len(train_df) - positives
    pos_weight = negatives / max(positives, 1)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train positive event rate: {train_positive_rate:.4f}")
    print(f"Test positive event rate: {test_positive_rate:.4f}")
    print(f"Positive class weight: {pos_weight:.2f}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["high_pollution_event_24h"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["high_pollution_event_24h"]

    model = build_event_model(pos_weight=pos_weight)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    threshold_info = choose_operating_threshold(y_test.values, y_prob)
    threshold = threshold_info["threshold"]

    metrics = evaluate_classifier(y_test.values, y_prob, threshold)
    metrics["chosen_threshold_info"] = threshold_info
    metrics["train_positive_event_rate"] = float(train_positive_rate)
    metrics["test_positive_event_rate"] = float(test_positive_rate)

    event_output = build_event_risk_output(test_df, y_prob, threshold, metrics)

    model_path = ARTIFACTS_DIR / "event_risk_model_lightgbm.joblib"
    metrics_path = ARTIFACTS_DIR / "event_risk_metrics.json"
    output_path = ARTIFACTS_DIR / "event_risk_output.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(event_output, f, indent=2)

    print("\nEvent risk model evaluation")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved event risk output to: {output_path}")


if __name__ == "__main__":
    main()