import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ml.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE


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

TARGET_COLUMN = "dominant_source_target_24h"


SOURCE_DISPLAY_NAMES = {
    "road_dust": "road dust / resuspension",
    "traffic": "traffic emissions",
    "construction": "construction dust",
    "industrial": "industrial influence",
    "waste_burning": "waste burning / thermal anomaly",
    "meteorology": "meteorological trapping",
}


def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "sample_chennai_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python -m ml.data_processing.create_sample_dataset"
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "ward_id"]).reset_index(drop=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    return df


def time_based_split(df: pd.DataFrame, train_fraction: float = 0.8):
    unique_times = np.array(sorted(df["timestamp"].unique()))
    split_idx = int(len(unique_times) * train_fraction)

    train_times = set(unique_times[:split_idx])
    test_times = set(unique_times[split_idx:])

    train_df = df[df["timestamp"].isin(train_times)].copy()
    test_df = df[df["timestamp"].isin(test_times)].copy()

    return train_df, test_df


def build_source_model() -> Pipeline:
    categorical_features = ["ward_id"]
    numeric_features = [col for col in FEATURE_COLUMNS if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        objective="multiclass",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def source_evidence_from_features(row: pd.Series, source: str) -> list[str]:
    """
    Evidence layer. The source probability comes from the learned model.
    This function converts strong feature signals into readable evidence cards.

    It is not the decision-maker; it explains the model output using available features.
    Later we can upgrade this with SHAP.
    """
    evidence = []

    if source == "traffic":
        if row["traffic_proxy"] >= 0.65:
            evidence.append("high learned traffic proxy")
        if row["rush_hour"] == 1:
            evidence.append("rush-hour temporal pattern")
        if row["no2"] >= 45:
            evidence.append("elevated NO2 signal associated with combustion traffic")

    elif source == "road_dust":
        if row["road_density"] >= 0.70:
            evidence.append("dense road network exposure")
        if row["pm10"] / max(row["pm25"], 1) >= 1.25:
            evidence.append("PM10-heavy particulate pattern")
        if row["green_cover"] <= 0.25:
            evidence.append("low green-cover buffering")

    elif source == "construction":
        if row["construction_score"] >= 0.65:
            evidence.append("high construction activity proxy")
        if row["local_event"] == 1:
            evidence.append("local episodic dust event detected")
        if row["pm10"] / max(row["pm25"], 1) >= 1.25:
            evidence.append("coarse particle signature")

    elif source == "industrial":
        if row["industrial_score"] >= 0.70:
            evidence.append("high industrial proximity score")
        if row["pm25"] >= 70:
            evidence.append("elevated fine-particle load")
        if row["no2"] >= 40:
            evidence.append("combustion-related NO2 elevation")

    elif source == "waste_burning":
        if row["waste_burning_event"] == 1:
            evidence.append("thermal/waste-burning event proxy active")
        if row["hour"] in [20, 21, 22, 23, 0, 1]:
            evidence.append("night-time burning-prone window")
        if row["pm25"] >= 75:
            evidence.append("sharp fine-particle elevation")

    elif source == "meteorology":
        if row["wind_speed"] <= 1.0:
            evidence.append("low wind speed indicates poor dispersion")
        if row["city_stagnation_event"] == 1:
            evidence.append("citywide stagnation episode detected")
        if row["humidity"] >= 75:
            evidence.append("high humidity supports pollutant accumulation")

    if not evidence:
        evidence.append("model probability supported by combined spatiotemporal feature pattern")

    return evidence[:4]


def evaluate_model(y_true, y_pred, label_encoder: LabelEncoder) -> dict:
    labels = list(range(len(label_encoder.classes_)))
    target_names = list(label_encoder.classes_)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "class_labels": target_names,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def build_attribution_output(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    label_encoder: LabelEncoder,
    metrics: dict,
) -> dict:
    latest_ts = test_df["timestamp"].max()
    latest_df = test_df[test_df["timestamp"] == latest_ts].copy()

    prob_df = pd.DataFrame(
        probabilities,
        index=test_df.index,
        columns=label_encoder.classes_,
    )

    wards = []

    for idx, row in latest_df.iterrows():
        source_probs = prob_df.loc[idx].sort_values(ascending=False)

        dominant_sources = []

        for source, probability in source_probs.head(3).items():
            dominant_sources.append(
                {
                    "source": SOURCE_DISPLAY_NAMES.get(source, source),
                    "source_key": source,
                    "probability": float(probability),
                    "evidence": source_evidence_from_features(row, source),
                }
            )

        top_probability = float(source_probs.iloc[0])

        if top_probability >= 0.70:
            confidence = "high"
        elif top_probability >= 0.50:
            confidence = "medium-high"
        elif top_probability >= 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        wards.append(
            {
                "ward_id": row["ward_id"],
                "ward_name": row["ward_name"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "dominant_sources": dominant_sources,
                "attribution_confidence": confidence,
                "causal_warning": "Probabilistic attribution, not direct causal proof.",
            }
        )

    return {
        "city": "Chennai",
        "generated_at": datetime.utcnow().isoformat(),
        "model_type": "lightgbm_multiclass_source_attribution",
        "target": "dominant probable source driver at 24h horizon",
        "metrics_summary": metrics,
        "wards": wards,
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_df, test_df = time_based_split(df)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("\nSource distribution:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).round(3))

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[TARGET_COLUMN])
    y_test = label_encoder.transform(test_df[TARGET_COLUMN])

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    model = build_source_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    metrics = evaluate_model(y_test, y_pred, label_encoder)
    attribution_output = build_attribution_output(
        test_df=test_df,
        probabilities=probabilities,
        label_encoder=label_encoder,
        metrics=metrics,
    )

    model_bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
    }

    model_path = ARTIFACTS_DIR / "source_attribution_model_lightgbm.joblib"
    metrics_path = ARTIFACTS_DIR / "source_attribution_metrics.json"
    output_path = ARTIFACTS_DIR / "attribution_output_model.json"

    joblib.dump(model_bundle, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(attribution_output, f, indent=2)

    print("\nSource attribution model evaluation")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved attribution output to: {output_path}")


if __name__ == "__main__":
    main()