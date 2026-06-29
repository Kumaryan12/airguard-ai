import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from ml.forecasting.train_forecast_model import FEATURE_COLUMNS


OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "model_explanations.json"


FEATURE_DISPLAY_NAMES = {
    "hour": "hour of day",
    "dayofweek": "day of week",
    "rush_hour": "rush-hour indicator",
    "night_stagnation": "night-time stagnation indicator",
    "weekend": "weekend indicator",
    "temperature": "temperature",
    "humidity": "humidity",
    "wind_speed": "wind speed",
    "wind_direction": "wind direction",
    "road_density": "road density",
    "construction_score": "construction activity proxy",
    "industrial_score": "industrial proximity score",
    "green_cover": "green cover",
    "population_vulnerability": "population vulnerability",
    "traffic_proxy": "traffic proxy",
    "city_stagnation_event": "citywide stagnation indicator",
    "waste_burning_event": "waste-burning event proxy",
    "local_event": "local episodic event indicator",
    "pm25": "current PM2.5",
    "pm10": "current PM10",
    "no2": "current NO2",
    "aqi": "current AQI",
}


def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "sample_chennai_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python -m ml.data_processing.create_sample_dataset"
        )

    return pd.read_csv(path, parse_dates=["timestamp"])


def load_forecast_models() -> dict:
    path = ARTIFACTS_DIR / "forecast_models_lightgbm.joblib"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {path}. "
            "Run: python -m ml.forecasting.train_forecast_model"
        )

    return joblib.load(path)


def time_based_test_split(df: pd.DataFrame, train_fraction: float = 0.8) -> pd.DataFrame:
    unique_times = np.array(sorted(df["timestamp"].unique()))
    split_idx = int(len(unique_times) * train_fraction)
    test_times = set(unique_times[split_idx:])
    return df[df["timestamp"].isin(test_times)].copy()


def make_feature_readable(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


def global_permutation_explanation(model, X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """
    Uses permutation importance instead of SHAP for first stable version.

    Why:
    - works cleanly with sklearn Pipeline
    - does not break with one-hot preprocessing
    - produces understandable global feature importance

    Later we can add true SHAP on the transformed LightGBM features.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=8,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    importances = []

    for feature, mean_importance, std_importance in zip(
        X.columns,
        result.importances_mean,
        result.importances_std,
    ):
        importances.append(
            {
                "feature": feature,
                "display_name": make_feature_readable(feature),
                "importance": float(mean_importance),
                "importance_std": float(std_importance),
            }
        )

    importances = sorted(
        importances,
        key=lambda item: item["importance"],
        reverse=True,
    )

    return importances


def local_driver_explanation(row: pd.Series) -> list[dict]:
    """
    Local readable evidence for the latest ward row.
    This is not a replacement for SHAP; it is a human-readable feature evidence layer.
    """
    drivers = []

    if row["wind_speed"] <= 1.2:
        drivers.append(
            {
                "driver": "low wind speed",
                "value": float(row["wind_speed"]),
                "interpretation": "Poor dispersion can allow pollutants to accumulate.",
            }
        )

    if row["traffic_proxy"] >= 0.65:
        drivers.append(
            {
                "driver": "high traffic proxy",
                "value": float(row["traffic_proxy"]),
                "interpretation": "Traffic intensity can increase NO2 and particulate exposure.",
            }
        )

    if row["road_density"] >= 0.7:
        drivers.append(
            {
                "driver": "dense road network",
                "value": float(row["road_density"]),
                "interpretation": "Dense roads increase exposure to traffic emissions and resuspended dust.",
            }
        )

    if row["construction_score"] >= 0.65:
        drivers.append(
            {
                "driver": "high construction activity proxy",
                "value": float(row["construction_score"]),
                "interpretation": "Construction activity can contribute coarse particulate dust.",
            }
        )

    if row["industrial_score"] >= 0.7:
        drivers.append(
            {
                "driver": "high industrial proximity score",
                "value": float(row["industrial_score"]),
                "interpretation": "Industrial proximity may contribute combustion and fine particulate load.",
            }
        )

    if row["waste_burning_event"] == 1:
        drivers.append(
            {
                "driver": "waste-burning event proxy active",
                "value": int(row["waste_burning_event"]),
                "interpretation": "Thermal/waste-burning episodes can sharply raise PM2.5.",
            }
        )

    if row["pm25"] >= 70:
        drivers.append(
            {
                "driver": "elevated current PM2.5",
                "value": float(row["pm25"]),
                "interpretation": "Current fine particulate pollution is already elevated.",
            }
        )

    if row["pm10"] >= 110:
        drivers.append(
            {
                "driver": "elevated current PM10",
                "value": float(row["pm10"]),
                "interpretation": "Current coarse particulate pollution is already elevated.",
            }
        )

    if not drivers:
        drivers.append(
            {
                "driver": "combined spatiotemporal signal",
                "value": None,
                "interpretation": "Prediction is driven by the combined pattern of weather, location, time, and current pollutant levels.",
            }
        )

    return drivers[:5]


def build_explanations() -> dict:
    df = load_dataset()
    test_df = time_based_test_split(df)

    models = load_forecast_models()
    aqi_model = models["aqi"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["aqi_target_24h"]

    print("Computing global permutation importance for AQI model...")
    global_importance = global_permutation_explanation(aqi_model, X_test, y_test)

    latest_ts = test_df["timestamp"].max()
    latest_df = test_df[test_df["timestamp"] == latest_ts].copy()

    wards = []

    for _, row in latest_df.iterrows():
        wards.append(
            {
                "ward_id": row["ward_id"],
                "ward_name": row["ward_name"],
                "local_drivers": local_driver_explanation(row),
            }
        )

    payload = {
        "project": "AirGuard AI",
        "city": "Chennai",
        "generated_at": datetime.utcnow().isoformat(),
        "explanation_type": "permutation_importance_plus_local_evidence",
        "model_explained": "AQI 24h LightGBM forecasting model",
        "global_top_features": global_importance[:12],
        "wards": wards,
        "limitations": [
            "Permutation importance is a global model explanation method, not causal proof.",
            "Local driver cards are evidence summaries from observed features.",
            "Future version can add SHAP values on transformed LightGBM features.",
        ],
    }

    return payload


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = build_explanations()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved model explanations to: {OUTPUT_PATH}")
    print("\nTop global AQI drivers:")
    for item in payload["global_top_features"][:8]:
        print(
            item["display_name"],
            "importance:",
            round(item["importance"], 4),
        )


if __name__ == "__main__":
    main()