import json
from datetime import datetime
from pathlib import Path

from ml.config import ARTIFACTS_DIR, PROJECT_ROOT


REPORT_PATH = PROJECT_ROOT / "docs" / "model_report.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run baseline, forecasting, event-risk, and attribution scripts first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def num(value: float) -> str:
    return f"{value:.4f}"


def build_report() -> str:
    baseline_metrics = load_json(ARTIFACTS_DIR / "baseline_persistence_metrics.json")
    forecast_metrics = load_json(ARTIFACTS_DIR / "forecast_model_metrics.json")
    event_metrics = load_json(ARTIFACTS_DIR / "event_risk_metrics.json")
    attribution_metrics = load_json(ARTIFACTS_DIR / "source_attribution_metrics.json")

    aqi_rmse_improvement = forecast_metrics["aqi"]["rmse_improvement_vs_persistence"]
    pm25_rmse_improvement = forecast_metrics["pm25"]["rmse_improvement_vs_persistence"]
    pm10_rmse_improvement = forecast_metrics["pm10"]["rmse_improvement_vs_persistence"]

    report = f"""# AirGuard AI — ML Model Report

Generated at: `{datetime.utcnow().isoformat()} UTC`

## 1. Purpose

This report summarises the current Machine Learning foundation for **AirGuard AI**, a multi-agent urban air quality intelligence platform.

The ML layer supports three core intelligence tasks:

1. **Hyperlocal pollutant forecasting**
2. **High-pollution event risk detection**
3. **Probabilistic pollution source attribution**

The goal is not only to forecast average AQI, but to identify dangerous pollution episodes early enough for city-level intervention.

---

## 2. Dataset Status

Current dataset type:

- Synthetic but mechanism-driven development dataset
- City: Chennai
- Ward count: 10
- Time granularity: hourly
- Forecast horizon: 24 hours
- Contains simulated pollution mechanisms:
  - traffic emissions
  - road dust / resuspension
  - construction dust
  - industrial influence
  - waste-burning events
  - meteorological stagnation

Important note:

> This dataset is used for system development, model pipeline validation, and demo integration. The same pipeline is designed to later accept real CAAQMS, weather, satellite, road-network, and land-use data.

---

## 3. Baseline Model: Persistence Forecast

The persistence baseline assumes:

> Future pollutant value = current pollutant value

This is a necessary scientific benchmark.

### Baseline Metrics

| Target | MAE | RMSE |
|---|---:|---:|
| PM2.5 | {num(baseline_metrics["pm25"]["mae"])} | {num(baseline_metrics["pm25"]["rmse"])} |
| PM10 | {num(baseline_metrics["pm10"]["mae"])} | {num(baseline_metrics["pm10"]["rmse"])} |
| AQI | {num(baseline_metrics["aqi"]["mae"])} | {num(baseline_metrics["aqi"]["rmse"])} |

AQI category accuracy: **{pct(baseline_metrics["aqi_category_accuracy"])}**

High-pollution event recall: **{pct(baseline_metrics["high_pollution_event_recall"])}**

### Interpretation

The persistence model provides a simple but weak intervention baseline. It may capture stable conditions but misses many dangerous future pollution events.

---

## 4. Learned Forecasting Model

Model:

- LightGBM regression
- Targets:
  - PM2.5 at 24h horizon
  - PM10 at 24h horizon
  - AQI at 24h horizon

### Forecasting Metrics

| Target | MAE | RMSE | Baseline RMSE | RMSE Improvement |
|---|---:|---:|---:|---:|
| PM2.5 | {num(forecast_metrics["pm25"]["mae"])} | {num(forecast_metrics["pm25"]["rmse"])} | {num(forecast_metrics["pm25"]["baseline_rmse"])} | {pct(pm25_rmse_improvement)} |
| PM10 | {num(forecast_metrics["pm10"]["mae"])} | {num(forecast_metrics["pm10"]["rmse"])} | {num(forecast_metrics["pm10"]["baseline_rmse"])} | {pct(pm10_rmse_improvement)} |
| AQI | {num(forecast_metrics["aqi"]["mae"])} | {num(forecast_metrics["aqi"]["rmse"])} | {num(forecast_metrics["aqi"]["baseline_rmse"])} | {pct(aqi_rmse_improvement)} |

AQI category accuracy: **{pct(forecast_metrics["aqi_category_accuracy"])}**

High-pollution event recall from regression forecast: **{pct(forecast_metrics["high_pollution_event_recall"])}**

### Interpretation

The learned forecasting model significantly improves AQI RMSE over the persistence baseline.

However, the regression model alone is not sufficient for intervention planning because dangerous pollution events remain relatively rare and can be under-predicted when optimising average error.

This motivates a separate event-risk classifier.

---

## 5. High-Pollution Event Risk Model

Model:

- LightGBM binary classifier
- Event definition: **AQI >= 201 within 24 hours**

### Event-Risk Metrics

| Metric | Value |
|---|---:|
| Threshold | {num(event_metrics["threshold"])} |
| Accuracy | {pct(event_metrics["accuracy"])} |
| Precision | {pct(event_metrics["precision"])} |
| Recall | {pct(event_metrics["recall"])} |
| F1 | {pct(event_metrics["f1"])} |
| ROC-AUC | {num(event_metrics["roc_auc"])} |
| Average Precision | {num(event_metrics["average_precision"])} |

Confusion matrix:

```text
{event_metrics["confusion_matrix"]}