# Person 1 ML Handoff — AirGuard AI

## Owner

**Person 1: AI/ML + Geospatial Intelligence Owner**

This document explains the current Machine Learning and geospatial intelligence layer for AirGuard AI and how Person 2 can consume the outputs for the agentic backend and dashboard.

---

## 1. Purpose of Person 1 Layer

The Person 1 ML layer provides intelligence outputs for:

1. Hyperlocal AQI / PM2.5 / PM10 forecasting
2. High-pollution event risk detection
3. Probabilistic pollution source attribution
4. Forecast explanation and evidence signals
5. Integration-ready JSON outputs for the agentic platform

The Person 2 layer should consume these outputs and build:

1. Supervisor Agent
2. Intervention Optimisation Agent
3. Evidence Verification Agent
4. Citizen Advisory Agent
5. Command Centre Dashboard
6. Decision Memo Generator

---

## 2. Current Dataset

Current dataset file:

```text
ml/data/processed/sample_chennai_features.csv
```

This file is generated locally and is not committed to Git because processed CSV files are ignored.

Generate it using:

```bash
python -m ml.data_processing.create_sample_dataset
```

Dataset type:

- Synthetic but mechanism-driven development dataset
- City: Chennai
- Ward count: 10
- Time granularity: hourly
- Forecast horizon: 24 hours

The dataset simulates pollution mechanisms such as:

- traffic emissions
- road dust / resuspension
- construction dust
- industrial influence
- waste-burning events
- meteorological stagnation

Important note:

This dataset is used for development, pipeline validation, and demo integration. The same code structure is designed to later accept real CAAQMS, weather, satellite, road-network, and land-use data.

---

## 3. ML Models Built

### 3.1 Persistence Baseline

Script:

```text
ml/forecasting/baseline_persistence.py
```

Run:

```bash
python -m ml.forecasting.baseline_persistence
```

Purpose:

This creates the scientific baseline.

The persistence model assumes:

```text
future pollutant value = current pollutant value
```

Outputs generated locally:

```text
ml/artifacts/baseline_persistence_metrics.json
ml/artifacts/forecast_output_baseline.json
```

---

### 3.2 Learned Forecasting Model

Script:

```text
ml/forecasting/train_forecast_model.py
```

Run:

```bash
python -m ml.forecasting.train_forecast_model
```

Model:

```text
LightGBM regression
```

Targets:

- PM2.5 at 24h horizon
- PM10 at 24h horizon
- AQI at 24h horizon

Outputs generated locally:

```text
ml/artifacts/forecast_models_lightgbm.joblib
ml/artifacts/forecast_model_metrics.json
ml/artifacts/forecast_output_model.json
```

Main use for Person 2:

Use the forecast output to display ward-level current AQI, forecast AQI, PM2.5 forecast interval, and risk level.

---

### 3.3 High-Pollution Event Risk Model

Script:

```text
ml/forecasting/train_event_risk_model.py
```

Run:

```bash
python -m ml.forecasting.train_event_risk_model
```

Model:

```text
LightGBM binary classifier
```

Event definition:

```text
AQI >= 201 within 24 hours
```

Outputs generated locally:

```text
ml/artifacts/event_risk_model_lightgbm.joblib
ml/artifacts/event_risk_metrics.json
ml/artifacts/event_risk_output.json
```

Main use for Person 2:

Use this output as the hotspot trigger for the intervention agent.

The event-risk classifier is separate from the regression model because regression can improve average RMSE while still missing dangerous rare spikes.

---

### 3.4 Source Attribution Model

Script:

```text
ml/attribution/train_source_attribution_model.py
```

Run:

```bash
python -m ml.attribution.train_source_attribution_model
```

Model:

```text
LightGBM multiclass classifier
```

Source classes:

- road dust / resuspension
- traffic emissions
- construction dust
- industrial influence
- waste burning / thermal anomaly
- meteorological trapping

Outputs generated locally:

```text
ml/artifacts/source_attribution_model_lightgbm.joblib
ml/artifacts/source_attribution_metrics.json
ml/artifacts/attribution_output_model.json
```

Main use for Person 2:

Use this output to show probable source drivers, confidence, and supporting evidence cards in the dashboard.

Important wording:

The attribution output must be presented as:

```text
Probabilistic source attribution, not direct causal proof.
```

---

### 3.5 Forecast Explanation Output

Script:

```text
ml/explainability/explain_forecast_model.py
```

Run:

```bash
python -m ml.explainability.explain_forecast_model
```

Current method:

```text
Permutation importance + local feature evidence cards
```

Output committed for integration:

```text
backend/data/sample/model_explanations.json
```

Main use for Person 2:

Use this output to explain why the AQI forecast model is making certain predictions.

Example dashboard section:

```text
Top AQI forecast drivers:
1. rush-hour indicator
2. construction activity proxy
3. current NO2
4. industrial proximity score
5. current AQI
```

Limitation:

Permutation importance is not causal proof. It explains model sensitivity, not real-world causality.

---

## 4. Integration-Ready Files for Person 2

Person 2 should mainly consume this file:

```text
backend/data/sample/airguard_intelligence_output.json
```

This is the combined ML intelligence output.

It contains:

- ward ID
- ward name
- latitude / longitude
- current PM2.5
- forecast PM2.5
- forecast PM2.5 uncertainty interval
- current AQI
- forecast AQI
- AQI risk level
- event-risk probability
- high-pollution event prediction
- source attribution probabilities
- source evidence cards
- priority score
- priority band

---

## 5. Main Combined Output Structure

File:

```text
backend/data/sample/airguard_intelligence_output.json
```

Top-level structure:

```json
{
  "project": "AirGuard AI",
  "city": "Chennai",
  "generated_at": "...",
  "output_type": "combined_ml_intelligence",
  "description": "...",
  "models_summary": {},
  "wards": []
}
```

Each ward contains:

```json
{
  "ward_id": "W02",
  "ward_name": "Guindy",
  "lat": 13.0108,
  "lon": 80.2206,
  "forecast": {},
  "event_risk": {},
  "source_attribution": {},
  "priority": {}
}
```

---

## 6. Forecast Object

Inside each ward:

```json
"forecast": {
  "horizon_hours": 24,
  "current_pm25": 74.2,
  "forecast_pm25": 86.5,
  "forecast_pm25_p10": 72.1,
  "forecast_pm25_p90": 101.3,
  "current_aqi": 148.2,
  "forecast_aqi": 171.0,
  "risk_level": "Moderate",
  "model_confidence": 0.73
}
```

Person 2 dashboard use:

- current AQI card
- forecast AQI card
- forecast curve / interval
- risk-level badge
- confidence badge

---

## 7. Event-Risk Object

Inside each ward:

```json
"event_risk": {
  "event_definition": "AQI >= 201 within 24h",
  "event_threshold_aqi": 201,
  "high_pollution_probability_24h": 0.705,
  "high_pollution_event_predicted": true
}
```

Person 2 agent use:

This should trigger hotspot investigation.

Suggested logic:

```text
If high_pollution_probability_24h is high:
    Supervisor Agent routes ward to Attribution Agent and Intervention Agent.
```

Do not treat the event model as final proof. Treat it as an early warning signal.

---

## 8. Source Attribution Object

Inside each ward:

```json
"source_attribution": {
  "dominant_sources": [
    {
      "source": "traffic emissions",
      "source_key": "traffic",
      "probability": 0.64,
      "evidence": [
        "high learned traffic proxy",
        "rush-hour temporal pattern",
        "elevated NO2 signal associated with combustion traffic"
      ]
    }
  ],
  "attribution_confidence": "medium-high",
  "causal_warning": "Probabilistic attribution, not direct causal proof."
}
```

Person 2 dashboard use:

- dominant-source bar chart
- evidence cards
- confidence badge
- causal-warning note

Person 2 agent use:

The Intervention Agent should use the top source probabilities to choose candidate interventions.

---

## 9. Priority Object

Inside each ward:

```json
"priority": {
  "priority_score": 0.6021,
  "priority_band": "High",
  "priority_reason": "Computed from forecast AQI, high-pollution event probability, model confidence, and placeholder population vulnerability."
}
```

Person 2 dashboard use:

- intervention queue ordering
- hotspot ranking
- priority badge

Important limitation:

The current priority score uses placeholder vulnerability. Later, Person 2 or Person 1 should replace this with actual school/hospital/population vulnerability data.

---

## 10. Other Sample Files Available

Person 2 can also use these if needed:

```text
backend/data/sample/forecast_output_model.json
backend/data/sample/event_risk_output.json
backend/data/sample/attribution_output_model.json
backend/data/sample/model_explanations.json
backend/data/sample/ml_summary.json
backend/data/sample/forecast_model_metrics.json
backend/data/sample/event_risk_metrics.json
backend/data/sample/source_attribution_metrics.json
```

The recommended primary file is:

```text
backend/data/sample/airguard_intelligence_output.json
```

---

## 11. How to Reproduce Person 1 Pipeline

From repo root:

```bash
source .venv/bin/activate
```

Then run:

```bash
python -m ml.data_processing.create_sample_dataset
python -m ml.forecasting.baseline_persistence
python -m ml.forecasting.train_forecast_model
python -m ml.forecasting.train_event_risk_model
python -m ml.attribution.train_source_attribution_model
python -m scripts.export_sample_outputs
python -m scripts.build_airguard_intelligence_output
python -m ml.explainability.explain_forecast_model
python -m ml.evaluation.model_report
```

This regenerates the dataset, models, metrics, sample outputs, explanations, and model report.

---

## 12. Technical Claims Available for Demo

Current demo-ready claims:

1. The learned AQI forecasting model improves AQI RMSE over the persistence baseline.
2. A separate event-risk classifier is used because average AQI regression alone can miss rare dangerous spikes.
3. The event-risk model provides high-pollution probability at ward level.
4. The source-attribution model estimates probable pollution drivers across six source categories.
5. Forecast and attribution outputs include confidence/evidence fields.
6. All ML outputs are structured in JSON for agentic backend integration.
7. The system explicitly avoids causal overclaiming by labelling attribution as probabilistic.

---

## 13. Current Limitations

1. Current dataset is synthetic and mechanism-driven.
2. Real CPCB, weather, satellite, traffic, and land-use ingestion is not yet implemented.
3. Source attribution is probabilistic, not causal proof.
4. Event-risk precision is limited, so the Evidence Verifier and human approval layer are required.
5. Priority score currently uses placeholder vulnerability.
6. Meteorology is difficult to classify as a standalone source because it acts as a dispersion driver rather than a direct emission source.

---

## 14. What Person 2 Should Build Next

Person 2 should start from:

```text
backend/data/sample/airguard_intelligence_output.json
```

Recommended backend endpoints:

```text
GET /api/intelligence
GET /api/wards
GET /api/wards/{ward_id}
GET /api/agent/run
GET /api/reports/decision-memo
```

Recommended dashboard screens:

1. City Risk Map
2. Agent Run Trace
3. Hotspot Drilldown
4. Source Attribution Panel
5. Intervention Queue
6. Citizen Advisory Generator
7. Decision Memo Page

Recommended agent flow:

```text
Supervisor Agent
    ↓
Forecast Review Tool
    ↓
Event-Risk Screening Tool
    ↓
Source Attribution Tool
    ↓
Intervention Optimisation Agent
    ↓
Evidence Verification Agent
    ↓
Citizen Advisory Agent
    ↓
Decision Memo Agent
```

---

## 15. Final Notes

Person 1 output should be treated as the intelligence layer.

Person 2 should not hard-code fake AQI/source values in the dashboard. The dashboard and agent layer should consume the JSON outputs produced by Person 1.

The final product should show:

```text
Forecast → Event Risk → Source Attribution → Intervention → Evidence → Advisory → Decision Memo
```

This is the core AirGuard AI intelligence loop.