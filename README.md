# AirGuard AI

AirGuard AI is a multi-agent urban air quality intelligence platform for smart city intervention.

It predicts ward-level pollution risk, estimates probable source drivers, recommends evidence-backed interventions, generates citizen advisories, and stores intervention outcomes for continuous learning.

## Problem Statement

AI-Powered Urban Air Quality Intelligence for Smart City Intervention

The system fuses monitoring station data, satellite imagery, mobility feeds, meteorological forecasts, and geospatial land-use layers to move from reactive air quality monitoring to proactive, evidence-based intervention.

## Core Capabilities

1. Hyperlocal AQI Forecasting
2. Probabilistic Pollution Source Attribution
3. Intervention Optimisation
4. Citizen Health Advisory
5. Evidence Verification
6. Agentic Decision Memo
7. Learning Memory from Past Events

## Team Roles

### Person 1 — AI/ML + Geospatial Intelligence Owner
Owns:
- AQI/weather/geospatial data pipeline
- feature engineering
- forecasting model
- source attribution model
- model evaluation
- model output contracts

### Person 2 — Agentic Platform + Product Experience Owner
Owns:
- backend APIs
- multi-agent orchestration
- intervention optimisation agent
- evidence verifier
- dashboard
- decision memo generation
- demo flow

## Tech Stack

- Python
- FastAPI
- PostgreSQL/PostGIS
- GeoPandas
- LightGBM/CatBoost
- SHAP
- LangGraph or OpenAI Agents SDK
- Next.js
- Leaflet/Mapbox