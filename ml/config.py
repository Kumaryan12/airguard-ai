from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DIR = PROJECT_ROOT / "ml"
DATA_DIR = ML_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONTRACTS_DIR = DATA_DIR / "contracts"
ARTIFACTS_DIR = ML_DIR / "artifacts"

CITY_NAME = "Chennai"

TARGET_COLUMNS = [
    "pm25",
    "pm10",
    "no2",
    "aqi",
]

FORECAST_HORIZON_HOURS = 24

RANDOM_STATE = 42