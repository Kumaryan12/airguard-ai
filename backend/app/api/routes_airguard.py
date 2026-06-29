import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.app.agents.groq_supervisor_agent import GroqSupervisorAgent
from ml.config import PROJECT_ROOT


router = APIRouter(prefix="/api/airguard", tags=["airguard"])

DATA_DIR = PROJECT_ROOT / "backend" / "data" / "sample"


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path.name}",
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/health")
def airguard_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "AirGuard AI",
        "message": "AirGuard API is running",
    }


@router.get("/real-snapshot")
def get_real_snapshot() -> Dict[str, Any]:
    return read_json_file(DATA_DIR / "real_geospatial_snapshot.json")


@router.get("/forecast-benchmark")
def get_forecast_benchmark() -> Dict[str, Any]:
    return read_json_file(DATA_DIR / "real_forecast_benchmark_metrics.json")


@router.get("/forecast-validation")
def get_forecast_validation() -> Dict[str, Any]:
    return read_json_file(DATA_DIR / "forecast_validation_tool_output.json")


@router.get("/supervisor")
def get_supervisor_output() -> Dict[str, Any]:
    return read_json_file(DATA_DIR / "supervisor_agent_output.json")


@router.get("/groq-supervisor")
def get_groq_supervisor_output() -> Dict[str, Any]:
    return read_json_file(DATA_DIR / "groq_supervisor_agent_output.json")


@router.post("/run-groq-supervisor")
def run_groq_supervisor() -> Dict[str, Any]:
    try:
        agent = GroqSupervisorAgent()
        result = agent.run()

        output_path = DATA_DIR / "groq_supervisor_agent_output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))