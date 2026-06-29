import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUPERVISOR_OUTPUT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "supervisor_agent_output.json"
)


class DecisionMemoAgent:
    """
    Decision Memo Agent.

    Role:
    - Reads Supervisor Agent output.
    - Converts tool outputs and decisions into an official-style operational memo.
    - Keeps claims safe, evidence-backed, and human-review friendly.
    """

    def __init__(self, supervisor_output_path: Path = SUPERVISOR_OUTPUT_PATH) -> None:
        self.supervisor_output_path = supervisor_output_path

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _format_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []

        for idx, action in enumerate(actions, start=1):
            formatted.append(
                {
                    "rank": idx,
                    "action": action.get("intervention"),
                    "priority": action.get("priority"),
                    "source_addressed": action.get("source_addressed"),
                    "why": action.get("why", []),
                    "human_review_required": True,
                    "evidence_required_before_enforcement": action.get(
                        "required_evidence_before_enforcement", []
                    ),
                }
            )

        return formatted

    def run(self) -> Dict[str, Any]:
        supervisor = self._load_json(self.supervisor_output_path)

        decision = supervisor["decision"]
        memo = supervisor["decision_memo"]
        forecast = supervisor["tool_outputs"]["forecast_validation"]
        evidence = supervisor["tool_outputs"]["evidence_guardrail"]

        selected_metrics = forecast["selected_method_metrics"]

        official_memo = {
            "agent_name": "Decision Memo Agent",
            "agent_type": "official_decision_synthesis_agent",
            "city": "Chennai",
            "station_location_id": forecast["station_location_id"],
            "headline": memo["headline"],
            "operational_status": {
                "intervention_required_now": decision["intervention_required_now"],
                "monitoring_priority": decision["monitoring_priority"],
                "current_estimated_aqi": decision["current_aqi"],
                "current_estimated_aqi_category": decision["current_aqi_category"],
                "dispersion_risk": decision["dispersion_risk"],
                "data_status": decision["data_status"],
            },
            "forecast_validation": {
                "selected_method": forecast["selected_forecast_method"],
                "selected_method_type": forecast["selected_method_type"],
                "rmse": selected_metrics["rmse"],
                "mae": selected_metrics["mae"],
                "rmse_improvement_vs_persistence_percent": round(
                    selected_metrics["rmse_improvement_vs_persistence"] * 100, 2
                ),
                "best_learned_model": forecast["best_learned_model"]["name"],
                "why_this_method": forecast["rationale"],
            },
            "evidence_summary": {
                "claims_checked": evidence["claims_checked"],
                "supported_claims": evidence["summary"]["supported"],
                "weak_or_not_supported": evidence["summary"]["weak_or_not_supported"],
                "blocked_overclaims": evidence["summary"]["do_not_claim"],
                "safe_claims": memo["safe_claims"],
                "claims_to_avoid": memo["claims_to_avoid"],
            },
            "recommended_actions": self._format_actions(memo["recommended_actions"]),
            "decision_text": (
                f"{memo['headline']}. Current estimated AQI is "
                f"{decision['current_aqi']} ({decision['current_aqi_category']}) with "
                f"{decision['dispersion_risk']} dispersion risk. The system recommends "
                f"{decision['monitoring_priority']} monitoring priority. Enforcement should not be "
                f"initiated solely from this output; field verification and official records are required."
            ),
            "limitations": [
                "AQI is currently approximate, not official CPCB breakpoint AQI.",
                "Source outputs are evidence-backed hypotheses, not causal proof.",
                "Current real validation is based on one OpenAQ station.",
                "Enforcement requires human review and official evidence.",
            ],
        }

        return official_memo


def main() -> None:
    agent = DecisionMemoAgent()
    result = agent.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()