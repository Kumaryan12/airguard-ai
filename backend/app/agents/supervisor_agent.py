import json
from typing import Any, Dict, List

from backend.app.tools.forecast_validation_tool import ForecastValidationTool
from backend.app.tools.evidence_guardrail_tool import EvidenceGuardrailTool
from backend.app.tools.intervention_ranking_tool import InterventionRankingTool


class SupervisorAgent:
    """
    Supervisor Agent.

    This is the first true orchestration agent in AirGuard AI.

    Role:
    - Accepts the goal of assessing current urban air-quality intervention need.
    - Decides which tools are needed.
    - Calls deterministic tools.
    - Synthesizes a decision.
    - Produces a decision memo with caveats and next actions.
    """

    def __init__(self) -> None:
        self.goal = (
            "Assess whether the current Chennai station context requires intervention, "
            "what evidence supports the decision, and what should be communicated safely."
        )

    def _summarize_tool_plan(self) -> List[Dict[str, str]]:
        return [
            {
                "tool": "ForecastValidationTool",
                "reason": "Determine which forecast method is validated on real historical data.",
            },
            {
                "tool": "EvidenceGuardrailTool",
                "reason": "Check which claims are supported and prevent overclaiming.",
            },
            {
                "tool": "InterventionRankingTool",
                "reason": "Map source evidence and vulnerability context to candidate actions.",
            },
        ]

    def _decide_monitoring_priority(
        self,
        forecast_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        intervention_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        station_context = forecast_result.get("latest_station_context") or {}

        aqi = station_context.get("estimated_aqi")
        category = station_context.get("estimated_aqi_category")
        dispersion_risk = station_context.get("dispersion_risk")
        data_status = station_context.get("data_status")

        interventions = []
        if intervention_result.get("stations"):
            interventions = intervention_result["stations"][0].get("ranked_interventions", [])

        top_priority = interventions[0]["priority"] if interventions else "low"

        supported_claims = evidence_result.get("summary", {}).get("supported", 0)
        do_not_claim = evidence_result.get("summary", {}).get("do_not_claim", 0)

        intervention_required_now = False
        monitoring_priority = "low"

        if category in ["Poor", "Very Poor", "Severe"]:
            intervention_required_now = True
            monitoring_priority = "high"
        elif dispersion_risk in ["medium", "high"] and top_priority in ["medium", "high"]:
            intervention_required_now = False
            monitoring_priority = "medium"
        elif data_status != "fresh_realtime_sensor_snapshot":
            intervention_required_now = False
            monitoring_priority = "data_quality_review"
        else:
            intervention_required_now = False
            monitoring_priority = "low"

        return {
            "intervention_required_now": intervention_required_now,
            "monitoring_priority": monitoring_priority,
            "current_aqi": aqi,
            "current_aqi_category": category,
            "dispersion_risk": dispersion_risk,
            "data_status": data_status,
            "top_intervention_priority": top_priority,
            "supported_claims": supported_claims,
            "blocked_overclaims": do_not_claim,
        }

    def run(self) -> Dict[str, Any]:
        tool_plan = self._summarize_tool_plan()

        forecast_tool = ForecastValidationTool()
        evidence_tool = EvidenceGuardrailTool()
        intervention_tool = InterventionRankingTool()

        forecast_result = forecast_tool.run()
        evidence_result = evidence_tool.run()
        intervention_result = intervention_tool.run()

        decision = self._decide_monitoring_priority(
            forecast_result=forecast_result,
            evidence_result=evidence_result,
            intervention_result=intervention_result,
        )

        recommended_actions = []
        if intervention_result.get("stations"):
            recommended_actions = intervention_result["stations"][0].get("ranked_interventions", [])

        memo = {
            "headline": self._make_headline(decision),
            "summary": self._make_summary(decision),
            "recommended_actions": recommended_actions[:3],
            "safe_claims": self._extract_safe_claims(evidence_result),
            "claims_to_avoid": self._extract_claims_to_avoid(evidence_result),
        }

        return {
            "agent_name": "Supervisor Agent",
            "agent_type": "tool_orchestrating_decision_agent",
            "goal": self.goal,
            "tool_plan": tool_plan,
            "tool_outputs": {
                "forecast_validation": forecast_result,
                "evidence_guardrail": evidence_result,
                "intervention_ranking": intervention_result,
            },
            "decision": decision,
            "decision_memo": memo,
        }

    def _make_headline(self, decision: Dict[str, Any]) -> str:
        if decision["intervention_required_now"]:
            return "Immediate intervention recommended"
        if decision["monitoring_priority"] == "medium":
            return "No emergency intervention; medium-priority monitoring and preventive action recommended"
        if decision["monitoring_priority"] == "data_quality_review":
            return "Data quality review required before intervention decision"
        return "No immediate intervention required"

    def _make_summary(self, decision: Dict[str, Any]) -> str:
        return (
            f"Current estimated AQI is {decision['current_aqi']} "
            f"({decision['current_aqi_category']}) with {decision['dispersion_risk']} "
            f"dispersion risk. The system blocked {decision['blocked_overclaims']} "
            f"unsupported/overstrong claims and found {decision['supported_claims']} supported claims. "
            f"Recommended monitoring priority: {decision['monitoring_priority']}."
        )

    def _extract_safe_claims(self, evidence_result: Dict[str, Any]) -> List[str]:
        safe = []

        for claim in evidence_result.get("claims", []):
            if claim.get("status") in ["supported", "partially_supported"]:
                safe.append(claim["claim"])

        return safe

    def _extract_claims_to_avoid(self, evidence_result: Dict[str, Any]) -> List[Dict[str, str]]:
        avoid = []

        for claim in evidence_result.get("claims", []):
            if claim.get("status") == "do_not_claim":
                avoid.append(
                    {
                        "claim": claim["claim"],
                        "safe_rewording": claim.get("safe_rewording", ""),
                    }
                )

        return avoid


def main() -> None:
    agent = SupervisorAgent()
    result = agent.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()