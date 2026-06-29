import json
from typing import Any, Dict, Optional
from backend.app.tools.remote_sensing_evidence_tool import RemoteSensingEvidenceTool
from pydantic import BaseModel, Field, ValidationError

from backend.app.llm.groq_client import get_groq_client, get_groq_model
from backend.app.tools.forecast_validation_tool import ForecastValidationTool
from backend.app.tools.evidence_guardrail_tool import EvidenceGuardrailTool
from backend.app.tools.intervention_ranking_tool import InterventionRankingTool
from ml.config import PROJECT_ROOT
from backend.app.tools.wind_sector_evidence_tool import WindSectorEvidenceTool
from backend.app.tools.cpcb_aqi_tool import CpcbAqiTool

OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "groq_supervisor_agent_output.json"


class GroqSupervisorDecision(BaseModel):
    intervention_required_now: bool
    monitoring_priority: str
    decision_headline: str
    reasoning_summary: str
    selected_forecast_method: str
    key_evidence_used: list[str]
    recommended_actions: list[dict]
    safe_claims: list[str]
    claims_to_avoid: list[str]
    human_review_required: bool
    limitations: list[str]


class GroqSupervisorAgent:
    """
    Groq-powered supervisor agent.

    This is an actual LLM reasoning layer over deterministic evidence tools.
    Tools calculate. Groq reasons and synthesizes.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.client = get_groq_client()
        self.model = model or get_groq_model()

    def collect_tool_outputs(self) -> Dict[str, Any]:
        forecast = ForecastValidationTool().run()
        evidence = EvidenceGuardrailTool().run()
        interventions = InterventionRankingTool().run()
        remote_sensing = RemoteSensingEvidenceTool().run()
        cpcb_aqi = CpcbAqiTool().run()
        wind_sector = WindSectorEvidenceTool().run()

        return {
            "forecast_validation_tool": forecast,
            "evidence_guardrail_tool": evidence,
            "intervention_ranking_tool": interventions,
            "remote_sensing_evidence_tool": remote_sensing,
            "cpcb_aqi": cpcb_aqi,
            "wind_sector": wind_sector,
        }

    def _build_prompt(self, tool_outputs: Dict[str, Any]) -> list[dict]:
        system_prompt = """
You are AirGuard AI's Groq-powered Supervisor Agent for urban air-quality intervention.

You must only use the provided tool outputs.
You must not invent station readings, model metrics, sources, or enforcement facts.
You must distinguish evidence-backed hypotheses from causal proof.
You must not claim that the learned ML model is best if the forecast validation tool says a baseline is best.
You must recommend human review before enforcement.

You must not describe nearby industrial POIs as confirmed pollution sources.
You may only say industrial influence is a low-confidence hypothesis requiring verification.
For source attribution, use the phrase "plausible hypothesis" unless causal proof exists.
Do not put industrial influence in safe_claims unless it includes a verification caveat.

Use satellite NO2 only as regional combustion context.
Do not claim satellite NO2 proves ground-level AQI or exact source attribution.
If remote sensing is unavailable, continue using ground/geospatial evidence and mention the limitation. In key_evidence_used, do not use generic labels like "geospatial evidence" or "satellite context".
Always include concrete values from the tools where available, such as AQI value, category, RMSE improvement, road density, PM10/PM2.5 ratio, Sentinel-5P image count, and relative NO2 signal.
Recommended actions must preserve the evidence values from the intervention tool where available.
Use clean professional writing with proper spacing.
Do not concatenate words.
When referencing remote sensing, include the Sentinel-5P image count if available.
Use the exact satellite signal label from the tool, for example "moderate_relative_no2".
Recommended actions must preserve concrete evidence values where available.
When describing forecast method selection, always qualify it as "in the current one-station real-data benchmark" unless broader validation exists.
Use wind-sector evidence when available.

If wind-sector evidence says wind speed class is low, mention low-wind / poor-dispersion / local accumulation risk when relevant.

If wind-sector evidence increases confidence for road dust or traffic corridor exposure, include that in reasoning_summary and key_evidence_used.

Do not claim exact upwind source attribution unless exact source-coordinate geometry exists. Phrase it as "wind-sector screening supports" or "meteorological conditions strengthen the hypothesis."
Prefer "medium-priority preventive action" over "medium-term interventions" for the current scenario.

Safe claims must not sound globally true unless the tool evidence validates them globally.
Return ONLY valid JSON matching this schema:
...


{
  "intervention_required_now": boolean,
  "monitoring_priority": "low" | "medium" | "high" | "data_quality_review",
  "decision_headline": string,
  "reasoning_summary": string,
  "selected_forecast_method": string,
  "key_evidence_used": [
  "Current estimated AQI: 76.275 (Satisfactory)",
  "Forecast method: rolling_mean_24h; RMSE improvement vs persistence: 24.48%",
  "PM10/PM2.5 ratio: 3.92; road density: 26.05 km/km²",
  "Sentinel-5P NO2 signal: moderate_relative_no2 from 269 images"
  key_evidence_used should include concrete values where available:
- CPCB AQI and dominant pollutant
- estimated/model AQI
- forecast RMSE improvement
- PM10/PM2.5 ratio and road density
- Sentinel-5P NO2 signal and image count
- wind direction sector and wind speed class
]
  "recommended_actions": [
    {
      "action": string,
      "priority": string,
      "why": [string],
      "human_review_required": boolean
    }
  ],
  "safe_claims": [string],
  "claims_to_avoid": [string],
  "human_review_required": boolean,
  "limitations": [string]
}
"""

        user_prompt = f"""
Goal:
Assess whether the current Chennai station context requires intervention,
what actions should be recommended, and what should be communicated safely.

Tool outputs:
{json.dumps(tool_outputs, indent=2)}

Make a smart-city command-center decision.
Return only JSON.
"""

        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def run(self) -> Dict[str, Any]:
        tool_outputs = self.collect_tool_outputs()
        messages = self._build_prompt(tool_outputs)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content

        try:
            raw_decision = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Groq returned invalid JSON: {exc}\nRaw output:\n{content}"
            )

        try:
            decision = GroqSupervisorDecision(**raw_decision)
        except ValidationError as exc:
            raise RuntimeError(
                f"Groq JSON did not match expected schema: {exc}\nRaw output:\n{content}"
            )

        return {
            "agent_name": "Groq Supervisor Agent",
            "agent_type": "llm_tool_orchestrating_reasoning_agent",
            "llm_provider": "Groq",
            "model": self.model,
            "tool_outputs": tool_outputs,
            "decision": decision.model_dump(),
        }


def main() -> None:
    agent = GroqSupervisorAgent()
    result = agent.run()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved Groq supervisor output to: {OUTPUT_PATH}")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()