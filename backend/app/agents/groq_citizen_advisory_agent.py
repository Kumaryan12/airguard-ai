import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError, field_validator

from backend.app.llm.groq_client import get_groq_client, get_groq_model
from ml.config import PROJECT_ROOT


SUPERVISOR_OUTPUT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "groq_supervisor_agent_output.json"
)

REAL_SNAPSHOT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "real_geospatial_snapshot.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "sample" / "citizen_advisory_agent_output.json"
)


class CitizenAdvisoryOutput(BaseModel):
    advisory_level: str
    panic_level: str
    english_advisory: str
    hindi_advisory: str
    who_should_take_care: list[str]
    recommended_precautions: list[str]
    what_not_to_claim: list[str]
    data_limitations: list[str]

    @field_validator(
        "who_should_take_care",
        "recommended_precautions",
        "what_not_to_claim",
        "data_limitations",
        mode="before",
    )
    @classmethod
    def coerce_string_to_list(cls, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
        return value


class GroqCitizenAdvisoryAgent:
    """
    Groq-powered citizen advisory agent.

    LLM-first design:
    - Groq generates English + Hindi advisory.
    - Deterministic safety repair layer checks weak/unsafe output.
    - Fallbacks are used only when needed.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.client = get_groq_client()
        self.model = model or get_groq_model()

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def collect_inputs(self) -> Dict[str, Any]:
        supervisor = self._load_json(SUPERVISOR_OUTPUT_PATH)

        snapshot = {}
        if REAL_SNAPSHOT_PATH.exists():
            snapshot = self._load_json(REAL_SNAPSHOT_PATH)

        return {
            "groq_supervisor_output": supervisor,
            "real_station_snapshot": snapshot,
        }

    def _build_compact_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        supervisor = inputs["groq_supervisor_output"]
        snapshot = inputs.get("real_station_snapshot", {})

        decision = supervisor.get("decision", {})
        stations = snapshot.get("stations", [])
        station = stations[0] if stations else {}

        geospatial_features = station.get("geospatial_features", {}) if station else {}
        aqi_estimate = station.get("aqi_estimate", {}) if station else {}
        dispersion = station.get("dispersion", {}) if station else {}

        return {
            "current_estimated_aqi": aqi_estimate.get("estimated_aqi"),
            "current_estimated_aqi_category": aqi_estimate.get("estimated_aqi_category"),
            "aqi_note": aqi_estimate.get(
                "note",
                "Approximate AQI estimate; not official CPCB breakpoint AQI yet.",
            ),
            "decision_headline": decision.get("decision_headline"),
            "monitoring_priority": decision.get("monitoring_priority"),
            "intervention_required_now": decision.get("intervention_required_now"),
            "reasoning_summary": decision.get("reasoning_summary"),
            "dispersion_risk": dispersion.get("dispersion_risk"),
            "vulnerability_poi_count": geospatial_features.get("vulnerability_poi_count"),
            "safe_claims": decision.get("safe_claims", []),
            "claims_to_avoid": decision.get("claims_to_avoid", []),
            "limitations": decision.get("limitations", []),
        }

    def _fallback_english_advisory(self) -> str:
        return (
            "Current air quality near the Chennai station area is satisfactory. "
            "There is no need to panic. Because dispersion risk is medium, people with asthma, "
            "heart disease, breathing difficulty, children, elderly people, and outdoor workers "
            "should stay aware and reduce exposure near dusty or high-traffic roads if they feel discomfort."
        )

    def _fallback_hindi_advisory(self) -> str:
        return (
            "चेन्नई स्टेशन क्षेत्र के आसपास वर्तमान वायु गुणवत्ता संतोषजनक है। "
            "घबराने की जरूरत नहीं है। लेकिन फैलाव की स्थिति मध्यम है, इसलिए अस्थमा, "
            "हृदय रोग या सांस की समस्या वाले लोग, बच्चे, बुजुर्ग और बाहर काम करने वाले लोग "
            "यदि असुविधा महसूस करें तो धूल भरी या अधिक ट्रैफिक वाली सड़कों के पास "
            "अनावश्यक बाहरी गतिविधियों को कम करें।"
        )

    def _fix_spacing_artifacts(self, text: str) -> str:
        if not isinstance(text, str):
            return text

        replacements = {
            "issatisfactory": "is satisfactory",
            "neardusty": "near dusty",
            "nearhigh": "near high",
            "high-trafficroads": "high-traffic roads",
            "operationalforecast": "operational forecast",
            "officialCPCB": "official CPCB",
            "officialAQI": "official AQI",
            "ground-levelAQI": "ground-level AQI",
            "airquality": "air quality",
            "mediumpriority": "medium priority",
            "hightraffic": "high traffic",
            "PM10-heavysignal": "PM10-heavy signal",
            "road densityis": "road density is",
            "snapshotincludes": "snapshot includes",
            "supportsregional": "supports regional",
            "canaffect": "can affect",
            "isconfirmed": "is confirmed",
            "currentAQI": "current AQI",
            "meanbaseline": "mean baseline",
            "thecurrent": "the current",
        }

        for bad, good in replacements.items():
            text = text.replace(bad, good)

        return " ".join(text.split())

    def _looks_like_bad_hindi(self, text: str) -> bool:
        if not text or len(text.strip()) < 40:
            return True

        meaningful_words = [
            "वायु",
            "गुणवत्ता",
            "संतोषजनक",
            "घबराने",
            "अस्थमा",
            "हृदय",
            "सांस",
            "बच्चे",
            "बुजुर्ग",
            "ट्रैफिक",
            "धूल",
        ]

        meaningful_hits = sum(word in text for word in meaningful_words)

        if meaningful_hits < 4:
            return True

        bad_patterns = [
            "asdf",
            "lorem",
            "random",
            "अनुवाद अनुवाद",
        ]

        return any(pattern in text.lower() for pattern in bad_patterns)

    def _contains_unsafe_claim(self, text: str) -> bool:
        lower = text.lower()

        unsafe_phrases = [
            "official cpcb",
            "confirmed source",
            "confirmed pollution source",
            "industrial pollution is confirmed",
            "hazardous",
            "severe",
            "emergency",
        ]

        return any(phrase in lower for phrase in unsafe_phrases)

    def _repair_precautions_if_needed(self, precautions: list[str]) -> list[str]:
        repaired = []

        for item in precautions:
            fixed = self._fix_spacing_artifacts(item)

            if "avoid heavy exertion outdoors" in fixed.lower():
                fixed = "Reduce strenuous outdoor activity only if you feel discomfort."

            repaired.append(fixed)

        return repaired

    def _repair_advisory_if_needed(
        self,
        advisory_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        repairs_applied = []

        advisory_payload["english_advisory"] = self._fix_spacing_artifacts(
            advisory_payload.get("english_advisory", "")
        )
        advisory_payload["hindi_advisory"] = self._fix_spacing_artifacts(
            advisory_payload.get("hindi_advisory", "")
        )

        advisory_payload["what_not_to_claim"] = [
            self._fix_spacing_artifacts(item)
            for item in advisory_payload.get("what_not_to_claim", [])
        ]

        advisory_payload["data_limitations"] = [
            self._fix_spacing_artifacts(item)
            for item in advisory_payload.get("data_limitations", [])
        ]

        advisory_payload["recommended_precautions"] = self._repair_precautions_if_needed(
            advisory_payload.get("recommended_precautions", [])
        )

        english = advisory_payload.get("english_advisory", "")
        hindi = advisory_payload.get("hindi_advisory", "")

        advisory_level = advisory_payload.get("advisory_level", "low")
        panic_level = advisory_payload.get("panic_level", "none")

        if advisory_level == "low" and panic_level == "none":
            if len(english.strip()) < 90 or self._contains_unsafe_claim(english):
                advisory_payload["english_advisory"] = self._fallback_english_advisory()
                repairs_applied.append("english_advisory_repaired")

            if self._looks_like_bad_hindi(hindi) or self._contains_unsafe_claim(hindi):
                advisory_payload["hindi_advisory"] = self._fallback_hindi_advisory()
                repairs_applied.append("hindi_advisory_repaired")

            if len(advisory_payload.get("who_should_take_care", [])) <= 2:
                advisory_payload["who_should_take_care"] = [
                    "People with asthma or breathing difficulty",
                    "People with heart disease",
                    "Children",
                    "Elderly people",
                    "Outdoor workers",
                ]
                repairs_applied.append("who_should_take_care_expanded")

            if len(advisory_payload.get("recommended_precautions", [])) <= 2:
                advisory_payload["recommended_precautions"] = [
                    "No need to panic.",
                    "Continue normal activities if you feel comfortable.",
                    "Sensitive groups should reduce exposure near dusty or high-traffic roads if they feel discomfort.",
                    "People with breathing issues should keep prescribed medicines available.",
                    "Follow local air-quality updates.",
                ]
                repairs_applied.append("recommended_precautions_expanded")

        advisory_payload["safety_repair"] = {
            "repairs_applied": repairs_applied,
            "used_llm_output_directly": len(repairs_applied) == 0,
        }

        return advisory_payload

    def run(self) -> Dict[str, Any]:
        inputs = self.collect_inputs()
        compact_context = self._build_compact_context(inputs)

        system_prompt = """
You are AirGuard AI's Citizen Advisory Agent.

Create a short public-safe air-quality advisory from the provided verified context.

Rules:
- Return ONLY valid JSON.
- Keep the response short but useful.
- Do not create panic.
- Do not claim official CPCB AQI if the input says AQI is approximate.
- Do not claim confirmed source attribution.
- Do not mention enforcement details to citizens.
- If AQI is satisfactory, do not advise extreme precautions.
- Hindi advisory must be natural, simple, and meaningful.
- Fields that require arrays must be JSON arrays, not strings.

Required JSON:
{
  "advisory_level": "low" | "moderate" | "high",
  "panic_level": "none" | "low" | "moderate",
  "english_advisory": "short useful string",
  "hindi_advisory": "natural Hindi advisory string",
  "who_should_take_care": ["short string"],
  "recommended_precautions": ["short string"],
  "what_not_to_claim": ["short string"],
  "data_limitations": ["short string"]
}
"""

        user_prompt = f"""
Verified context:
{json.dumps(compact_context, indent=2)}

Create a concise citizen advisory for people near the Chennai station area.

Hindi meaning to preserve:
"The current air quality is satisfactory. There is no need to panic. People with asthma, heart disease, children, elderly people, and outdoor workers should stay aware and reduce exposure near dusty or high-traffic roads if they feel discomfort."

Return only JSON.
"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.1,
            max_completion_tokens=800,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Groq returned invalid JSON: {exc}\nRaw:\n{content}")

        try:
            advisory = CitizenAdvisoryOutput(**raw)
        except ValidationError as exc:
            raise RuntimeError(
                f"Groq JSON did not match expected schema: {exc}\nRaw:\n{content}"
            )

        advisory_payload = advisory.model_dump()
        advisory_payload = self._repair_advisory_if_needed(advisory_payload)

        return {
            "agent_name": "Groq Citizen Advisory Agent",
            "agent_type": "llm_public_health_advisory_agent",
            "llm_provider": "Groq",
            "model": self.model,
            "input_summary": {
                "source": "Groq Supervisor Agent output + real station snapshot",
                "mode": "LLM-generated English/Hindi advisory with deterministic safety repair layer",
            },
            "advisory": advisory_payload,
        }


def main() -> None:
    agent = GroqCitizenAdvisoryAgent()
    result = agent.run()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved citizen advisory output to: {OUTPUT_PATH}")
    print(json.dumps(result["advisory"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()