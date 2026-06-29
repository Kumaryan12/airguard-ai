import json

from backend.app.agents.supervisor_agent import SupervisorAgent
from ml.config import PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "supervisor_agent_output.json"


def main() -> None:
    agent = SupervisorAgent()
    result = agent.run()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved supervisor agent output to: {OUTPUT_PATH}")
    print("Headline:", result["decision_memo"]["headline"])
    print("Monitoring priority:", result["decision"]["monitoring_priority"])
    print("Intervention required now:", result["decision"]["intervention_required_now"])


if __name__ == "__main__":
    main()