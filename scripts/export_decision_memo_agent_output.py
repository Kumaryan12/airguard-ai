import json

from backend.app.agents.decision_memo_agent import DecisionMemoAgent
from ml.config import PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "decision_memo_agent_output.json"


def main() -> None:
    agent = DecisionMemoAgent()
    result = agent.run()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved decision memo agent output to: {OUTPUT_PATH}")
    print("Headline:", result["headline"])
    print("Monitoring priority:", result["operational_status"]["monitoring_priority"])
    print("Recommended actions:", len(result["recommended_actions"]))


if __name__ == "__main__":
    main()