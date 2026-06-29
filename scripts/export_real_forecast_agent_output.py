import json

from backend.app.agents.real_forecast_agent import RealForecastAgent
from ml.config import PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "real_forecast_agent_output.json"


def main() -> None:
    agent = RealForecastAgent()
    result = agent.run()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved real forecast agent output to: {OUTPUT_PATH}")
    print("Deployment mode:", result["deployment_mode"])
    print("Selected method:", result["selected_forecast_method"])
    print("Best learned model:", result["best_learned_model"]["name"])


if __name__ == "__main__":
    main()