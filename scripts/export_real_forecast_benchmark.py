import json
import shutil

from ml.config import ARTIFACTS_DIR, PROJECT_ROOT


INPUT_PATH = ARTIFACTS_DIR / "real_forecast_benchmark_metrics.json"
OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "real_forecast_benchmark_metrics.json"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing benchmark metrics: {INPUT_PATH}. "
            "Run: python -m ml.forecasting.train_real_forecast_model"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT_PATH, OUTPUT_PATH)

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"Exported real forecast benchmark to: {OUTPUT_PATH}")
    print("Best overall:", metrics["best_overall"])
    print("Best learned model:", metrics["best_learned_model"])
    print("Best baseline:", metrics["best_baseline"])

    best_overall = metrics["results"][metrics["best_overall"]]
    best_learned = metrics["results"][metrics["best_learned_model"]]

    print("\nBest overall RMSE:", round(best_overall["rmse"], 4))
    print("Best overall improvement vs persistence:", round(best_overall["rmse_improvement_vs_persistence"] * 100, 2), "%")
    print("Best learned RMSE:", round(best_learned["rmse"], 4))
    print("Best learned improvement vs persistence:", round(best_learned["rmse_improvement_vs_persistence"] * 100, 2), "%")


if __name__ == "__main__":
    main()