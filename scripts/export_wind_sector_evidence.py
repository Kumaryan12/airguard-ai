import json

from backend.app.tools.wind_sector_evidence_tool import WindSectorEvidenceTool
from ml.config import PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "wind_sector_evidence.json"


def main() -> None:
    tool = WindSectorEvidenceTool()
    result = tool.run()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    station = result["stations"][0] if result.get("stations") else {}

    print(f"Saved wind sector evidence to: {OUTPUT_PATH}")
    print("Wind from sector:", station.get("interpreted_wind_from_sector"))
    print("Wind speed class:", station.get("wind_speed_class"))


if __name__ == "__main__":
    main()