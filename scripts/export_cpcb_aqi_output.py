import json

from backend.app.tools.cpcb_aqi_tool import CpcbAqiTool
from ml.config import PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "cpcb_aqi_output.json"


def main() -> None:
    tool = CpcbAqiTool()
    result = tool.run()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    station = result["stations"][0] if result.get("stations") else {}
    aqi = station.get("cpcb_aqi", {})

    print(f"Saved CPCB AQI output to: {OUTPUT_PATH}")
    print("AQI:", aqi.get("aqi"))
    print("Category:", aqi.get("category"))
    print("Dominant pollutant:", aqi.get("dominant_pollutant"))


if __name__ == "__main__":
    main()