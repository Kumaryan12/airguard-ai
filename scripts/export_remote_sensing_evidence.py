import json
import shutil

from ml.config import PROCESSED_DATA_DIR, PROJECT_ROOT


INPUT_PATH = PROCESSED_DATA_DIR / "chennai_remote_sensing_evidence.json"
OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "sample" / "remote_sensing_evidence.json"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing remote sensing evidence: {INPUT_PATH}. "
            "Run: python -m ml.remote_sensing.fetch_sentinel5p_no2_gee"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT_PATH, OUTPUT_PATH)

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Exported remote sensing evidence to: {OUTPUT_PATH}")
    print("Satellite layer:", data["satellite_layer"])
    print("Relative NO2 signal:", data["relative_no2_signal"])
    print("Image count:", data["collection_image_count"])


if __name__ == "__main__":
    main()