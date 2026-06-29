import pandas as pd

from ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


INPUT_PATH = RAW_DATA_DIR / "chennai_openaq_latest_measurements.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "chennai_openaq_latest_clean.csv"


CORE_PARAMETERS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. "
            "Run: python -m ml.data_ingestion.fetch_openaq_measurements --city Chennai"
        )

    df = pd.read_csv(INPUT_PATH)

    if "usable_for_realtime" not in df.columns:
        raise ValueError(
            "Missing usable_for_realtime column. "
            "Update and rerun fetch_openaq_measurements.py first."
        )

    usable = df[df["usable_for_realtime"] == True].copy()
    usable = usable[usable["parameter"].isin(CORE_PARAMETERS)].copy()

    if usable.empty:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        usable.to_csv(OUTPUT_PATH, index=False)
        print("No usable real-time rows available.")
        print(f"Saved empty cleaned file to: {OUTPUT_PATH}")
        return

    station_meta = (
        usable.groupby("location_id")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            latest_datetime_utc=("datetime_utc", "max"),
            max_age_hours=("age_hours", "max"),
        )
        .reset_index()
    )

    pivot = (
        usable.pivot_table(
            index="location_id",
            columns="parameter",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )

    clean = station_meta.merge(pivot, on="location_id", how="left")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cleaned OpenAQ latest data to: {OUTPUT_PATH}")
    print(f"Shape: {clean.shape}")
    print("\nColumns:")
    print(clean.columns.tolist())
    print("\nPreview:")
    print(clean.head())


if __name__ == "__main__":
    main()