from typing import Any, Dict, List, Optional


AQI_CATEGORIES = [
    {"category": "Good", "aqi_low": 0, "aqi_high": 50},
    {"category": "Satisfactory", "aqi_low": 51, "aqi_high": 100},
    {"category": "Moderately Polluted", "aqi_low": 101, "aqi_high": 200},
    {"category": "Poor", "aqi_low": 201, "aqi_high": 300},
    {"category": "Very Poor", "aqi_low": 301, "aqi_high": 400},
    {"category": "Severe", "aqi_low": 401, "aqi_high": 500},
]


CPCB_BREAKPOINTS = {
    # Units: µg/m3, 24-hour
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    # Units: µg/m3, 24-hour
    "pm25": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    # Units: µg/m3, 24-hour
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    # Units: µg/m3, 24-hour
    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500),
    ],
    # Units: µg/m3, 8-hour
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
    # Units: mg/m3, 8-hour
    "co": [
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500),
    ],
    # Units: µg/m3, 24-hour
    "nh3": [
        (0, 200, 0, 50),
        (201, 400, 51, 100),
        (401, 800, 101, 200),
        (801, 1200, 201, 300),
        (1201, 1800, 301, 400),
        (1801, 2400, 401, 500),
    ],
    # Units: µg/m3, 24-hour
    "pb": [
        (0, 0.5, 0, 50),
        (0.5, 1.0, 51, 100),
        (1.1, 2.0, 101, 200),
        (2.1, 3.0, 201, 300),
        (3.1, 3.5, 301, 400),
        (3.6, 4.0, 401, 500),
    ],
}


def get_aqi_category(aqi: Optional[float]) -> Optional[str]:
    if aqi is None:
        return None

    for item in AQI_CATEGORIES:
        if item["aqi_low"] <= aqi <= item["aqi_high"]:
            return item["category"]

    if aqi > 500:
        return "Severe"

    return None


def calculate_sub_index(
    concentration: Optional[float],
    pollutant: str,
) -> Optional[float]:
    if concentration is None:
        return None

    try:
        concentration = float(concentration)
    except (TypeError, ValueError):
        return None

    if concentration < 0:
        return None

    breakpoints = CPCB_BREAKPOINTS.get(pollutant.lower())

    if not breakpoints:
        return None

    for bp_low, bp_high, i_low, i_high in breakpoints:
        if bp_low <= concentration <= bp_high:
            if bp_high == bp_low:
                return float(i_high)

            sub_index = (
                ((i_high - i_low) / (bp_high - bp_low))
                * (concentration - bp_low)
                + i_low
            )
            return round(float(sub_index), 2)

    highest_bp = breakpoints[-1]
    if concentration > highest_bp[1]:
        return 500.0

    return None


def calculate_cpcb_aqi(pollutants: Dict[str, Any]) -> Dict[str, Any]:
    sub_indices = {}

    for pollutant, concentration in pollutants.items():
        pollutant_key = pollutant.lower()
        sub_index = calculate_sub_index(concentration, pollutant_key)

        if sub_index is not None:
            sub_indices[pollutant_key] = {
                "concentration": concentration,
                "sub_index": sub_index,
                "category": get_aqi_category(sub_index),
            }

    if not sub_indices:
        return {
            "aqi": None,
            "category": None,
            "dominant_pollutant": None,
            "sub_indices": {},
            "method": "CPCB breakpoint AQI",
            "status": "insufficient_pollutant_data",
        }

    dominant_pollutant = max(
        sub_indices,
        key=lambda key: sub_indices[key]["sub_index"],
    )

    aqi = sub_indices[dominant_pollutant]["sub_index"]

    return {
        "aqi": round(aqi, 2),
        "category": get_aqi_category(aqi),
        "dominant_pollutant": dominant_pollutant,
        "sub_indices": sub_indices,
        "method": "CPCB breakpoint AQI",
        "status": "computed_from_available_pollutants",
    }


def main() -> None:
    sample = {
        "pm25": 14.42,
        "pm10": 56.5,
        "co": 0.7,
        "o3": 8.35,
        "no2": None,
        "so2": None,
    }

    result = calculate_cpcb_aqi(sample)
    print(result)


if __name__ == "__main__":
    main()