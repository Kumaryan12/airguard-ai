import math
from typing import Dict, Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from ml.config import PROCESSED_DATA_DIR


INPUT_PATH = PROCESSED_DATA_DIR / "chennai_real_latest_features.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "chennai_osm_station_features.csv"


SEARCH_RADIUS_METERS = 1500


MAJOR_ROAD_TYPES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
}


def estimate_utm_crs(latitude: float, longitude: float) -> str:
    zone = int((longitude + 180) / 6) + 1
    hemisphere = "326" if latitude >= 0 else "327"
    return f"EPSG:{hemisphere}{zone:02d}"


def get_area_km2(radius_meters: float) -> float:
    return math.pi * (radius_meters / 1000) ** 2


def get_road_features(latitude: float, longitude: float, radius_meters: int) -> Dict:
    graph = ox.graph_from_point(
        (latitude, longitude),
        dist=radius_meters,
        network_type="drive",
        simplify=True,
    )

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)

    if edges.empty:
        return {
            "total_road_length_km": 0.0,
            "major_road_length_km": 0.0,
            "road_density_km_per_km2": 0.0,
            "major_road_density_km_per_km2": 0.0,
            "nearest_major_road_m": None,
        }

    utm_crs = estimate_utm_crs(latitude, longitude)
    edges_m = edges.to_crs(utm_crs)

    edges_m["length_m_calc"] = edges_m.geometry.length

    total_road_length_km = edges_m["length_m_calc"].sum() / 1000

    def is_major_road(value) -> bool:
        if isinstance(value, list):
            return any(v in MAJOR_ROAD_TYPES for v in value)
        return value in MAJOR_ROAD_TYPES

    edges_m["is_major"] = edges_m["highway"].apply(is_major_road)
    major_edges = edges_m[edges_m["is_major"]].copy()

    major_road_length_km = major_edges["length_m_calc"].sum() / 1000

    area_km2 = get_area_km2(radius_meters)

    point = gpd.GeoSeries(
        [Point(longitude, latitude)],
        crs="EPSG:4326",
    ).to_crs(utm_crs).iloc[0]

    nearest_major_road_m = None
    if not major_edges.empty:
        nearest_major_road_m = float(major_edges.geometry.distance(point).min())

    return {
        "total_road_length_km": round(float(total_road_length_km), 4),
        "major_road_length_km": round(float(major_road_length_km), 4),
        "road_density_km_per_km2": round(float(total_road_length_km / area_km2), 4),
        "major_road_density_km_per_km2": round(float(major_road_length_km / area_km2), 4),
        "nearest_major_road_m": round(nearest_major_road_m, 2)
        if nearest_major_road_m is not None
        else None,
    }


def get_poi_count(latitude: float, longitude: float, radius_meters: int, tags: Dict) -> int:
    try:
        pois = ox.features_from_point(
            (latitude, longitude),
            tags=tags,
            dist=radius_meters,
        )

        if pois.empty:
            return 0

        return int(len(pois))

    except Exception:
        return 0


def get_poi_features(latitude: float, longitude: float, radius_meters: int) -> Dict:
    industrial_count = get_poi_count(
        latitude,
        longitude,
        radius_meters,
        tags={
            "landuse": ["industrial"],
            "man_made": ["works"],
        },
    )

    construction_count = get_poi_count(
        latitude,
        longitude,
        radius_meters,
        tags={
            "landuse": ["construction"],
            "building": ["construction"],
        },
    )

    green_count = get_poi_count(
        latitude,
        longitude,
        radius_meters,
        tags={
            "leisure": ["park", "garden"],
            "landuse": ["grass", "forest", "recreation_ground"],
            "natural": ["wood"],
        },
    )

    school_count = get_poi_count(
        latitude,
        longitude,
        radius_meters,
        tags={
            "amenity": ["school", "college", "university", "kindergarten"],
        },
    )

    hospital_count = get_poi_count(
        latitude,
        longitude,
        radius_meters,
        tags={
            "amenity": ["hospital", "clinic"],
        },
    )

    return {
        "industrial_poi_count": industrial_count,
        "construction_poi_count": construction_count,
        "green_poi_count": green_count,
        "school_poi_count": school_count,
        "hospital_poi_count": hospital_count,
        "vulnerability_poi_count": school_count + hospital_count,
    }


def build_osm_features_for_station(row: pd.Series) -> Dict:
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])

    road_features = get_road_features(
        latitude=latitude,
        longitude=longitude,
        radius_meters=SEARCH_RADIUS_METERS,
    )

    poi_features = get_poi_features(
        latitude=latitude,
        longitude=longitude,
        radius_meters=SEARCH_RADIUS_METERS,
    )

    features = {
        "location_id": row["location_id"],
        "latitude": latitude,
        "longitude": longitude,
        "search_radius_meters": SEARCH_RADIUS_METERS,
        **road_features,
        **poi_features,
        "geospatial_source": "OpenStreetMap via OSMnx",
    }

    return features


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. "
            "Run: python -m ml.data_ingestion.build_real_latest_features"
        )

    station_df = pd.read_csv(INPUT_PATH)

    if station_df.empty:
        print("No real latest feature rows available.")
        station_df.to_csv(OUTPUT_PATH, index=False)
        return

    rows = []

    for _, row in station_df.iterrows():
        print(
            f"Fetching OSM features for location_id={row['location_id']} "
            f"lat={row['latitude']} lon={row['longitude']}"
        )
        features = build_osm_features_for_station(row)
        rows.append(features)

    osm_features = pd.DataFrame(rows)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    osm_features.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved OSM geospatial features to: {OUTPUT_PATH}")
    print(f"Shape: {osm_features.shape}")
    print("\nPreview:")
    print(osm_features.T)


if __name__ == "__main__":
    main()