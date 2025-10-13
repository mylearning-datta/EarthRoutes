#!/usr/bin/env python3
"""
Canonical script to create tables and load data from CSV into PostgreSQL.
- Creates schema (with pgvector) for hotels, places, cities
- Loads data from data/hotel_details.csv and data/Top Indian Places to Visit.csv
- Performs basic cleaning and sustainability classification
"""

import psycopg2
import pandas as pd
import sys
from pathlib import Path

# Ensure backend package imports work when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config.settings import settings  # type: ignore


def get_postgres_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


def create_database_schema(cursor):
    """Create the database schema for hotels, places, and cities tables."""
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS hotels (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            city TEXT,
            description TEXT,
            condition TEXT,
            rating REAL,
            price_range TEXT,
            amenities TEXT,
            is_sustainable BOOLEAN DEFAULT FALSE,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS places (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            category TEXT,
            description TEXT,
            rating REAL,
            zone TEXT,
            state TEXT,
            type TEXT,
            establishment_year INTEGER,
            time_needed_hrs REAL,
            google_review_rating REAL,
            entrance_fee_inr INTEGER,
            airport_within_50km TEXT,
            weekly_off TEXT,
            significance TEXT,
            dslr_allowed TEXT,
            google_reviews_lakhs REAL,
            best_time_to_visit TEXT,
            is_sustainable BOOLEAN DEFAULT FALSE,
            sustainability_reason TEXT,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cities (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            country TEXT,
            latitude REAL,
            longitude REAL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS hotels_embedding_idx 
        ON hotels USING hnsw (embedding vector_cosine_ops);
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS places_embedding_idx 
        ON places USING hnsw (embedding vector_cosine_ops);
        """
    )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hotels_city ON hotels(city);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_places_city ON places(city);")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_places_sustainable ON places(is_sustainable);"
    )


def clear_existing_data(cursor):
    """Clear existing data from target tables (idempotent reloads)."""
    cursor.execute("DELETE FROM hotels")
    cursor.execute("DELETE FROM places")
    cursor.execute("DELETE FROM cities")


def classify_review_score(row):
    """Classify review score based on rating when original condition is 'Review score'."""
    try:
        rating = float(row["Rating"])
        condition = str(row["Condition"]).lower()
        if "review score" in condition:
            if 5 <= rating <= 7:
                return "average"
            elif rating < 5:
                return "poor"
            else:
                return "excellent"
        return row["Condition"]
    except Exception:
        return row.get("Condition")


def detect_sustainability(row):
    """Detect sustainable hotels from description text."""
    try:
        description = str(row.get("description", "")).lower()
        return 1 if "travel sustainable property" in description else 0
    except Exception:
        return 0


def clean_hotel_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare hotel data for insertion."""
    df["Rating"] = df["Rating"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
    df["Total Reviews"] = (
        df["Total Reviews"].astype(str).str.replace(",", "").str.extract(r"(\d+)")
    )
    df = df.replace("", None)
    df = df.dropna(subset=["Condition", "Rating", "Total Reviews"]).copy()
    df["Condition"] = df.apply(classify_review_score, axis=1)
    df["is_sustainable"] = df.apply(detect_sustainability, axis=1).astype(bool)
    return df


def clean_places_data(df: pd.DataFrame, sustainability_map: dict) -> pd.DataFrame:
    """Clean and prepare places data with sustainability classification."""
    numeric_columns = [
        "establishment_year",
        "time_needed_hrs",
        "google_review_rating",
        "entrance_fee_inr",
        "google_reviews_lakhs",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_columns = [
        "zone",
        "state",
        "city",
        "name",
        "type",
        "airport_within_50km",
        "weekly_off",
        "significance",
        "dslr_allowed",
        "best_time_to_visit",
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].replace("", None)

    df["is_sustainable"] = False
    df["sustainability_reason"] = "Unknown place type"
    for idx, row in df.iterrows():
        place_type = row.get("Type")
        if place_type and place_type in sustainability_map:
            is_sustainable, reason = sustainability_map[place_type]
            df.at[idx, "is_sustainable"] = bool(is_sustainable)
            df.at[idx, "sustainability_reason"] = reason
        else:
            df.at[idx, "sustainability_reason"] = (
                f'Place type "{place_type}" not found in mapping'
            )
    return df


def extract_cities_from_places(df: pd.DataFrame) -> list[dict]:
    df_with_cities = df.dropna(subset=["City"]) if "City" in df.columns else df
    cities_data = []
    for _, row in df_with_cities.iterrows():
        city_name = row.get("City")
        state = row.get("State", "India")
        lat, lon = get_city_coordinates(city_name, state)
        cities_data.append(
            {
                "name": city_name,
                "country": "India",
                "latitude": lat,
                "longitude": lon,
            }
        )
    unique = {}
    for c in cities_data:
        if c["name"] not in unique:
            unique[c["name"]] = c
    return list(unique.values())


def load_sustainability_mapping() -> dict:
    csv_path = PROJECT_ROOT / "data" / "place_type_sustainability_with_reasons.csv"
    if not csv_path.exists():
        print(f"Warning: Sustainability mapping file not found at {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        place_type = row["Place Type"]
        is_sustainable = row["is_sustainable"]
        reason = row["Reason"]
        if is_sustainable == "Yes":
            mapping[place_type] = (1, reason)
        elif is_sustainable == "No":
            mapping[place_type] = (0, reason)
        else:
            mapping[place_type] = (0, f"Conditional: {reason}")
    return mapping


def get_city_coordinates(city_name: str, state: str) -> tuple[float, float]:
    # Placeholder coordinates for common Indian cities; falls back to India center
    known = {
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Bangalore": (12.9716, 77.5946),
        "Hyderabad": (17.3850, 78.4867),
        "Kolkata": (22.5726, 88.3639),
        "Chennai": (13.0827, 80.2707),
        "Pune": (18.5204, 73.8567),
        "Ahmedabad": (23.0225, 72.5714),
        "Jaipur": (26.9124, 75.7873),
        "Surat": (21.1702, 72.8311),
    }
    if city_name in known:
        return known[city_name]
    for k, v in known.items():
        if city_name and (city_name.lower() in k.lower() or k.lower() in city_name.lower()):
            return v
    return (20.5937, 78.9629)


def load_hotels_data(cursor, csv_path: Path):
    print(f"Loading hotel data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = clean_hotel_data(df)
    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO hotels (name, city, description, condition, rating, amenities, is_sustainable)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["Hotel Name"],
                row["Place"],
                row.get("description"),
                row.get("Condition"),
                row.get("Rating"),
                row.get("Total Reviews"),
                bool(row.get("is_sustainable", False)),
            ),
        )


def load_places_data(cursor, csv_path: Path, sustainability_map: dict):
    print(f"Loading places data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = clean_places_data(df, sustainability_map)
    cities_list = extract_cities_from_places(df)
    load_cities_data(cursor, cities_list)

    for _, row in df.iterrows():
        establishment_year = row.get("Establishment Year")
        if establishment_year == "Unknown" or not str(establishment_year).isdigit():
            establishment_year = None
        cursor.execute(
            """
            INSERT INTO places (zone, state, city, name, type, establishment_year,
                                time_needed_hrs, google_review_rating, entrance_fee_inr,
                                airport_within_50km, weekly_off, significance, dslr_allowed,
                                google_reviews_lakhs, best_time_to_visit, is_sustainable, sustainability_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row.get("Zone"),
                row.get("State"),
                row.get("City"),
                row.get("Name"),
                row.get("Type"),
                establishment_year,
                row.get("time needed to visit in hrs"),
                row.get("Google review rating"),
                row.get("Entrance Fee in INR"),
                row.get("Airport with 50km Radius"),
                row.get("Weekly Off"),
                row.get("Significance"),
                row.get("DSLR Allowed"),
                row.get("Number of google review in lakhs"),
                row.get("Best Time to visit"),
                bool(row.get("is_sustainable", False)),
                row.get("sustainability_reason"),
            ),
        )


def load_cities_data(cursor, cities_list: list[dict]):
    cursor.execute("DELETE FROM cities")
    for city in cities_list:
        cursor.execute(
            """
            INSERT INTO cities (name, country, latitude, longitude)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO NOTHING
            """,
            (
                city["name"],
                city.get("country", "India"),
                city.get("latitude"),
                city.get("longitude"),
            ),
        )


def main():
    data_dir = PROJECT_ROOT / "data"
    hotel_csv = data_dir / "hotel_details.csv"
    places_csv = data_dir / "Top Indian Places to Visit.csv"

    if not hotel_csv.exists():
        print(f"Error: Hotel CSV file not found at {hotel_csv}")
        sys.exit(1)
    if not places_csv.exists():
        print(f"Error: Places CSV file not found at {places_csv}")
        sys.exit(1)

    print(f"Connecting to PostgreSQL database: {settings.POSTGRES_DB}")
    conn = get_postgres_connection()
    cursor = conn.cursor()

    try:
        create_database_schema(cursor)
        clear_existing_data(cursor)
        sustainability_map = load_sustainability_mapping()
        load_hotels_data(cursor, hotel_csv)
        load_places_data(cursor, places_csv, sustainability_map)
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM hotels")
        hotels_n = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM places")
        places_n = cursor.fetchone()[0]
        print("\nAll data loaded successfully!")
        print(f"- Hotels: {hotels_n} records")
        print(f"- Places: {places_n} records")
        print(f"- PostgreSQL database: {settings.POSTGRES_DB}")
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    main()
