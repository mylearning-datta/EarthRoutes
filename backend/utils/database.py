import sqlite3
import pandas as pd
from typing import List, Dict, Optional
from config.settings import settings

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create hotels table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS hotels (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        city TEXT NOT NULL,
                        rating REAL,
                        price_range TEXT,
                        amenities TEXT
                    )
                """)
                
                # Create places table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS places (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        city TEXT NOT NULL,
                        category TEXT,
                        description TEXT,
                        rating REAL
                    )
                """)
                
                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create travel_data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS travel_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        source TEXT NOT NULL,
                        destination TEXT NOT NULL,
                        travel_mode TEXT NOT NULL,
                        distance REAL,
                        co2_emissions REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Create travel_modes table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS travel_modes (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        category TEXT NOT NULL,
                        emission_factor REAL NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create cities table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        country TEXT,
                        latitude REAL,
                        longitude REAL,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                print("✅ Database tables initialized successfully")
                
                # Initialize with default data if tables are empty
                self._initialize_default_data()
                
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    def _initialize_default_data(self):
        """Initialize database with default travel modes and cities if empty"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if travel_modes table is empty
                cursor.execute("SELECT COUNT(*) FROM travel_modes")
                if cursor.fetchone()[0] == 0:
                    # Insert default travel modes
                    default_modes = [
                        ("flight", "Flight (average)", "Air Travel", 0.255),
                        ("diesel_car", "Diesel Car", "Road Transport", 0.171),
                        ("petrol_car", "Petrol Car", "Road Transport", 0.192),
                        ("electric_car", "Electric Car", "Road Transport", 0.053),
                        ("train_diesel", "Train (diesel)", "Public Transport", 0.041),
                        ("bus_shared", "Bus (shared)", "Public Transport", 0.089),
                        ("train_electric", "Train (electric)", "Public Transport", 0.041),
                        ("bicycle", "Bicycle", "Active Transport", 0.0),
                        ("walking", "Walking", "Active Transport", 0.0)
                    ]
                    cursor.executemany(
                        "INSERT INTO travel_modes (id, name, category, emission_factor) VALUES (?, ?, ?, ?)",
                        default_modes
                    )
                    print("✅ Default travel modes initialized")
                
                # Check if cities table is empty
                cursor.execute("SELECT COUNT(*) FROM cities")
                if cursor.fetchone()[0] == 0:
                    # Insert default cities
                    default_cities = [
                        ("Delhi", "India", 28.6139, 77.2090),
                        ("Mumbai", "India", 19.0760, 72.8777),
                        ("Bangalore", "India", 12.9716, 77.5946),
                        ("Hyderabad", "India", 17.3850, 78.4867),
                        ("Kolkata", "India", 22.5726, 88.3639),
                        ("Chennai", "India", 13.0827, 80.2707),
                        ("Pune", "India", 18.5204, 73.8567),
                        ("Ahmedabad", "India", 23.0225, 72.5714),
                        ("Jaipur", "India", 26.9124, 75.7873),
                        ("Surat", "India", 21.1702, 72.8311),
                        ("New York", "USA", 40.7128, -74.0060),
                        ("San Francisco", "USA", 37.7749, -122.4194),
                        ("London", "UK", 51.5074, -0.1278),
                        ("Paris", "France", 48.8566, 2.3522),
                        ("Tokyo", "Japan", 35.6762, 139.6503)
                    ]
                    cursor.executemany(
                        "INSERT INTO cities (name, country, latitude, longitude) VALUES (?, ?, ?, ?)",
                        default_cities
                    )
                    print("✅ Default cities initialized")
                
                conn.commit()
                
        except Exception as e:
            print(f"❌ Error initializing default data: {e}")
    
    def get_travel_modes(self) -> Dict[str, Dict]:
        """Get all travel modes from the database"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT id, name, category, emission_factor 
                FROM travel_modes 
                WHERE is_active = 1 
                ORDER BY category, name
                """
                df = pd.read_sql_query(query, conn)
                
                # Convert to dictionary format
                modes = {}
                for _, row in df.iterrows():
                    modes[row['id']] = {
                        'name': row['name'],
                        'category': row['category'],
                        'emission_factor': row['emission_factor']
                    }
                
                return modes
        except Exception as e:
            print(f"Error getting travel modes: {e}")
            # Fallback to hardcoded values
            return {
                "flight": {"name": "Flight (average)", "category": "Air Travel", "emission_factor": 0.255},
                "diesel_car": {"name": "Diesel Car", "category": "Road Transport", "emission_factor": 0.171},
                "petrol_car": {"name": "Petrol Car", "category": "Road Transport", "emission_factor": 0.192},
                "electric_car": {"name": "Electric Car", "category": "Road Transport", "emission_factor": 0.053},
                "train_diesel": {"name": "Train (diesel)", "category": "Public Transport", "emission_factor": 0.041},
                "bus_shared": {"name": "Bus (shared)", "category": "Public Transport", "emission_factor": 0.089},
                "train_electric": {"name": "Train (electric)", "category": "Public Transport", "emission_factor": 0.041},
                "bicycle": {"name": "Bicycle", "category": "Active Transport", "emission_factor": 0.0},
                "walking": {"name": "Walking", "category": "Active Transport", "emission_factor": 0.0}
            }
    
    def get_cities(self) -> List[str]:
        """Get all available cities from the database"""
        try:
            with self.get_connection() as conn:
                # First try to get from cities table
                cities_query = "SELECT name FROM cities WHERE is_active = 1 ORDER BY name"
                cities_df = pd.read_sql_query(cities_query, conn)
                
                if not cities_df.empty:
                    return cities_df['name'].tolist()
                
                # Fallback to hotels and places tables
                hotels_query = "SELECT DISTINCT city FROM hotels WHERE city IS NOT NULL"
                hotels_df = pd.read_sql_query(hotels_query, conn)
                
                places_query = "SELECT DISTINCT city FROM places WHERE city IS NOT NULL"
                places_df = pd.read_sql_query(places_query, conn)
                
                # Combine and deduplicate
                all_cities = set()
                all_cities.update(hotels_df['city'].tolist())
                all_cities.update(places_df['city'].tolist())
                
                return sorted(list(all_cities))
        except Exception as e:
            print(f"Error getting cities: {e}")
            return settings.MAJOR_CITIES
    
    def get_city_coordinates(self, city_name: str) -> Optional[tuple]:
        """Get coordinates for a specific city"""
        try:
            with self.get_connection() as conn:
                query = "SELECT latitude, longitude FROM cities WHERE name = ? AND is_active = 1"
                cursor = conn.cursor()
                cursor.execute(query, (city_name,))
                result = cursor.fetchone()
                
                if result:
                    return (result[0], result[1])
                return None
        except Exception as e:
            print(f"Error getting city coordinates: {e}")
            return None
    
    def get_hotels_in_city(self, city: str) -> List[Dict]:
        """Get hotels in a specific city"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT name, rating, price_range, amenities 
                FROM hotels 
                WHERE city = ? 
                ORDER BY rating DESC 
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                return df.to_dict('records')
        except Exception as e:
            print(f"Error getting hotels: {e}")
            return []
    
    def get_places_in_city(self, city: str) -> List[Dict]:
        """Get tourist places in a specific city"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT name, type, significance, google_review_rating, is_sustainable, sustainability_reason,
                       time_needed_hrs, entrance_fee_inr, best_time_to_visit, weekly_off, dslr_allowed,
                       google_reviews_lakhs, establishment_year
                FROM places 
                WHERE city = ? 
                ORDER BY google_review_rating DESC 
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                return df.to_dict('records')
        except Exception as e:
            print(f"Error getting places: {e}")
            return []
    
    def get_sustainable_places_in_city(self, city: str) -> List[Dict]:
        """Get sustainable tourist places in a specific city"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT name, type, significance, google_review_rating, is_sustainable, sustainability_reason,
                       time_needed_hrs, entrance_fee_inr, best_time_to_visit, weekly_off, dslr_allowed,
                       google_reviews_lakhs, establishment_year
                FROM places 
                WHERE city = ? AND is_sustainable = 1
                ORDER BY google_review_rating DESC 
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                return df.to_dict('records')
        except Exception as e:
            print(f"Error getting sustainable places: {e}")
            return []
    
    def get_travel_suggestions(self, source: str, destination: str) -> Dict:
        """Get travel suggestions between two cities"""
        try:
            with self.get_connection() as conn:
                # Get hotels in destination
                hotels = self.get_hotels_in_city(destination)
                
                # Get all places in destination
                places = self.get_places_in_city(destination)
                
                # Get sustainable places in destination
                sustainable_places = self.get_sustainable_places_in_city(destination)
                
                return {
                    "hotels": hotels,
                    "places": places,
                    "sustainable_places": sustainable_places,
                    "source": source,
                    "destination": destination
                }
        except Exception as e:
            print(f"Error getting travel suggestions: {e}")
            return {"hotels": [], "places": [], "sustainable_places": [], "source": source, "destination": destination}
    
    def get_hotels_in_city_list(self) -> List[str]:
        """Get list of cities that have hotels"""
        try:
            with self.get_connection() as conn:
                query = "SELECT DISTINCT city FROM hotels WHERE city IS NOT NULL"
                df = pd.read_sql_query(query, conn)
                return sorted(df['city'].tolist())
        except Exception as e:
            print(f"Error getting hotel cities: {e}")
            return []
    
    def get_places_in_city_list(self) -> List[str]:
        """Get list of cities that have places"""
        try:
            with self.get_connection() as conn:
                query = "SELECT DISTINCT city FROM places WHERE city IS NOT NULL"
                df = pd.read_sql_query(query, conn)
                return sorted(df['city'].tolist())
        except Exception as e:
            print(f"Error getting place cities: {e}")
            return []

# Global database manager instance
db_manager = DatabaseManager()
