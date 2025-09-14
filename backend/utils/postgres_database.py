import psycopg2
import psycopg2.extras
import pandas as pd
from typing import List, Dict, Optional, Any
from config.settings import settings
import logging
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLDatabaseManager:
    def __init__(self):
        self.connection_params = {
            'host': settings.POSTGRES_HOST,
            'port': settings.POSTGRES_PORT,
            'database': settings.POSTGRES_DB,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD
        }
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable pgvector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create hotels table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS hotels (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        city TEXT NOT NULL,
                        rating REAL,
                        price_range TEXT,
                        amenities TEXT,
                        description TEXT,
                        condition TEXT,
                        total_reviews TEXT,
                        is_sustainable BOOLEAN DEFAULT FALSE,
                        embedding vector(1536),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create places table
                cursor.execute("""
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
                """)
                
                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create travel_data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS travel_data (
                        id SERIAL PRIMARY KEY,
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
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create cities table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cities (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        country TEXT,
                        latitude REAL,
                        longitude REAL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create co2_emissions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS co2_emissions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER,
                        origin TEXT NOT NULL,
                        destination TEXT NOT NULL,
                        distance_km REAL NOT NULL,
                        travel_mode TEXT NOT NULL,
                        emission_factor REAL NOT NULL,
                        total_emissions REAL NOT NULL,
                        trees_needed REAL,
                        daily_average_percentage REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Create indexes for vector similarity search
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS hotels_embedding_idx 
                    ON hotels USING hnsw (embedding vector_cosine_ops);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS places_embedding_idx 
                    ON places USING hnsw (embedding vector_cosine_ops);
                """)
                
                # Create other useful indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_hotels_city ON hotels(city);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_places_city ON places(city);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_places_sustainable ON places(is_sustainable);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_travel_data_user_id ON travel_data(user_id);")
                
                conn.commit()
                logger.info("✅ PostgreSQL database tables initialized successfully")
                
                # Initialize with default data if tables are empty
                self._initialize_default_data()
                
        except Exception as e:
            logger.error(f"❌ Error initializing PostgreSQL database: {e}")
            raise
    
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
                        "INSERT INTO travel_modes (id, name, category, emission_factor) VALUES (%s, %s, %s, %s)",
                        default_modes
                    )
                    logger.info("✅ Default travel modes initialized")
                
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
                        "INSERT INTO cities (name, country, latitude, longitude) VALUES (%s, %s, %s, %s)",
                        default_cities
                    )
                    logger.info("✅ Default cities initialized")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Error initializing default data: {e}")
            raise
    
    def get_travel_modes(self) -> Dict[str, Dict]:
        """Get all travel modes from the database"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT id, name, category, emission_factor 
                FROM travel_modes 
                WHERE is_active = TRUE 
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
            logger.error(f"Error getting travel modes: {e}")
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
                cities_query = "SELECT name FROM cities WHERE is_active = TRUE ORDER BY name"
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
            logger.error(f"Error getting cities: {e}")
            return settings.FALLBACK_CITIES
    
    def get_city_coordinates(self, city_name: str) -> Optional[tuple]:
        """Get coordinates for a specific city"""
        try:
            with self.get_connection() as conn:
                query = "SELECT latitude, longitude FROM cities WHERE name = %s AND is_active = TRUE"
                cursor = conn.cursor()
                cursor.execute(query, (city_name,))
                result = cursor.fetchone()
                
                if result:
                    return (result[0], result[1])
                return None
        except Exception as e:
            logger.error(f"Error getting city coordinates: {e}")
            return None
    
    def get_hotels_in_city(self, city: str) -> List[Dict]:
        """Get hotels in a specific city"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT name, rating, price_range, amenities, description, condition, 
                       total_reviews, is_sustainable
                FROM hotels 
                WHERE city = %s 
                ORDER BY rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting hotels: {e}")
            return []
    
    def get_places_in_city(self, city: str) -> List[Dict]:
        """Get tourist places in a specific city"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT name, type, significance, google_review_rating, is_sustainable, 
                       sustainability_reason, time_needed_hrs, entrance_fee_inr, 
                       best_time_to_visit, weekly_off, dslr_allowed, google_reviews_lakhs, 
                       establishment_year, zone, state, category, description
                FROM places 
                WHERE city = %s 
                ORDER BY google_review_rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting places: {e}")
            return []
    
    def get_sustainable_places_in_city(self, city: str) -> List[Dict]:
        """Get sustainable tourist places in a specific city"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT name, type, significance, google_review_rating, is_sustainable, 
                       sustainability_reason, time_needed_hrs, entrance_fee_inr, 
                       best_time_to_visit, weekly_off, dslr_allowed, google_reviews_lakhs, 
                       establishment_year, zone, state, category, description
                FROM places 
                WHERE city = %s AND is_sustainable = TRUE
                ORDER BY google_review_rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting sustainable places: {e}")
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
            logger.error(f"Error getting travel suggestions: {e}")
            return {"hotels": [], "places": [], "sustainable_places": [], "source": source, "destination": destination}
    
    def get_hotels_in_city_list(self) -> List[str]:
        """Get list of cities that have hotels"""
        try:
            with self.get_connection() as conn:
                query = "SELECT DISTINCT city FROM hotels WHERE city IS NOT NULL"
                df = pd.read_sql_query(query, conn)
                return sorted(df['city'].tolist())
        except Exception as e:
            logger.error(f"Error getting hotel cities: {e}")
            return []
    
    def get_places_in_city_list(self) -> List[str]:
        """Get list of cities that have places"""
        try:
            with self.get_connection() as conn:
                query = "SELECT DISTINCT city FROM places WHERE city IS NOT NULL"
                df = pd.read_sql_query(query, conn)
                return sorted(df['city'].tolist())
        except Exception as e:
            logger.error(f"Error getting place cities: {e}")
            return []
    
    # Vector embedding methods
    def add_hotel_embedding(self, hotel_id: int, embedding: List[float]) -> bool:
        """Add vector embedding for a hotel"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE hotels SET embedding = %s WHERE id = %s",
                    (embedding, hotel_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding hotel embedding: {e}")
            return False
    
    def add_place_embedding(self, place_id: int, embedding: List[float]) -> bool:
        """Add vector embedding for a place"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE places SET embedding = %s WHERE id = %s",
                    (embedding, place_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding place embedding: {e}")
            return False
    
    def search_similar_hotels(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar hotels using vector similarity"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT id, name, city, rating, price_range, amenities, description,
                       1 - (embedding <=> %s) as similarity
                FROM hotels 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """
                df = pd.read_sql_query(query, conn, params=(query_embedding, query_embedding, limit))
                return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error searching similar hotels: {e}")
            return []
    
    def search_similar_places(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar places using vector similarity"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT id, name, city, type, significance, google_review_rating,
                       is_sustainable, sustainability_reason, description,
                       1 - (embedding <=> %s) as similarity
                FROM places 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """
                df = pd.read_sql_query(query, conn, params=(query_embedding, query_embedding, limit))
                return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error searching similar places: {e}")
            return []

# Global database manager instance
postgres_db_manager = PostgreSQLDatabaseManager()
