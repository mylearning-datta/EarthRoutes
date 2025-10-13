import psycopg2
import psycopg2.extras
import pandas as pd
from typing import List, Dict, Optional, Any
from config.settings import settings
import logging
from contextlib import contextmanager
import math
import json

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
    
    def _json_safe(self, obj: Any) -> Any:
        """Recursively convert pandas/NumPy NaN/inf and string 'NaN' to None for JSON safety."""
        if isinstance(obj, list):
            return [self._json_safe(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self._json_safe(value) for key, value in obj.items()}
        # Handle floats (including numpy.float types)
        if isinstance(obj, float):
            # math.isfinite returns False for inf, -inf, and NaN
            if not math.isfinite(obj):
                return None
            # pd.isna catches pandas/NumPy NaN as well
            if pd.isna(obj):
                return None
            return obj
        # Handle pandas/NumPy NA types generically
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        # Clean string 'NaN'
        if isinstance(obj, str) and obj.strip().lower() == 'nan':
            return None
        return obj

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
                
                # Create chat_sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        chat_type TEXT NOT NULL DEFAULT 'regular', -- 'regular' or 'finetuned'
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Create chat_messages table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER NOT NULL,
                        message_type TEXT NOT NULL, -- 'user' or 'bot'
                        content TEXT NOT NULL,
                        travel_data JSONB, -- Store structured travel data if available
                        is_error BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
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
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at DESC);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);")
                
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
                WHERE LOWER(city) = LOWER(%s)
                ORDER BY rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                
                # Replace NaN values with None to prevent JSON serialization errors
                df = df.where(pd.notnull(df), None)
                
                # Also replace string 'NaN' values with None
                df = df.replace('NaN', None)
                
                return self._json_safe(df.to_dict('records'))
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
                WHERE LOWER(city) = LOWER(%s)
                ORDER BY google_review_rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                
                # Replace NaN values with None to prevent JSON serialization errors
                df = df.where(pd.notnull(df), None)
                
                # Also replace string 'NaN' values with None
                df = df.replace('NaN', None)
                
                return self._json_safe(df.to_dict('records'))
        except Exception as e:
            logger.error(f"Error getting places: {e}")
            return []
    
    def get_hotels_by_location(self, location: str) -> List[Dict]:
        """Get hotels by trying city, then state, then zone for the provided location name."""
        try:
            with self.get_connection() as conn:
                # 1) Try exact city match
                city_query = """
                SELECT name, rating, price_range, amenities, description, condition, 
                       total_reviews, is_sustainable
                FROM hotels 
                WHERE LOWER(city) = LOWER(%s)
                ORDER BY rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(city_query, conn, params=(location,))
                if not df.empty:
                    df = df.where(pd.notnull(df), None).replace('NaN', None)
                    return self._json_safe(df.to_dict('records'))

                # 2) Try state match (join with places to infer state if hotels table lacks state)
                state_query = """
                SELECT h.name, h.rating, h.price_range, h.amenities, h.description, h.condition,
                       h.total_reviews, h.is_sustainable
                FROM hotels h
                JOIN (
                    SELECT DISTINCT city, state FROM places WHERE LOWER(state) = LOWER(%s)
                ) p ON p.city = h.city
                ORDER BY h.rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(state_query, conn, params=(location,))
                if not df.empty:
                    df = df.where(pd.notnull(df), None).replace('NaN', None)
                    return self._json_safe(df.to_dict('records'))

                # 3) Try zone match
                zone_query = """
                SELECT h.name, h.rating, h.price_range, h.amenities, h.description, h.condition,
                       h.total_reviews, h.is_sustainable
                FROM hotels h
                JOIN (
                    SELECT DISTINCT city, zone FROM places WHERE LOWER(zone) = LOWER(%s)
                ) p ON p.city = h.city
                ORDER BY h.rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(zone_query, conn, params=(location,))
                if not df.empty:
                    df = df.where(pd.notnull(df), None).replace('NaN', None)
                    return self._json_safe(df.to_dict('records'))

                return []
        except Exception as e:
            logger.error(f"Error getting hotels by location: {e}")
            return []

    def get_places_by_location(self, location: str) -> List[Dict]:
        """Get places by trying city, then state, then zone for the provided location name."""
        try:
            with self.get_connection() as conn:
                # 1) Try exact city match
                city_query = """
                SELECT name, type, significance, google_review_rating, is_sustainable, 
                       sustainability_reason, time_needed_hrs, entrance_fee_inr, 
                       best_time_to_visit, weekly_off, dslr_allowed, google_reviews_lakhs, 
                       establishment_year, zone, state, category, description
                FROM places 
                WHERE LOWER(city) = LOWER(%s)
                ORDER BY google_review_rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(city_query, conn, params=(location,))
                if not df.empty:
                    df = df.where(pd.notnull(df), None).replace('NaN', None)
                    return self._json_safe(df.to_dict('records'))

                # 2) Try state match
                state_query = """
                SELECT name, type, significance, google_review_rating, is_sustainable, 
                       sustainability_reason, time_needed_hrs, entrance_fee_inr, 
                       best_time_to_visit, weekly_off, dslr_allowed, google_reviews_lakhs, 
                       establishment_year, zone, state, category, description
                FROM places 
                WHERE LOWER(state) = LOWER(%s)
                ORDER BY google_review_rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(state_query, conn, params=(location,))
                if not df.empty:
                    df = df.where(pd.notnull(df), None).replace('NaN', None)
                    return self._json_safe(df.to_dict('records'))

                # 3) Try zone match
                zone_query = """
                SELECT name, type, significance, google_review_rating, is_sustainable, 
                       sustainability_reason, time_needed_hrs, entrance_fee_inr, 
                       best_time_to_visit, weekly_off, dslr_allowed, google_reviews_lakhs, 
                       establishment_year, zone, state, category, description
                FROM places 
                WHERE LOWER(zone) = LOWER(%s)
                ORDER BY google_review_rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(zone_query, conn, params=(location,))
                if not df.empty:
                    df = df.where(pd.notnull(df), None).replace('NaN', None)
                    return self._json_safe(df.to_dict('records'))

                return []
        except Exception as e:
            logger.error(f"Error getting places by location: {e}")
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
                WHERE LOWER(city) = LOWER(%s) AND is_sustainable = TRUE
                ORDER BY google_review_rating DESC NULLS LAST
                LIMIT 10
                """
                df = pd.read_sql_query(query, conn, params=(city,))
                
                # Replace NaN values with None to prevent JSON serialization errors
                df = df.where(pd.notnull(df), None)
                
                # Also replace string 'NaN' values with None
                df = df.replace('NaN', None)
                
                return self._json_safe(df.to_dict('records'))
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
                # First check if any embeddings exist
                check_query = "SELECT COUNT(*) FROM hotels WHERE embedding IS NOT NULL"
                cursor = conn.cursor()
                cursor.execute(check_query)
                embedding_count = cursor.fetchone()[0]
                
                if embedding_count == 0:
                    # No embeddings available, return top-rated hotels as fallback
                    logger.warning("No embeddings available, returning top-rated hotels as fallback")
                    fallback_query = """
                    SELECT id, name, city, rating, price_range, amenities, description
                    FROM hotels 
                    WHERE rating IS NOT NULL
                    ORDER BY rating DESC
                    LIMIT %s
                    """
                    df = pd.read_sql_query(fallback_query, conn, params=(limit,))
                else:
                    # Use vector similarity search
                    query = """
                    SELECT id, name, city, rating, price_range, amenities, description,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM hotels 
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                    # Convert list to string format for vector casting
                    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                    df = pd.read_sql_query(query, conn, params=(embedding_str, embedding_str, limit))
                
                # Replace NaN values with None to prevent JSON serialization errors
                df = df.where(pd.notnull(df), None)
                
                # Also replace string 'NaN' values with None
                df = df.replace('NaN', None)
                
                return self._json_safe(df.to_dict('records'))
        except Exception as e:
            logger.error(f"Error searching similar hotels: {e}")
            return []
    
    def search_similar_places(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar places using vector similarity"""
        try:
            with self.get_connection() as conn:
                # First check if any embeddings exist
                check_query = "SELECT COUNT(*) FROM places WHERE embedding IS NOT NULL"
                cursor = conn.cursor()
                cursor.execute(check_query)
                embedding_count = cursor.fetchone()[0]
                
                if embedding_count == 0:
                    # No embeddings available, return top-rated places as fallback
                    logger.warning("No embeddings available, returning top-rated places as fallback")
                    fallback_query = """
                    SELECT id, name, city, type, significance, google_review_rating,
                           is_sustainable, sustainability_reason, description
                    FROM places 
                    WHERE google_review_rating IS NOT NULL
                    ORDER BY google_review_rating DESC
                    LIMIT %s
                    """
                    df = pd.read_sql_query(fallback_query, conn, params=(limit,))
                else:
                    # Use vector similarity search
                    query = """
                    SELECT id, name, city, type, significance, google_review_rating,
                           is_sustainable, sustainability_reason, description,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM places 
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                    # Convert list to string format for vector casting
                    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                    df = pd.read_sql_query(query, conn, params=(embedding_str, embedding_str, limit))
                
                # Replace NaN values with None to prevent JSON serialization errors
                df = df.where(pd.notnull(df), None)
                
                # Also replace string 'NaN' values with None
                df = df.replace('NaN', None)
                
                return self._json_safe(df.to_dict('records'))
        except Exception as e:
            logger.error(f"Error searching similar places: {e}")
            return []
    
    # Chat History Management Methods
    def create_chat_session(self, user_id: int, title: str, chat_type: str = 'regular') -> int:
        """Create a new chat session and return session ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chat_sessions (user_id, title, chat_type) VALUES (%s, %s, %s) RETURNING id",
                    (user_id, title, chat_type)
                )
                session_id = cursor.fetchone()[0]
                conn.commit()
                return session_id
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise
    
    def add_chat_message(self, session_id: int, message_type: str, content: str, 
                        travel_data: Optional[Dict] = None, is_error: bool = False) -> int:
        """Add a message to a chat session and return message ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert travel_data to JSON string if provided
                travel_data_json = json.dumps(travel_data) if travel_data else None
                
                cursor.execute(
                    "INSERT INTO chat_messages (session_id, message_type, content, travel_data, is_error) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (session_id, message_type, content, travel_data_json, is_error)
                )
                message_id = cursor.fetchone()[0]
                
                # Update session's updated_at timestamp
                cursor.execute(
                    "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (session_id,)
                )
                
                conn.commit()
                return message_id
        except Exception as e:
            logger.error(f"Error adding chat message: {e}")
            raise
    
    def get_user_chat_sessions(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get all chat sessions for a user, ordered by most recent"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT id, title, chat_type, created_at, updated_at,
                       (SELECT COUNT(*) FROM chat_messages WHERE session_id = chat_sessions.id) as message_count
                FROM chat_sessions 
                WHERE user_id = %s 
                ORDER BY updated_at DESC 
                LIMIT %s
                """
                df = pd.read_sql_query(query, conn, params=(user_id, limit))
                
                # Replace NaN values with None
                df = df.where(pd.notnull(df), None)
                
                return self._json_safe(df.to_dict('records'))
        except Exception as e:
            logger.error(f"Error getting user chat sessions: {e}")
            return []
    
    def get_chat_session_messages(self, session_id: int, user_id: int) -> List[Dict]:
        """Get all messages for a specific chat session"""
        try:
            with self.get_connection() as conn:
                # First verify the session belongs to the user
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM chat_sessions WHERE id = %s AND user_id = %s",
                    (session_id, user_id)
                )
                if not cursor.fetchone():
                    raise ValueError("Session not found or access denied")
                
                query = """
                SELECT id, message_type, content, travel_data, is_error, created_at
                FROM chat_messages 
                WHERE session_id = %s 
                ORDER BY created_at ASC
                """
                df = pd.read_sql_query(query, conn, params=(session_id,))
                
                # Replace NaN values with None
                df = df.where(pd.notnull(df), None)
                
                # Parse travel_data JSON if it exists
                messages = self._json_safe(df.to_dict('records'))
                for message in messages:
                    if message.get('travel_data'):
                        try:
                            message['travel_data'] = json.loads(message['travel_data'])
                        except (json.JSONDecodeError, TypeError):
                            message['travel_data'] = None
                
                return messages
        except Exception as e:
            logger.error(f"Error getting chat session messages: {e}")
            return []
    
    def delete_chat_session(self, session_id: int, user_id: int) -> bool:
        """Delete a chat session and all its messages"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Verify the session belongs to the user
                cursor.execute(
                    "SELECT id FROM chat_sessions WHERE id = %s AND user_id = %s",
                    (session_id, user_id)
                )
                if not cursor.fetchone():
                    return False
                
                # Delete the session (messages will be deleted due to CASCADE)
                cursor.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting chat session: {e}")
            return False
    
    def update_chat_session_title(self, session_id: int, user_id: int, new_title: str) -> bool:
        """Update the title of a chat session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Verify the session belongs to the user
                cursor.execute(
                    "SELECT id FROM chat_sessions WHERE id = %s AND user_id = %s",
                    (session_id, user_id)
                )
                if not cursor.fetchone():
                    return False
                
                cursor.execute(
                    "UPDATE chat_sessions SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (new_title, session_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating chat session title: {e}")
            return False

# Global database manager instance
postgres_db_manager = PostgreSQLDatabaseManager()
