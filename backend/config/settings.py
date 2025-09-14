import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    
    # Database
    DB_PATH = os.getenv("DB_PATH", "../db/travel_data.db")
    
    # JWT
    JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    
    # CO2 Service
    CO2_SERVICE_URL = os.getenv("CO2_SERVICE_URL", "http://localhost:5001")
    
    # Agent Configuration
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o")
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "2000"))
    
    # Fallback values (used if database is unavailable)
    FALLBACK_TRAVEL_MODES = [
        "flight", "diesel_car", "petrol_car", "electric_car",
        "train_diesel", "bus_shared", "train_electric", "bicycle", "walking"
    ]
    
    FALLBACK_CITIES = [
        "Delhi", "Mumbai", "Bangalore", "Hyderabad", "Kolkata",
        "Chennai", "Pune", "Ahmedabad", "Jaipur", "Surat",
        "New York", "San Francisco", "London", "Paris", "Tokyo"
    ]
    
    @property
    def TRAVEL_MODES(self):
        """Get travel modes from database"""
        try:
            from utils.database import db_manager
            return db_manager.get_travel_modes()
        except Exception as e:
            print(f"Warning: Could not load travel modes from database: {e}")
            # Return fallback as dictionary format
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
    
    @property
    def MAJOR_CITIES(self):
        """Get cities from database"""
        try:
            from utils.database import db_manager
            return db_manager.get_cities()
        except Exception as e:
            print(f"Warning: Could not load cities from database: {e}")
            return self.FALLBACK_CITIES

settings = Settings()
