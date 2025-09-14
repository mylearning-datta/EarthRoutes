#!/usr/bin/env python3
"""
Unified Travel Planning Backend with AI Agent
Combines Node.js backend functionality with Python ReAct agent
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import os
import psycopg2
import psycopg2.extras
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
import requests
import json
import math
from pathlib import Path

from workflows.advanced_react_agent import process_travel_query_advanced_react
from config.settings import settings
from services.google_maps import GoogleMapsService
from services.co2_service import CO2EmissionService

# Create FastAPI app
app = FastAPI(
    title="Travel Planning API with AI Agent",
    description="Unified backend combining travel planning, CO2 calculations, and AI agent",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database setup
JWT_SECRET = settings.JWT_SECRET
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Pydantic models
class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class DistanceRequest(BaseModel):
    source: str
    destination: str
    mode: str = "driving"
    includeTraffic: bool = False

class CO2Request(BaseModel):
    distanceKm: float
    travelMode: str
    options: Optional[Dict] = {}

class CO2CompareRequest(BaseModel):
    distanceKm: float
    travelModes: List[str]

class CO2SavingsRequest(BaseModel):
    distanceKm: float
    fromMode: str
    toMode: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    travel_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    intermediate_steps: Optional[List[Dict]] = None

# Database functions
def get_db_connection():
    """Get PostgreSQL database connection"""
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD
    )

def init_database():
    """Initialize the database with required tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create CO2 emissions table
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
        
        conn.commit()
        conn.close()
        print("PostgreSQL database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        salt, password_hash = hashed_password.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except:
        return False

def create_user(username: str, password: str, email: str = None) -> Dict:
    """Create a new user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    hashed_password = hash_password(password)
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password, email) VALUES (%s, %s, %s) RETURNING id",
            (username, hashed_password, email)
        )
        user_id = cursor.fetchone()[0]
        conn.commit()
        
        return {
            "id": user_id,
            "username": username,
            "email": email,
            "created_at": datetime.now().isoformat()
        }
    except psycopg2.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    finally:
        conn.close()

def find_user_by_username(username: str) -> Optional[Dict]:
    """Find user by username"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "username": row[1],
            "password": row[2],
            "email": row[3],
            "created_at": row[4]
        }
    return None

def find_user_by_id(user_id: int) -> Optional[Dict]:
    """Find user by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "username": row[1],
            "password": row[2],
            "email": row[3],
            "created_at": row[4]
        }
    return None

def create_jwt_token(user_id: int, username: str) -> str:
    """Create JWT token"""
    payload = {
        "id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    user = find_user_by_id(payload["id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Google Maps service
class GoogleMapsService:
    def __init__(self):
        self.api_key = settings.GOOGLE_MAPS_API_KEY
        self.base_url = "https://maps.googleapis.com/maps/api"
    
    def get_distance(self, source: str, destination: str, mode: str = "driving") -> Dict:
        """Get distance using Google Maps API"""
        if not self.api_key:
            return self._estimate_distance(source, destination)
        
        url = f"{self.base_url}/distancematrix/json"
        params = {
            "origins": source,
            "destinations": destination,
            "mode": mode,
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data["status"] == "OK" and data["rows"][0]["elements"][0]["status"] == "OK":
                element = data["rows"][0]["elements"][0]
                return {
                    "distance": element["distance"],
                    "duration": element["duration"],
                    "origin": source,
                    "destination": destination,
                    "mode": mode
                }
            else:
                return self._estimate_distance(source, destination)
        except Exception as e:
            print(f"Google Maps API error: {e}")
            return self._estimate_distance(source, destination)
    
    def _estimate_distance(self, source: str, destination: str) -> Dict:
        """Fallback distance estimation"""
        city_distances = {
            'Delhi-Mumbai': 1400,
            'Mumbai-Bangalore': 850,
            'Delhi-Bangalore': 2150,
            'Mumbai-Hyderabad': 710,
            'Delhi-Hyderabad': 1600,
            'Bangalore-Hyderabad': 570,
            'Delhi-Kolkata': 1500,
            'Mumbai-Kolkata': 2000,
            'Bangalore-Kolkata': 1900,
            'Hyderabad-Kolkata': 1200
        }
        
        route = f"{source}-{destination}"
        reverse_route = f"{destination}-{source}"
        
        distance_km = city_distances.get(route) or city_distances.get(reverse_route) or 500
        
        return {
            "distance": {
                "text": f"{distance_km} km",
                "value": distance_km * 1000
            },
            "duration": {
                "text": "Estimated",
                "value": 0
            },
            "origin": source,
            "destination": destination,
            "mode": "estimated"
        }

# CO2 Emission service
class CO2EmissionService:
    def __init__(self):
        self.emission_factors = {
            "flight": 0.255,  # kg CO2 per km
            "diesel_car": 0.171,
            "petrol_car": 0.192,
            "electric_car": 0.053,
            "train_diesel": 0.041,
            "bus_shared": 0.089,
            "train_electric": 0.041,
            "bicycle": 0.0,
            "walking": 0.0
        }
    
    def calculate_emissions(self, distance_km: float, travel_mode: str, options: Dict = {}) -> Dict:
        """Calculate CO2 emissions for a given distance and travel mode"""
        emission_factor = self.emission_factors.get(travel_mode, 0.1)
        total_emissions = distance_km * emission_factor
        
        # Calculate equivalent metrics
        trees_needed = math.ceil(total_emissions / 22)  # 1 tree absorbs ~22kg CO2/year
        daily_average_percentage = (total_emissions / 16.4) * 100  # Average person emits 16.4kg CO2/day
        
        return {
            "distanceKm": distance_km,
            "travelMode": travel_mode,
            "emissionFactor": emission_factor,
            "totalEmissions": total_emissions,
            "equivalentMetrics": {
                "treesNeeded": trees_needed,
                "dailyAveragePercentage": daily_average_percentage
            }
        }
    
    def compare_emissions(self, distance_km: float, travel_modes: List[str]) -> List[Dict]:
        """Compare CO2 emissions between different travel modes"""
        comparisons = []
        for mode in travel_modes:
            emission_data = self.calculate_emissions(distance_km, mode)
            comparisons.append(emission_data)
        
        # Sort by emissions (lowest first)
        comparisons.sort(key=lambda x: x["totalEmissions"])
        return comparisons
    
    def calculate_savings(self, distance_km: float, from_mode: str, to_mode: str) -> Dict:
        """Calculate CO2 savings by switching travel modes"""
        from_emissions = self.calculate_emissions(distance_km, from_mode)
        to_emissions = self.calculate_emissions(distance_km, to_mode)
        
        savings = from_emissions["totalEmissions"] - to_emissions["totalEmissions"]
        savings_percentage = (savings / from_emissions["totalEmissions"]) * 100 if from_emissions["totalEmissions"] > 0 else 0
        
        return {
            "distanceKm": distance_km,
            "fromMode": from_mode,
            "toMode": to_mode,
            "fromEmissions": from_emissions["totalEmissions"],
            "toEmissions": to_emissions["totalEmissions"],
            "savings": savings,
            "savingsPercentage": savings_percentage
        }
    
    def get_all_emission_factors(self) -> Dict:
        """Get all available emission factors"""
        return self.emission_factors

# Initialize services
google_maps_service = GoogleMapsService()
co2_service = CO2EmissionService()

# Initialize database
init_database()

# Routes

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"message": "Unified Travel Planning API is running!"}

@app.post("/api/register")
async def register_user(user_data: UserRegister):
    """Register a new user"""
    if not user_data.username or not user_data.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    # Check if user already exists
    existing_user = find_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create user
    new_user = create_user(user_data.username, user_data.password, user_data.email)
    
    return {
        "message": "User created successfully",
        "user": {
            "id": new_user["id"],
            "username": new_user["username"],
            "email": new_user["email"]
        }
    }

@app.post("/api/login")
async def login_user(login_data: UserLogin):
    """Login user and return JWT token"""
    if not login_data.username or not login_data.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    # Find user
    user = find_user_by_username(login_data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not verify_password(login_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate JWT token
    token = create_jwt_token(user["id"], user["username"])
    
    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"]
        }
    }

@app.get("/api/profile")
async def get_profile(current_user: Dict = Depends(get_current_user)):
    """Get user profile"""
    return {
        "user": {
            "id": current_user["id"],
            "username": current_user["username"],
            "email": current_user["email"],
            "created_at": current_user["created_at"]
        }
    }

@app.post("/api/distance")
async def calculate_distance(
    distance_request: DistanceRequest
):
    """Calculate distance between two cities"""
    if not distance_request.source or not distance_request.destination:
        raise HTTPException(status_code=400, detail="Source and destination are required")
    
    if distance_request.source == distance_request.destination:
        raise HTTPException(status_code=400, detail="Source and destination cannot be the same")
    
    # Get distance data
    distance_data = google_maps_service.get_distance(
        distance_request.source,
        distance_request.destination,
        distance_request.mode
    )
    
    # Calculate CO2 emissions
    distance_km = distance_data["distance"]["value"] / 1000
    co2_data = co2_service.calculate_emissions(distance_km, distance_request.mode)
    distance_data["co2Emissions"] = co2_data
    
    return {
        "success": True,
        "data": distance_data
    }

@app.post("/api/distance/batch")
async def calculate_multiple_distances(
    payload: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    origins = payload.get("origins")
    destinations = payload.get("destinations")
    mode = payload.get("mode", "driving")
    if not origins or not destinations or not isinstance(origins, list) or not isinstance(destinations, list):
        raise HTTPException(status_code=400, detail="Origins and destinations must be arrays")
    if len(origins) == 0 or len(destinations) == 0:
        raise HTTPException(status_code=400, detail="Origins and destinations arrays cannot be empty")
    distance_list = google_maps_service.get_multiple_distances(origins, destinations, mode)
    return {"success": True, "data": distance_list, "count": len(distance_list)}

@app.post("/api/co2/calculate")
async def calculate_co2(
    co2_request: CO2Request
):
    """Calculate CO2 emissions for a specific distance and travel mode"""
    if co2_request.distanceKm <= 0:
        raise HTTPException(status_code=400, detail="Distance must be greater than 0")
    
    co2_data = co2_service.calculate_emissions(
        co2_request.distanceKm,
        co2_request.travelMode,
        co2_request.options
    )
    
    return {
        "success": True,
        "data": co2_data
    }

@app.post("/api/co2/compare")
async def compare_co2(
    compare_request: CO2CompareRequest
):
    """Compare CO2 emissions between different travel modes"""
    if compare_request.distanceKm <= 0:
        raise HTTPException(status_code=400, detail="Distance must be greater than 0")
    
    comparisons = co2_service.compare_emissions(
        compare_request.distanceKm,
        compare_request.travelModes
    )
    
    return {
        "success": True,
        "data": {
            "distanceKm": compare_request.distanceKm,
            "comparisons": comparisons
        }
    }

@app.post("/api/co2/savings")
async def calculate_co2_savings(
    savings_request: CO2SavingsRequest
):
    """Calculate CO2 savings by switching travel modes"""
    if savings_request.distanceKm <= 0:
        raise HTTPException(status_code=400, detail="Distance must be greater than 0")
    
    savings = co2_service.calculate_savings(
        savings_request.distanceKm,
        savings_request.fromMode,
        savings_request.toMode
    )
    
    return {
        "success": True,
        "data": savings
    }

@app.get("/api/co2/modes")
async def get_emission_factors():
    """Get all available travel modes and emission factors"""
    emission_factors = co2_service.get_all_emission_factors()
    
    return {
        "success": True,
        "data": emission_factors
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(
    chat_request: ChatRequest
):
    """Chat with the AI travel planning agent"""
    if not chat_request.message or not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        # Process the query through the ReAct travel agent
        result = process_travel_query_advanced_react(chat_request.message)
        
        # Convert intermediate_steps tuples to dictionaries if they exist
        intermediate_steps = result.get("intermediate_steps")
        if intermediate_steps:
            # Convert tuples to dictionaries for Pydantic validation
            intermediate_steps = [
                {"step": str(step[0]), "result": str(step[1])} 
                for step in intermediate_steps 
                if isinstance(step, tuple) and len(step) >= 2
            ]
        
        return ChatResponse(
            success=True,
            response=result["response"],
            travel_data=result["travel_data"],
            error=result["error"],
            intermediate_steps=intermediate_steps
        )
    
    except Exception as e:
        return ChatResponse(
            success=False,
            response=f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
            error=str(e)
        )

# Additional endpoints for cities and travel data
@app.get("/api/cities")
async def get_cities():
    """Get available cities (public for login screens)"""
    # Try DB cities via utils if available, fallback to settings.MAJOR_CITIES
    try:
        from utils.postgres_database import postgres_db_manager
        cities = postgres_db_manager.get_cities()
        if not cities:
            cities = settings.MAJOR_CITIES
    except Exception:
        cities = settings.MAJOR_CITIES
    return {"cities": cities}

@app.get("/api/sustainable-places/{city}")
async def get_sustainable_places(city: str, current_user: Dict = Depends(get_current_user)):
    """Get sustainable places in a specific city"""
    try:
        from utils.postgres_database import postgres_db_manager
        sustainable_places = postgres_db_manager.get_sustainable_places_in_city(city)
        return {
            "success": True,
            "city": city,
            "sustainable_places": sustainable_places,
            "count": len(sustainable_places)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sustainable places: {str(e)}")

@app.get("/api/places/{city}")
async def get_all_places(city: str, current_user: Dict = Depends(get_current_user)):
    """Get all places in a specific city"""
    try:
        from utils.postgres_database import postgres_db_manager
        all_places = postgres_db_manager.get_places_in_city(city)
        sustainable_places = postgres_db_manager.get_sustainable_places_in_city(city)
        return {
            "success": True,
            "city": city,
            "places": all_places,
            "sustainable_places": sustainable_places,
            "total_count": len(all_places),
            "sustainable_count": len(sustainable_places)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting places: {str(e)}")

@app.get("/api/cities/all")
async def get_all_cities(current_user: Dict = Depends(get_current_user)):
    """Get hotel and places city lists (requires auth)"""
    try:
        from utils.postgres_database import postgres_db_manager
        hotel_cities = postgres_db_manager.get_hotels_in_city_list()
        place_cities = postgres_db_manager.get_places_in_city_list()
        common_cities = postgres_db_manager.get_cities()
        return {
            "hotelCities": hotel_cities,
            "placeCities": place_cities,
            "commonCities": common_cities
        }
    except Exception as e:
        return {
            "hotelCities": [],
            "placeCities": [],
            "commonCities": settings.MAJOR_CITIES,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
