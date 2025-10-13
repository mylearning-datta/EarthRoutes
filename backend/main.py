#!/usr/bin/env python3
"""
Unified Travel Planning Backend with AI Agent
Combines Node.js backend functionality with Python ReAct agent
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query
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
from workflows.advanced_react_agent import process_travel_query_advanced_react_finetuned
from config.settings import settings
from services.google_maps import GoogleMapsService
from services.co2_service import CO2EmissionService
from services.semantic_search_service import semantic_search_service
from services.batch_embedding_service import batch_embedding_service
from services.finetuned_model_service import get_finetuned_model_service

# Create FastAPI app
app = FastAPI(
    title="Travel Planning API with AI Agent",
    description="Unified backend combining travel planning, CO2 calculations, and AI agent",
    version="1.0.0"
)

# Preload finetuned model on startup (so first request doesn't fail)
@app.on_event("startup")
async def _preload_finetuned_model():
    try:
        # This will construct and load the model service once
        _ = get_finetuned_model_service()
        print("Finetuned model preload triggered (MLX/transformers)")
    except Exception as e:
        print(f"Warning: Finetuned model preload failed: {e}")

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
    session_id: Optional[int] = None

class ChatResponse(BaseModel):
    success: bool
    response: str
    travel_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    intermediate_steps: Optional[List[Dict]] = None
    session_id: Optional[int] = None

class ChatSessionRequest(BaseModel):
    title: str
    chat_type: str = 'regular'

class ChatSessionResponse(BaseModel):
    id: int
    title: str
    chat_type: str
    created_at: str
    updated_at: str
    message_count: int

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
        
        # Create users table (desired schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Lightweight migration: if an older schema with 'password' column exists,
        # and 'password_hash' does not, rename the column to 'password_hash'
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'users'
        """)
        existing_columns = {row[0] for row in cursor.fetchall()}
        if 'password' in existing_columns and 'password_hash' not in existing_columns:
            cursor.execute("ALTER TABLE users RENAME COLUMN password TO password_hash;")
        
        # Ensure users.email is nullable (frontend treats email as optional)
        cursor.execute(
            """
            SELECT is_nullable FROM information_schema.columns
            WHERE table_name='users' AND column_name='email'
            """
        )
        email_nullable_row = cursor.fetchone()
        if email_nullable_row and str(email_nullable_row[0]).upper() == 'NO':
            cursor.execute("ALTER TABLE users ALTER COLUMN email DROP NOT NULL;")
        
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
            "INSERT INTO users (username, password_hash, email) VALUES (%s, %s, %s) RETURNING id",
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
    
    cursor.execute(
        "SELECT id, username, email, password_hash, created_at FROM users WHERE username = %s",
        (username,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "username": row[1],
            "email": row[2],
            "password_hash": row[3],
            "created_at": row[4]
        }
    return None

def find_user_by_id(user_id: int) -> Optional[Dict]:
    """Find user by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, username, email, password_hash, created_at FROM users WHERE id = %s",
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "username": row[1],
            "email": row[2],
            "password_hash": row[3],
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

def generate_chat_title(user_message: str) -> str:
    """Generate a title for a chat session based on the user's first message"""
    # Clean and truncate the message
    clean_message = user_message.strip()
    if len(clean_message) > 50:
        clean_message = clean_message[:47] + "..."
    
    # Add emoji based on content
    message_lower = clean_message.lower()
    if any(word in message_lower for word in ['travel', 'trip', 'journey', 'visit']):
        emoji = "✈️"
    elif any(word in message_lower for word in ['hotel', 'accommodation', 'stay']):
        emoji = "🏨"
    elif any(word in message_lower for word in ['place', 'attraction', 'sightseeing']):
        emoji = "📍"
    elif any(word in message_lower for word in ['co2', 'emission', 'carbon', 'sustainable']):
        emoji = "🌱"
    elif any(word in message_lower for word in ['distance', 'route', 'way']):
        emoji = "🗺️"
    else:
        emoji = "💬"
    
    return f"{emoji} {clean_message}"

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

# Initialize services
google_maps_service = GoogleMapsService()
from services.co2_service import CO2EmissionService as SharedCO2Service
co2_service = SharedCO2Service()

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
    if not verify_password(login_data.password, user["password_hash"]):
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
    chat_request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Chat with the AI travel planning agent"""
    if not chat_request.message or not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        from utils.postgres_database import postgres_db_manager
        
        # Get or create session
        session_id = chat_request.session_id
        if not session_id:
            # Create new session with generated title
            title = generate_chat_title(chat_request.message)
            session_id = postgres_db_manager.create_chat_session(
                current_user["id"], title, "regular"
            )
        
        # Add user message to session
        postgres_db_manager.add_chat_message(
            session_id, "user", chat_request.message.strip()
        )
        
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
        
        # Add bot response to session
        postgres_db_manager.add_chat_message(
            session_id, "bot", result["response"], 
            result.get("travel_data"), result.get("error") is not None
        )
        
        return ChatResponse(
            success=True,
            response=result["response"],
            travel_data=result["travel_data"],
            error=result["error"],
            intermediate_steps=intermediate_steps,
            session_id=session_id
        )
    
    except Exception as e:
        return ChatResponse(
            success=False,
            response=f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
            error=str(e),
            session_id=session_id
        )

@app.post("/api/chat/finetuned", response_model=ChatResponse)
async def chat_with_finetuned_model(
    chat_request: ChatRequest,
    current_user: Dict = Depends(get_current_user),
    variant: Optional[str] = Query(None, description="Model variant: 'community' or 'finetuned'. Default from env MODEL_MODE")
):
    """Chat with the fine-tuned travel sustainability model through the same ReAct toolchain"""
    if not chat_request.message or not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        from utils.postgres_database import postgres_db_manager
        
        # Get or create session
        session_id = chat_request.session_id
        if not session_id:
            # Create new session with generated title
            title = generate_chat_title(chat_request.message)
            session_id = postgres_db_manager.create_chat_session(
                current_user["id"], title, "finetuned"
            )
        
        # Add user message to session
        postgres_db_manager.add_chat_message(
            session_id, "user", chat_request.message.strip()
        )
        
        # Map external variants to internal service names is handled by the LLM wrapper.
        # When variant is None, the wrapper falls back to env (MODEL_MODE/FINETUNED_MODEL_VARIANT).
        # Accepted values here: 'community' | 'finetuned' | None
        result = process_travel_query_advanced_react_finetuned(chat_request.message.strip(), variant=variant)

        # Normalize intermediate_steps like the GPT path (tuples -> dicts)
        intermediate_steps = result.get("intermediate_steps")
        if intermediate_steps:
            intermediate_steps = [
                {"step": str(step[0]), "result": str(step[1])}
                for step in intermediate_steps
                if isinstance(step, tuple) and len(step) >= 2
            ]

        # Add bot response to session
        postgres_db_manager.add_chat_message(
            session_id, "bot", result["response"], 
            result.get("travel_data"), result.get("error") is not None
        )
        
        return ChatResponse(
            success=result.get("error") is None,
            response=result["response"],
            travel_data=result.get("travel_data"),
            error=result.get("error"),
            intermediate_steps=intermediate_steps,
            session_id=session_id
        )
    
    except Exception as e:
        return ChatResponse(
            success=False,
            response=f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
            error=str(e),
            session_id=session_id
        )

@app.get("/api/chat/finetuned/status")
async def get_finetuned_model_status():
    """Get the status of the fine-tuned model"""
    try:
        model_service = get_finetuned_model_service()
        status = model_service.get_model_status()
        
        return {
            "success": True,
            "data": status
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

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

# Semantic Search Endpoints
@app.post("/api/search/places/semantic")
async def search_places_semantic(
    query: str,
    city: Optional[str] = None,
    limit: int = 10,
    current_user: Dict = Depends(get_current_user)
):
    """Search places using natural language queries with AI-powered semantic matching"""
    try:
        result = semantic_search_service.search_places_natural_language(query, city, limit)
        return {
            "success": True,
            "query": query,
            "city_filter": city,
            "results": result.get("results", []),
            "total_results": result.get("total_results", 0),
            "search_type": "semantic"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

@app.post("/api/search/hotels/semantic")
async def search_hotels_semantic(
    query: str,
    city: Optional[str] = None,
    limit: int = 10,
    current_user: Dict = Depends(get_current_user)
):
    """Search hotels using natural language queries with AI-powered semantic matching"""
    try:
        result = semantic_search_service.search_hotels_natural_language(query, city, limit)
        return {
            "success": True,
            "query": query,
            "city_filter": city,
            "results": result.get("results", []),
            "total_results": result.get("total_results", 0),
            "search_type": "semantic"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

@app.post("/api/search/unified")
async def search_unified_semantic(
    query: str,
    city: Optional[str] = None,
    limit: int = 10,
    current_user: Dict = Depends(get_current_user)
):
    """Unified semantic search across both places and hotels"""
    try:
        result = semantic_search_service.search_unified(query, city, limit)
        return {
            "success": True,
            "query": query,
            "city_filter": city,
            "places": result.get("places", []),
            "hotels": result.get("hotels", []),
            "combined_ranking": result.get("combined_ranking", []),
            "total_places": result.get("total_places", 0),
            "total_hotels": result.get("total_hotels", 0),
            "search_type": "unified_semantic"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

# Enhanced Travel Analysis Endpoint
@app.post("/api/travel/analyze-enhanced")
async def analyze_travel_enhanced(
    source: str,
    destination: str,
    preferences: Optional[Dict] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Enhanced travel analysis with AI-powered semantic ranking and recommendations"""
    try:
        # Get travel data using existing tools
        from tools.travel_tools import travel_tools
        travel_data = travel_tools.get_travel_suggestions(source, destination)
        
        # Use enhanced analysis from the advanced react agent
        from workflows.advanced_react_agent import AdvancedTravelPlanningTool
        tool = AdvancedTravelPlanningTool()
        
        # Create analysis with enhanced AI features
        analysis = tool._analyze_travel_options(travel_data, preferences or {})
        
        return {
            "success": True,
            "source": source,
            "destination": destination,
            "travel_data": travel_data,
            "analysis": analysis,
            "preferences": preferences,
            "analysis_type": "enhanced_ai"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source": source,
            "destination": destination
        }

# Batch Embedding Management Endpoints
@app.post("/api/embeddings/batch/places")
async def start_places_batch_embeddings(current_user: Dict = Depends(get_current_user)):
    """Start batch embedding generation for places"""
    try:
        from scripts.populate_embeddings_batch import populate_place_embeddings_batch
        result = populate_place_embeddings_batch()
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/embeddings/batch/hotels")
async def start_hotels_batch_embeddings(current_user: Dict = Depends(get_current_user)):
    """Start batch embedding generation for hotels"""
    try:
        from scripts.populate_embeddings_batch import populate_hotel_embeddings_batch
        result = populate_hotel_embeddings_batch()
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/embeddings/batch/status/{batch_id}")
async def get_batch_job_status(batch_id: str, current_user: Dict = Depends(get_current_user)):
    """Get status of a batch embedding job"""
    try:
        status = batch_embedding_service.get_batch_job_status(batch_id)
        return status
    except Exception as e:
        return {
            "error": str(e),
            "batch_id": batch_id
        }

@app.post("/api/embeddings/batch/cancel/{batch_id}")
async def cancel_batch_job(batch_id: str, current_user: Dict = Depends(get_current_user)):
    """Cancel a batch embedding job"""
    try:
        result = batch_embedding_service.cancel_batch_job(batch_id)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "batch_id": batch_id
        }

@app.get("/api/embeddings/stats")
async def get_embedding_stats(current_user: Dict = Depends(get_current_user)):
    """Get statistics about embeddings in the database"""
    try:
        with postgres_db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get places stats
            cursor.execute("SELECT COUNT(*) as total, COUNT(embedding) as with_embeddings FROM places")
            places_stats = cursor.fetchone()
            
            # Get hotels stats
            cursor.execute("SELECT COUNT(*) as total, COUNT(embedding) as with_embeddings FROM hotels")
            hotels_stats = cursor.fetchone()
            
            return {
                "success": True,
                "places": {
                    "total": places_stats[0],
                    "with_embeddings": places_stats[1],
                    "percentage": round((places_stats[1] / places_stats[0]) * 100, 2) if places_stats[0] > 0 else 0
                },
                "hotels": {
                    "total": hotels_stats[0],
                    "with_embeddings": hotels_stats[1],
                    "percentage": round((hotels_stats[1] / hotels_stats[0]) * 100, 2) if hotels_stats[0] > 0 else 0
                }
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Chat History Management Endpoints
@app.get("/api/chat/sessions")
async def get_chat_sessions(current_user: Dict = Depends(get_current_user)):
    """Get all chat sessions for the current user"""
    try:
        from utils.postgres_database import postgres_db_manager
        sessions = postgres_db_manager.get_user_chat_sessions(current_user["id"])
        return {
            "success": True,
            "sessions": sessions
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/chat/sessions/{session_id}/messages")
async def get_chat_session_messages(
    session_id: int, 
    current_user: Dict = Depends(get_current_user)
):
    """Get all messages for a specific chat session"""
    try:
        from utils.postgres_database import postgres_db_manager
        messages = postgres_db_manager.get_chat_session_messages(session_id, current_user["id"])
        return {
            "success": True,
            "session_id": session_id,
            "messages": messages
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: int, 
    current_user: Dict = Depends(get_current_user)
):
    """Delete a chat session and all its messages"""
    try:
        from utils.postgres_database import postgres_db_manager
        success = postgres_db_manager.delete_chat_session(session_id, current_user["id"])
        if success:
            return {
                "success": True,
                "message": "Chat session deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.put("/api/chat/sessions/{session_id}/title")
async def update_chat_session_title(
    session_id: int,
    title_data: Dict[str, str],
    current_user: Dict = Depends(get_current_user)
):
    """Update the title of a chat session"""
    try:
        from utils.postgres_database import postgres_db_manager
        new_title = title_data.get("title")
        if not new_title:
            raise HTTPException(status_code=400, detail="Title is required")
        
        success = postgres_db_manager.update_chat_session_title(
            session_id, current_user["id"], new_title
        )
        if success:
            return {
                "success": True,
                "message": "Chat session title updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
