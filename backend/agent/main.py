from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
from workflows.advanced_react_agent import process_travel_query_advanced_react
from config.settings import settings

# Create FastAPI app
app = FastAPI(
    title="Travel Planning Agent API",
    description="AI-powered travel planning assistant with CO2 emission calculations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    travel_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    intermediate_steps: Optional[List[Dict]] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="Travel Planning Agent API is running!"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Agent service is operational"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return travel planning response"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Process the query through the advanced ReAct travel agent
        result = process_travel_query_advanced_react(request.message)
        
        return ChatResponse(
            success=True,
            response=result["response"],
            travel_data=result["travel_data"],
            error=result["error"],
            intermediate_steps=result.get("intermediate_steps")
        )
        
    except Exception as e:
        return ChatResponse(
            success=False,
            response=f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
            travel_data=None,
            error=str(e)
        )

@app.get("/cities")
async def get_cities():
    """Get list of available cities"""
    try:
        from utils.database import db_manager
        cities = db_manager.get_cities()
        return {
            "success": True,
            "cities": cities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cities: {str(e)}")

@app.get("/travel-modes")
async def get_travel_modes():
    """Get available travel modes and their emission factors"""
    try:
        from utils.database import db_manager
        travel_modes = db_manager.get_travel_modes()
        return {
            "success": True,
            "travel_modes": travel_modes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting travel modes: {str(e)}")

@app.post("/compare-route")
async def compare_route(request: Dict[str, str]):
    """Compare travel options for a specific route"""
    try:
        source = request.get("source")
        destination = request.get("destination")
        
        if not source or not destination:
            raise HTTPException(status_code=400, detail="Source and destination are required")
        
        from tools.travel_tools import travel_tools
        result = travel_tools.compare_travel_modes(source, destination)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing route: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
