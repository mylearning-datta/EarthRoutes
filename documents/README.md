# Unified Travel Planning Backend

A unified Python FastAPI backend that combines travel planning, CO2 calculations, and AI agent functionality.

## Features

- **Authentication**: JWT-based user authentication with registration and login
- **Travel Planning**: Distance calculations using Google Maps API
- **CO2 Emissions**: Comprehensive CO2 emission calculations and comparisons
- **AI Agent**: Integrated ReAct agent for intelligent travel recommendations
- **Database**: SQLite database for user data and CO2 tracking

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │  Unified Backend │    │   AI Agent      │
│   (React)       │◄──►│   (FastAPI)      │◄──►│  (LangChain)    │
│   Port 3000     │    │   Port 8000      │    │   Integrated    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   SQLite DB      │
                       │   (Users, CO2)   │
                       └──────────────────┘
```

## API Endpoints

### Authentication
- `POST /api/register` - Register new user
- `POST /api/login` - Login user
- `GET /api/profile` - Get user profile

### Travel Planning
- `POST /api/distance` - Calculate distance between cities
- `GET /api/cities` - Get available cities

### CO2 Emissions
- `POST /api/co2/calculate` - Calculate CO2 emissions
- `POST /api/co2/compare` - Compare emissions between modes
- `POST /api/co2/savings` - Calculate savings by switching modes
- `GET /api/co2/modes` - Get all travel modes

### AI Agent
- `POST /api/chat` - Chat with AI travel agent

### Health
- `GET /api/health` - Health check

## Setup

1. **Install Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

3. **Start the Server**
   ```bash
   python start_backend.py
   ```

## Environment Variables

- `OPENAI_API_KEY` - OpenAI API key (required)
- `GOOGLE_MAPS_API_KEY` - Google Maps API key (optional)
- `JWT_SECRET` - JWT secret key
- `DB_PATH` - Database file path
- `AGENT_MODEL` - AI model to use (default: gpt-4o)

## Database Schema

### Users Table
- `id` - Primary key
- `username` - Unique username
- `password` - Hashed password
- `email` - User email
- `created_at` - Creation timestamp

### CO2 Emissions Table
- `id` - Primary key
- `user_id` - Foreign key to users
- `origin` - Origin city
- `destination` - Destination city
- `distance_km` - Distance in kilometers
- `travel_mode` - Travel mode used
- `emission_factor` - CO2 emission factor
- `total_emissions` - Total CO2 emissions
- `trees_needed` - Trees needed to offset
- `daily_average_percentage` - Percentage of daily average
- `created_at` - Creation timestamp

## AI Agent Integration

The backend includes a sophisticated ReAct (Reasoning and Acting) agent that can:

- Analyze travel requirements
- Calculate CO2 emissions
- Compare different travel modes
- Provide eco-friendly recommendations
- Suggest offsetting strategies

## Development

The backend uses FastAPI with automatic API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Test the API endpoints using the interactive documentation or curl:

```bash
# Health check
curl http://localhost:8000/api/health

# Register user
curl -X POST http://localhost:8000/api/register \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test123"}'

# Chat with agent
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"message": "What is the most eco-friendly way to travel from Delhi to Mumbai?"}'
```
