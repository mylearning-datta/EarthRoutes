# Backend Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRAVEL PLANNING APPLICATION                           │
│                              (Full Stack System)                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Frontend Layer
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND (React)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   LoginForm     │  │  RegisterForm   │  │   SearchPage    │  │  ChatPage   │ │
│  │                 │  │                 │  │                 │  │             │ │
│  │ • Authentication│  │ • User Creation │  │ • Travel Search │  │ • AI Chat   │ │
│  │ • JWT Handling  │  │ • Form Validation│  │ • CO2 Display  │  │ • ReAct UI  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                              App.js                                        │ │
│  │                    • Route Management                                      │ │
│  │                    • State Management                                      │ │
│  │                    • Navigation Control                                    │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                            Auth Service                                    │ │
│  │                    • API Communication                                     │ │
│  │                    • Token Management                                      │ │
│  │                    • Chat API Integration                                  │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/HTTPS Requests
                                    │ (REST API)
                                    ▼
```

## Backend Layer (Node.js)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            NODE.JS BACKEND SERVER                              │
│                              (Express.js)                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                              server.js                                     │ │
│  │                    • Express App Setup                                     │ │
│  │                    • Middleware Configuration                              │ │
│  │                    • Route Definitions                                     │ │
│  │                    • Error Handling                                        │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                           │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐ │
│  │                                 │                                         │ │
│  │  ┌─────────────────────────────┴─────────────────────────────┐           │ │
│  │  │                    API ROUTES                             │           │ │
│  │  │                                                           │           │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │           │ │
│  │  │  │   Auth      │  │   Travel    │  │       Chat          │ │           │ │
│  │  │  │   Routes    │  │   Routes    │  │      Routes         │ │           │ │
│  │  │  │             │  │             │  │                     │ │           │ │
│  │  │  │ • /register │  │ • /distance │  │ • /chat             │ │           │ │
│  │  │  │ • /login    │  │ • /co2/*    │  │ • /cities           │ │           │ │
│  │  │  │ • /profile  │  │ • /cities   │  │ • /travel-modes     │ │           │ │
│  │  │  │ • /users    │  │             │  │ • /compare-route    │ │           │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │           │ │
│  │  └─────────────────────────────────────────────────────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                           │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐ │
│  │                                 │                                         │ │
│  │  ┌─────────────────────────────┴─────────────────────────────┐           │ │
│  │  │                  MIDDLEWARE LAYER                         │           │ │
│  │  │                                                           │           │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │           │ │
│  │  │  │     CORS    │  │ Body Parser │  │   Authentication    │ │           │ │
│  │  │  │             │  │             │  │                     │ │           │ │
│  │  │  │ • Cross-    │  │ • JSON      │  │ • JWT Verification  │ │           │ │
│  │  │  │   Origin    │  │   Parsing   │  │ • Token Validation  │ │           │ │
│  │  │  │   Requests  │  │ • URL       │  │ • User Context      │ │           │ │
│  │  │  │             │  │   Encoding  │  │                     │ │           │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │           │ │
│  │  └─────────────────────────────────────────────────────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Service Calls
                                    ▼
```

## Service Layer
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SERVICE LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                           Node.js Services                                 │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐   │ │
│  │  │   googleMapsService.js  │  │        co2EmissionService.js            │   │ │
│  │  │                         │  │                                         │   │ │
│  │  │ • Distance Matrix API   │  │ • CO2 Emission Calculations             │   │ │
│  │  │ • Traffic Data          │  │ • Emission Factors                      │   │ │
│  │  │ • Route Optimization    │  │ • Environmental Metrics                 │   │ │
│  │  │ • Geocoding             │  │ • Tree Offset Calculations              │   │ │
│  │  └─────────────────────────┘  └─────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                           │
│                                    │ HTTP Requests to Python Agent            │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        Python Agent Communication                          │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                    Chat Endpoint Handler                            │   │ │
│  │  │                                                                     │   │ │
│  │  │  POST /api/chat → Forward to Python Agent → Return Response        │   │ │
│  │  │                                                                     │   │ │
│  │  │  • Message Validation                                               │   │ │
│  │  │  • Agent Communication                                              │   │ │
│  │  │  • Response Formatting                                              │   │ │
│  │  │  • Error Handling                                                   │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP Requests (Port 8000)
                                    ▼
```

## Python Agent Layer (ReAct Agent)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PYTHON AGENT (FastAPI)                                │
│                            (ReAct Agent with GPT-4o)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                              main.py                                       │ │
│  │                    • FastAPI Application                                   │ │
│  │                    • CORS Configuration                                    │ │
│  │                    • API Endpoints                                         │ │
│  │                    • Request/Response Models                               │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                           │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐ │
│  │                                 │                                         │ │
│  │  ┌─────────────────────────────┴─────────────────────────────┐           │ │
│  │  │                    API ENDPOINTS                          │           │ │
│  │  │                                                           │           │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │           │ │
│  │  │  │    Chat     │  │   Cities    │  │   Travel Modes      │ │           │ │
│  │  │  │             │  │             │  │                     │ │           │ │
│  │  │  │ POST /chat  │  │ GET /cities │  │ GET /travel-modes   │ │           │ │
│  │  │  │             │  │             │  │ POST /compare-route │ │           │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │           │ │
│  │  └─────────────────────────────────────────────────────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                           │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐ │
│  │                                 │                                         │ │
│  │  ┌─────────────────────────────┴─────────────────────────────┐           │ │
│  │  │                  REACT AGENT WORKFLOW                     │           │ │
│  │  │                                                           │           │ │
│  │  │  ┌─────────────────────────────────────────────────────┐ │           │ │
│  │  │  │              advanced_react_agent.py                │ │           │ │
│  │  │  │                                                     │ │           │ │
│  │  │  │  ┌─────────────────────────────────────────────┐   │ │           │ │
│  │  │  │  │            ReAct Process                    │   │ │           │ │
│  │  │  │  │                                             │   │ │           │ │
│  │  │  │  │  1. Question: User Query                    │   │ │           │ │
│  │  │  │  │  2. Thought: Analyze Requirements           │   │ │           │ │
│  │  │  │  │  3. Action: Use Appropriate Tool            │   │ │           │ │
│  │  │  │  │  4. Observation: Process Tool Result        │   │ │           │ │
│  │  │  │  │  5. Repeat: Until Problem Solved            │   │ │           │ │
│  │  │  │  │  6. Final Answer: Comprehensive Response    │   │ │           │ │
│  │  │  │  └─────────────────────────────────────────────┘   │ │           │ │
│  │  │  └─────────────────────────────────────────────────────┘ │           │ │
│  │  └─────────────────────────────────────────────────────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                           │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐ │
│  │                                 │                                         │ │
│  │  ┌─────────────────────────────┴─────────────────────────────┐           │ │
│  │  │                      AGENT TOOLS                          │           │ │
│  │  │                                                           │           │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │           │ │
│  │  │  │ Advanced Travel │  │ Environmental   │  │ City Info   │ │           │ │
│  │  │  │ Planning Tool   │  │ Impact Tool     │  │ Tool        │ │           │ │
│  │  │  │                 │  │                 │  │             │ │           │ │
│  │  │  │ • Route Analysis│  │ • CO2 Analysis  │  │ • Hotels    │ │           │ │
│  │  │  │ • Multi-criteria│  │ • Footprint     │  │ • Places    │ │           │ │
│  │  │  │   Optimization  │  │   Categorization│  │ • Cities    │ │           │ │
│  │  │  │ • Recommendations│  │ • Offsetting    │  │ • Database  │ │           │ │
│  │  │  │                 │  │   Strategies    │  │   Queries   │ │           │ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────┘ │           │ │
│  │  └─────────────────────────────────────────────────────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Tool Calls
                                    ▼
```

## Data Layer
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               DATA LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                           SQLite Databases                                 │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐   │ │
│  │  │      users.db           │  │           travel_data.db                │   │ │
│  │  │                         │  │                                         │   │ │
│  │  │ • User Authentication   │  │ • Hotels Data                           │   │ │
│  │  │ • User Profiles         │  │ • Tourist Places                        │   │ │
│  │  │ • CO2 Emission History  │  │ • City Information                      │   │ │
│  │  │ • Travel Preferences    │  │ • Travel Routes                         │   │ │
│  │  └─────────────────────────┘  └─────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                           │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐ │
│  │                                 │                                         │ │
│  │  ┌─────────────────────────────┴─────────────────────────────┐           │ │
│  │  │                    Database Services                      │           │ │
│  │  │                                                           │           │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │           │ │
│  │  │  │   database.js   │  │  database.py    │  │   Utils     │ │           │ │
│  │  │  │   (Node.js)     │  │   (Python)      │  │             │ │           │ │
│  │  │  │                 │  │                 │  │ • City      │ │           │ │
│  │  │  │ • User CRUD     │  │ • Travel Data   │  │   Queries   │ │           │ │
│  │  │  │ • Auth Queries  │  │ • Hotel Queries │  │ • Data      │ │           │ │
│  │  │  │ • CO2 Tracking  │  │ • Place Queries │  │   Processing│ │           │ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────┘ │           │ │
│  │  └─────────────────────────────────────────────────────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ External API Calls
                                    ▼
```

## External Services
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            EXTERNAL SERVICES                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │ │
│  │  │   OpenAI API    │  │ Google Maps API │  │     Other Services          │ │ │
│  │  │                 │  │                 │  │                             │ │ │
│  │  │ • GPT-4o Model  │  │ • Distance      │  │ • Weather APIs              │ │ │
│  │  │ • ReAct Agent   │  │   Matrix        │  │ • Traffic Data              │ │ │
│  │  │ • Chat Completions│ │ • Geocoding     │  │ • Booking APIs              │ │ │
│  │  │ • Embeddings    │  │ • Traffic Info  │  │ • Payment Gateways          │ │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  User Query → Frontend → Node.js Backend → Python Agent → Tools → Database     │
│      ↑                                                                     │    │
│      └──────────────── Response ←─────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                            DETAILED FLOW                                   │ │
│  │                                                                             │ │
│  │  1. User types message in ChatPage                                         │ │
│  │  2. Frontend sends POST /api/chat to Node.js backend                      │ │
│  │  3. Node.js validates request and forwards to Python agent                │ │
│  │  4. Python agent processes with ReAct workflow:                           │ │
│  │     a. Thought: Analyze user requirements                                  │ │
│  │     b. Action: Use appropriate tool (travel_planner, co2_calculator, etc.)│ │
│  │     c. Observation: Process tool results                                   │ │
│  │     d. Repeat until problem solved                                         │ │
│  │     e. Final Answer: Comprehensive response                                │ │
│  │  5. Python agent returns response to Node.js                              │ │
│  │  6. Node.js forwards response to frontend                                  │ │
│  │  7. Frontend displays response with travel data cards                     │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Port Configuration
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            PORT CONFIGURATION                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Frontend (React)     : http://localhost:3000                                  │
│  Node.js Backend      : http://localhost:5000                                  │
│  Python Agent         : http://localhost:8000                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                            API ENDPOINTS                                   │ │
│  │                                                                             │ │
│  │  Node.js Backend (Port 5000):                                              │ │
│  │  • POST /api/register                                                      │ │
│  │  • POST /api/login                                                         │ │
│  │  • GET  /api/profile                                                       │ │
│  │  • GET  /api/cities                                                        │ │
│  │  • POST /api/distance                                                      │ │
│  │  • POST /api/co2/calculate                                                 │ │
│  │  • POST /api/chat (forwards to Python agent)                              │ │
│  │                                                                             │ │
│  │  Python Agent (Port 8000):                                                 │ │
│  │  • POST /chat                                                              │ │
│  │  • GET  /cities                                                            │ │
│  │  • GET  /travel-modes                                                      │ │
│  │  • POST /compare-route                                                     │ │
│  │  • GET  /health                                                            │ │
│  │  • GET  /docs (FastAPI documentation)                                     │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TECHNOLOGY STACK                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Frontend:                                                                      │
│  • React 18+                                                                    │
│  • JavaScript ES6+                                                              │
│  • CSS3 with modern features                                                    │
│  • Axios for HTTP requests                                                      │
│                                                                                 │
│  Node.js Backend:                                                               │
│  • Node.js 18+                                                                  │
│  • Express.js                                                                   │
│  • SQLite3                                                                      │
│  • JWT for authentication                                                       │
│  • bcryptjs for password hashing                                                │
│  • node-fetch for HTTP requests                                                 │
│                                                                                 │
│  Python Agent:                                                                  │
│  • Python 3.8+                                                                  │
│  • FastAPI                                                                      │
│  • LangChain 0.3+                                                               │
│  • LangGraph                                                                    │
│  • OpenAI GPT-4o                                                                │
│  • Pandas for data processing                                                   │
│  • SQLite3 for database access                                                  │
│                                                                                 │
│  External APIs:                                                                 │
│  • OpenAI API (GPT-4o)                                                         │
│  • Google Maps API (Distance Matrix)                                           │
│                                                                                 │
│  Databases:                                                                     │
│  • SQLite (users.db, travel_data.db)                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Security & Authentication Flow
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY & AUTHENTICATION                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. User Registration/Login:                                                   │
│     Frontend → Node.js Backend → Database → JWT Token → Frontend               │
│                                                                                 │
│  2. Protected Routes:                                                          │
│     Frontend (with JWT) → Node.js (verify JWT) → Process Request               │
│                                                                                 │
│  3. Chat Authentication:                                                       │
│     Frontend → Node.js (verify JWT) → Python Agent (no auth needed)           │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                            SECURITY FEATURES                               │ │
│  │                                                                             │ │
│  │  • JWT-based authentication                                                 │ │
│  │  • Password hashing with bcryptjs                                           │ │
│  │  • CORS configuration                                                       │ │
│  │  • Input validation and sanitization                                        │ │
│  │  • Error handling without information leakage                               │ │
│  │  • Rate limiting (can be added)                                             │ │
│  │  • HTTPS support (for production)                                           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

This comprehensive diagram shows how all components of your travel planning application are integrated, from the React frontend through the Node.js backend to the Python ReAct agent, with proper data flow, security, and external service integration.
