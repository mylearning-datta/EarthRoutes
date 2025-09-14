#!/bin/bash

# Travel Planning Application Startup Script
# This script starts PostgreSQL, backend and frontend services

echo "🚀 Starting Travel Planning Application..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSTGRES_DATA_DIR="$PROJECT_DIR/db/postgres"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Add common Node.js and PostgreSQL paths to PATH
export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/homebrew/opt/postgresql@14/bin:$PATH"

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check Node.js
if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js first.${NC}"
    echo -e "${YELLOW}💡 You can install Node.js from: https://nodejs.org/${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Node.js found: $(node --version)${NC}"

# Check Python
if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python found: $(python3 --version)${NC}"

# Check if ports are available
if port_in_use 5432; then
    echo -e "${YELLOW}⚠️  Port 5432 is already in use. PostgreSQL may already be running.${NC}"
fi

if port_in_use 8000; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use. Backend may already be running.${NC}"
fi

if port_in_use 3000; then
    echo -e "${YELLOW}⚠️  Port 3000 is already in use. Frontend may already be running.${NC}"
fi

echo ""
echo -e "${BLUE}🗄️  Starting PostgreSQL...${NC}"

# Check if PostgreSQL data directory exists
if [ ! -d "$POSTGRES_DATA_DIR" ]; then
    echo -e "${RED}❌ PostgreSQL data directory not found: $POSTGRES_DATA_DIR${NC}"
    echo -e "${YELLOW}💡 Please run setup_postgres.sh first${NC}"
    exit 1
fi

# Check if PostgreSQL is already running
if pgrep -f "postgres.*-D.*db/postgres" > /dev/null; then
    echo -e "${YELLOW}⚠️  PostgreSQL is already running with our data directory${NC}"
    echo -e "${GREEN}✅ Using existing PostgreSQL instance${NC}"
else
    # Start PostgreSQL with custom data directory
    echo -e "${GREEN}🔄 Starting PostgreSQL server...${NC}"
    postgres -D "$POSTGRES_DATA_DIR" -c config_file="$POSTGRES_DATA_DIR/postgresql.conf" &
    POSTGRES_PID=$!
    
    # Wait a moment for PostgreSQL to start
    sleep 3
    
    # Check if PostgreSQL started successfully
    if pgrep -x "postgres" > /dev/null; then
        echo -e "${GREEN}✅ PostgreSQL started successfully!${NC}"
        echo -e "${BLUE}📋 PostgreSQL connection details:${NC}"
        echo -e "   • Host: localhost"
        echo -e "   • Port: 5432"
        echo -e "   • Database: travel_data"
        echo -e "   • Username: postgres"
        echo -e "   • Password: password"
    else
        echo -e "${RED}❌ Failed to start PostgreSQL${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}🔧 Setting up backend...${NC}"

# Install backend dependencies if needed
if [ ! -d "backend/venv" ]; then
    echo -e "${YELLOW}📦 Creating Python virtual environment...${NC}"
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..
fi

# Check if backend .env exists
if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}⚠️  Backend .env file not found. Copying from example...${NC}"
    cp backend/env.example backend/.env
    echo -e "${YELLOW}📝 Please edit backend/.env and configure your API keys${NC}"
fi

# Install frontend dependencies if needed
echo ""
echo -e "${BLUE}🎨 Setting up frontend...${NC}"
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

echo ""
echo -e "${BLUE}🎯 Starting services...${NC}"

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"
    if [ ! -z "$POSTGRES_PID" ]; then
        kill $POSTGRES_PID 2>/dev/null
    fi
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the Python backend in background
echo -e "${GREEN}🔧 Starting backend on port 8000...${NC}"
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run the setup first.${NC}"
    exit 1
fi

# Activate virtual environment and start backend
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Check if required packages are installed
if ! ./venv/bin/python -c "import fastapi, uvicorn, langchain, openai" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Some required packages are missing. Installing...${NC}"
    ./venv/bin/pip install -r requirements.txt
fi

# Start the backend with logging
echo -e "${GREEN}📝 Starting backend with server logging...${NC}"
./venv/bin/python start_backend.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 5

# Start the frontend in background with logging
echo -e "${GREEN}🎨 Starting frontend on port 3000...${NC}"
cd frontend
npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 5

echo ""
echo -e "${GREEN}🎉 Application started successfully!${NC}"
echo "=================================================="
echo -e "${BLUE}📡 Services running:${NC}"
echo -e "   • PostgreSQL: ${GREEN}localhost:5432${NC}"
echo -e "   • Backend: ${GREEN}http://localhost:8000${NC}"
echo -e "   • Frontend: ${GREEN}http://localhost:3000${NC}"
echo ""
echo -e "${BLUE}📚 API Documentation:${NC}"
echo -e "   • Backend API: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "   • Health Check: ${GREEN}http://localhost:8000/api/health${NC}"
echo ""
echo -e "${YELLOW}💡 To stop the application, press Ctrl+C or run ./scripts/stop.sh${NC}"
echo -e "${BLUE}📝 Server logs are being saved to:${NC}"
echo -e "   • Backend logs: ${GREEN}logs/backend.log${NC}"
echo -e "   • Frontend logs: ${GREEN}logs/frontend.log${NC}"
echo "=================================================="

# Wait for background processes
wait