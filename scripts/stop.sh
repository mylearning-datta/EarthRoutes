#!/bin/bash

# Travel Planning Application Stop Script
# This script stops all running services

echo "🛑 Stopping Travel Planning Application..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper to kill by port if running
kill_port() {
  local PORT=$1
  local PIDS
  PIDS=$(lsof -ti :"${PORT}" 2>/dev/null)
  if [ -n "${PIDS}" ]; then
    echo -e "${YELLOW}⚠️  Port ${PORT} is in use by: ${PIDS}. Stopping...${NC}"
    kill ${PIDS} 2>/dev/null || true
    # Force kill if still running after short wait
    sleep 1
    PIDS=$(lsof -ti :"${PORT}" 2>/dev/null)
    if [ -n "${PIDS}" ]; then
      echo -e "${YELLOW}⚠️  Force stopping processes on port ${PORT}...${NC}"
      kill -9 ${PIDS} 2>/dev/null || true
    fi
  else
    echo -e "${GREEN}✅ Port ${PORT} is free${NC}"
  fi
}

# Stop backend (8000) and frontend (3000)
echo -e "${BLUE}🔍 Checking for running services...${NC}"
kill_port 8000
kill_port 3000

# Also kill any remaining processes by name
echo -e "${BLUE}🔍 Checking for remaining processes...${NC}"

# Kill Python backend processes
if pgrep -f "python start_backend.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  Stopping Python backend processes...${NC}"
    pkill -f "python start_backend.py" 2>/dev/null || true
fi

# Kill React frontend processes
if pgrep -f "react-scripts start" > /dev/null; then
    echo -e "${YELLOW}⚠️  Stopping React frontend processes...${NC}"
    pkill -f "react-scripts start" 2>/dev/null || true
fi

# Kill npm processes
if pgrep -f "npm start" > /dev/null; then
    echo -e "${YELLOW}⚠️  Stopping npm processes...${NC}"
    pkill -f "npm start" 2>/dev/null || true
fi

# Wait until ports are freed
echo -e "${BLUE}⏳ Waiting for ports to be freed...${NC}"
for PORT in 8000 3000; do
  for i in {1..10}; do
    if lsof -ti :"${PORT}" >/dev/null 2>&1; then
      sleep 0.5
    else
      break
    fi
  done
done

echo ""
echo -e "${GREEN}✅ All services stopped successfully!${NC}"
echo "=================================================="
echo -e "${BLUE}📡 Services status:${NC}"
echo -e "   • Backend (port 8000): ${GREEN}Stopped${NC}"
echo -e "   • Frontend (port 3000): ${GREEN}Stopped${NC}"
echo ""
echo -e "${YELLOW}💡 To start the application, run ./scripts/start.sh${NC}"
echo "=================================================="
