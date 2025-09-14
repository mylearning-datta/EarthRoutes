#!/bin/bash

# Travel Planning Application Restart Script
# This script stops and then starts all services

echo "🔄 Restarting Travel Planning Application..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}📁 Project root: ${PROJECT_ROOT}${NC}"

# Change to project root directory
cd "$PROJECT_ROOT"

echo ""
echo -e "${BLUE}🛑 Step 1: Stopping services...${NC}"
echo "----------------------------------------"

# Run the stop script
"$SCRIPT_DIR/stop.sh"

echo ""
echo -e "${BLUE}⏳ Waiting 3 seconds before restarting...${NC}"
sleep 3

echo ""
echo -e "${BLUE}🚀 Step 2: Starting services...${NC}"
echo "----------------------------------------"

# Run the start script
"$SCRIPT_DIR/start.sh"
