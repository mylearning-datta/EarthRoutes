#!/bin/bash

# Load CSVs into PostgreSQL and populate embeddings
# - Starts PostgreSQL using db/postgres if not already running
# - Ensures backend venv and deps
# - Runs backend/scripts/load_csv_to_postgres.py
# - Runs backend/scripts/populate_embeddings.py

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSTGRES_DATA_DIR="$PROJECT_DIR/db/postgres"

# Add PostgreSQL to PATH (Homebrew default)
export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/homebrew/opt/postgresql@14/bin:$PATH"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

echo -e "${BLUE}📋 Checking prerequisites...${NC}"
if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed.${NC}"; exit 1
fi

if [ ! -d "$POSTGRES_DATA_DIR" ]; then
    echo -e "${RED}❌ PostgreSQL data directory not found: $POSTGRES_DATA_DIR${NC}"
    echo -e "${YELLOW}💡 Run: bash scripts/setup_postgres.sh${NC}"; exit 1
fi

echo -e "${BLUE}🗄️  Starting PostgreSQL (if needed)...${NC}"
if pgrep -f "postgres.*-D.*db/postgres" >/dev/null; then
    echo -e "${GREEN}✅ PostgreSQL already running${NC}"
else
    postgres -D "$POSTGRES_DATA_DIR" -c config_file="$POSTGRES_DATA_DIR/postgresql.conf" &
    POSTGRES_PID=$!
    sleep 3
    if pgrep -x "postgres" >/dev/null; then
        echo -e "${GREEN}✅ PostgreSQL started${NC}"
    else
        echo -e "${RED}❌ Failed to start PostgreSQL${NC}"; exit 1
    fi
fi

echo -e "${BLUE}🔧 Ensuring backend environment...${NC}"
cd "$PROJECT_DIR/backend"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating Python virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  backend/.env not found. Copying from example...${NC}"
    cp env.example .env
    echo -e "${YELLOW}📝 Edit backend/.env and set OPENAI_API_KEY, DB settings${NC}"
fi

# Verify OpenAI key presence (warn only)
if ! grep -q "OPENAI_API_KEY" .env; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY not set in backend/.env. Embeddings will fail.${NC}"
fi

cd "$PROJECT_DIR"

echo -e "${BLUE}📥 Loading CSVs into PostgreSQL...${NC}"
python backend/scripts/load_csv_to_postgres.py

echo -e "${BLUE}🧠 Populating embeddings (standard path)...${NC}"
python backend/scripts/populate_embeddings.py

echo -e "${GREEN}🎉 Done: Data loaded and embeddings populated.${NC}"
