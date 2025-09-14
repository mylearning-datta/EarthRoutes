#!/bin/bash

# Start PostgreSQL with custom data directory for Travel App

set -e

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSTGRES_DATA_DIR="$PROJECT_DIR/db/postgres"

# Add PostgreSQL to PATH
export PATH="/opt/homebrew/opt/postgresql@14/bin:$PATH"

echo "🚀 Starting PostgreSQL for Travel App..."
echo "📁 Data directory: $POSTGRES_DATA_DIR"

# Check if data directory exists
if [ ! -d "$POSTGRES_DATA_DIR" ]; then
    echo "❌ PostgreSQL data directory not found: $POSTGRES_DATA_DIR"
    echo "Please run setup_postgres.sh first"
    exit 1
fi

# Check if PostgreSQL is already running
if pgrep -x "postgres" > /dev/null; then
    echo "⚠️  PostgreSQL is already running"
    echo "To stop it, run: ./scripts/stop_postgres.sh"
    exit 0
fi

# Start PostgreSQL with custom data directory
echo "🔄 Starting PostgreSQL server..."
postgres -D "$POSTGRES_DATA_DIR" -c config_file="$POSTGRES_DATA_DIR/postgresql.conf" &

# Wait a moment for PostgreSQL to start
sleep 3

# Check if PostgreSQL started successfully
if pgrep -x "postgres" > /dev/null; then
    echo "✅ PostgreSQL started successfully!"
    echo "📋 Connection details:"
    echo "   Host: localhost"
    echo "   Port: 5432"
    echo "   Database: travel_data"
    echo "   Username: postgres"
    echo "   Password: password"
    echo ""
    echo "🛑 To stop PostgreSQL, run: ./scripts/stop_postgres.sh"
else
    echo "❌ Failed to start PostgreSQL"
    exit 1
fi
