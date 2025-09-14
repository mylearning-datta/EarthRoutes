#!/bin/bash

# PostgreSQL Setup Script for Travel App
# This script sets up PostgreSQL with pgvector extension

set -e

echo "🚀 Setting up PostgreSQL with pgvector for Travel App..."

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📱 Detected macOS"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is not installed. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install PostgreSQL
    echo "📦 Installing PostgreSQL..."
    brew install postgresql@14
    
    # Install pgvector
    echo "📦 Installing pgvector..."
    brew install pgvector
    
    # Add PostgreSQL to PATH
    echo "🔧 Adding PostgreSQL to PATH..."
    export PATH="/opt/homebrew/opt/postgresql@14/bin:$PATH"
    
    # Start PostgreSQL service
    echo "🔄 Starting PostgreSQL service..."
    brew services start postgresql@14
    
    # Wait for PostgreSQL to start
    sleep 3
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Detected Linux"
    
    # Update package list
    sudo apt-get update
    
    # Install PostgreSQL
    echo "📦 Installing PostgreSQL..."
    sudo apt-get install -y postgresql postgresql-contrib
    
    # Install pgvector
    echo "📦 Installing pgvector..."
    sudo apt-get install -y postgresql-15-pgvector
    
    # Start PostgreSQL service
    echo "🔄 Starting PostgreSQL service..."
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    
else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "Please install PostgreSQL and pgvector manually"
    exit 1
fi

# Create database and user
echo "🗄️  Setting up database and user..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSTGRES_DATA_DIR="$PROJECT_DIR/db/postgres"

echo "📁 PostgreSQL data directory: $POSTGRES_DATA_DIR"

# Create PostgreSQL data directory if it doesn't exist
mkdir -p "$POSTGRES_DATA_DIR"

# Add PostgreSQL to PATH for the rest of the script
export PATH="/opt/homebrew/opt/postgresql@14/bin:$PATH"

# Get PostgreSQL version for the correct path
PG_VERSION=$(psql --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "PostgreSQL version: $PG_VERSION"

# Initialize PostgreSQL data directory (only if not already initialized)
if [ ! -f "$POSTGRES_DATA_DIR/PG_VERSION" ]; then
    echo "🔧 Initializing PostgreSQL data directory..."
    # Remove .gitkeep temporarily for initdb
    if [ -f "$POSTGRES_DATA_DIR/.gitkeep" ]; then
        mv "$POSTGRES_DATA_DIR/.gitkeep" "$POSTGRES_DATA_DIR/.gitkeep.bak"
    fi
    initdb -D "$POSTGRES_DATA_DIR" -U postgres
    # Restore .gitkeep
    if [ -f "$POSTGRES_DATA_DIR/.gitkeep.bak" ]; then
        mv "$POSTGRES_DATA_DIR/.gitkeep.bak" "$POSTGRES_DATA_DIR/.gitkeep"
    fi
fi

# Create database
echo "🗄️  Creating database..."
psql -h localhost -p 5432 -U postgres -d postgres -c "CREATE DATABASE travel_data;" || echo "Database travel_data might already exist"

# Create user (postgres user already exists with our custom setup)
echo "👤 User postgres already exists with our custom setup"

# Grant privileges
psql -h localhost -p 5432 -U postgres -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE travel_data TO postgres;"

# Enable pgvector extension
echo "🔧 Enabling pgvector extension..."
psql -h localhost -p 5432 -U postgres -d travel_data -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "✅ PostgreSQL setup completed!"
echo ""
echo "📋 Database connection details:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: travel_data"
echo "   Username: postgres"
echo "   Password: password"
echo ""
echo "🔧 To update your environment variables, add these to your .env file:"
echo "   POSTGRES_HOST=localhost"
echo "   POSTGRES_PORT=5432"
echo "   POSTGRES_DB=travel_data"
echo "   POSTGRES_USER=postgres"
echo "   POSTGRES_PASSWORD=password"
echo ""
echo "🚀 Next steps:"
echo "   1. Install Python dependencies: pip install -r backend/requirements.txt"
echo "   2. Run migration script: python backend/scripts/migrate_to_postgres.py"
echo "   3. Start your application!"
