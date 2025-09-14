#!/bin/bash

# Stop PostgreSQL for Travel App

set -e

echo "🛑 Stopping PostgreSQL for Travel App..."

# Find and stop PostgreSQL processes
if pgrep -x "postgres" > /dev/null; then
    echo "🔄 Stopping PostgreSQL processes..."
    
    # Try graceful shutdown first
    pkill -TERM postgres
    
    # Wait for processes to stop
    sleep 3
    
    # Force kill if still running
    if pgrep -x "postgres" > /dev/null; then
        echo "⚠️  Force stopping PostgreSQL processes..."
        pkill -KILL postgres
        sleep 1
    fi
    
    if ! pgrep -x "postgres" > /dev/null; then
        echo "✅ PostgreSQL stopped successfully!"
    else
        echo "❌ Failed to stop PostgreSQL"
        exit 1
    fi
else
    echo "ℹ️  PostgreSQL is not running"
fi
