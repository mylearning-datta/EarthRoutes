#!/bin/bash

# Initialize project directory structure and placeholder files
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$PROJECT_DIR/backend/scripts"
mkdir -p "$PROJECT_DIR/backend/services"
mkdir -p "$PROJECT_DIR/backend/utils"
mkdir -p "$PROJECT_DIR/backend/tools"
mkdir -p "$PROJECT_DIR/backend/workflows"
mkdir -p "$PROJECT_DIR/backend/config"
mkdir -p "$PROJECT_DIR/backend/tests"
mkdir -p "$PROJECT_DIR/backend/venv"

mkdir -p "$PROJECT_DIR/frontend/src/components"
mkdir -p "$PROJECT_DIR/frontend/src/services"
mkdir -p "$PROJECT_DIR/frontend/public"

mkdir -p "$PROJECT_DIR/finetuning/scripts"
mkdir -p "$PROJECT_DIR/finetuning/models"
mkdir -p "$PROJECT_DIR/finetuning/data/raw"
mkdir -p "$PROJECT_DIR/finetuning/data/processed"
mkdir -p "$PROJECT_DIR/finetuning/results"

mkdir -p "$PROJECT_DIR/db/postgres"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/temp"
mkdir -p "$PROJECT_DIR/data"

# Ensure .gitkeep files for empty dirs
for d in \
  backend/scripts backend/services backend/utils backend/tools backend/workflows backend/config backend/tests \
  frontend/src/components frontend/src/services frontend/public \
  finetuning/scripts finetuning/models finetuning/data/raw finetuning/data/processed finetuning/results \
  db/postgres logs temp data
do
  touch "$PROJECT_DIR/$d/.gitkeep"
done

# Create default logs files
: > "$PROJECT_DIR/logs/backend.log"
: > "$PROJECT_DIR/logs/frontend.log"

echo "✅ Project directory structure initialized at $PROJECT_DIR"
