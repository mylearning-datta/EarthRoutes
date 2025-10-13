## Travel Planning Project - Setup and Run Guide

This README covers how to set up PostgreSQL, load data into the database, install backend requirements, and run the backend server.

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ with `pgvector` extension
- macOS/Linux terminal (commands shown assume macOS; Linux notes inline)

Project root: `/Users/arpita/Documents/project`

---

## 1) Set up PostgreSQL (with pgvector)

Recommended (macOS):
```bash
bash scripts/setup_postgres.sh
```
What it does:
- Installs PostgreSQL 14 and `pgvector` (via Homebrew on macOS)
- Initializes data dir at `db/postgres`
- Creates database `travel_data`
- Enables `vector` extension
- Prints connection details to use in your `.env`

If you manage Postgres yourself, ensure:
- Database exists: `travel_data`
- `CREATE EXTENSION IF NOT EXISTS vector;` in `travel_data`
- You have credentials to connect (host, port, user, password)

Add/update these in `backend/.env`:
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=travel_data
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DATA_DIR=../db/postgres
```

---

## 2) Install backend requirements
From project root:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: Each new terminal session, activate the environment first:
```bash
# macOS/Linux
source backend/venv/bin/activate
```
Deactivate anytime with:
```bash
deactivate
```

If `venv` is missing on your system
```bash
# Check if venv is available
python3 -m venv --help || true

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y python3-venv

# Fedora/RHEL (use yum if dnf not available)
sudo dnf install -y python3-venv || sudo yum install -y python3-venv

# Arch Linux (venv comes with python)
sudo pacman -Syu python

# macOS (install Python 3 which includes venv)
brew install python  # or download from python.org

# If needed, bootstrap pip and venv helpers
python3 -m ensurepip --upgrade || true
```

Alternative: use virtualenv (if you prefer)
```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv backend/venv
source backend/venv/bin/activate
```
Create your environment file if missing:
```bash
cp env.example .env
```
Update `backend/.env` with required keys (at minimum set your OpenAI key to enable embeddings):
```env
OPENAI_API_KEY=your_openai_api_key
```
Optionally add provider keys you use (e.g., Google Maps):
```env
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

---

## 3) Load the database (CSV → PostgreSQL)
Load `data/hotel_details.csv` and `data/Top Indian Places to Visit.csv` into PostgreSQL with a clean schema and indexes.
```bash
# From project root
source backend/venv/bin/activate
python backend/scripts/load_csv_to_postgres.py
```
This creates tables (`hotels`, `places`, `cities`), clears existing rows, loads fresh data, and prints a summary.

### Populate vector embeddings (recommended)
Populate embeddings so semantic search and recommendations work:
```bash
python backend/scripts/populate_embeddings_batch.py
```
Note: requires `OPENAI_API_KEY` in `backend/.env`.

---

## 4) Run the backend
### Quick start (starts Postgres + backend + frontend)
```bash
# From project root
bash scripts/start.sh
```
This will start PostgreSQL (using `db/postgres`), launch the backend on port 8000, and the frontend on port 3000.

### Backend only
```bash
# From project root
cd backend
source venv/bin/activate
python start_backend.py
```
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

---

## Troubleshooting
- Port in use (5432/8000/3000): stop existing services or change ports.
- `.env` missing keys: copy `backend/env.example` to `backend/.env` and fill values.
- PostgreSQL not running: ensure `db/postgres` exists and either use `scripts/start.sh` or `brew services start postgresql@14` (macOS).
- Embeddings fail: confirm `OPENAI_API_KEY` is set and network access is available.

---

## Useful scripts
- `scripts/setup_postgres.sh`: one-time PostgreSQL + pgvector setup
- `backend/scripts/load_csv_to_postgres.py`: create tables and load CSV data into PostgreSQL
- `backend/scripts/populate_embeddings.py`: generate embeddings for places/hotels
- `backend/scripts/populate_embeddings_batch.py`: embeddings via OpenAI Batch API
- `scripts/start.sh`: start Postgres, backend, and frontend with logs
