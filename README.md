## Travel Planning Project - Setup and Run Guide

This README covers how to set up PostgreSQL, load data into the database, install backend requirements, and run the backend server.

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ with `pgvector` extension
- macOS/Linux terminal (commands shown assume macOS; Linux notes inline)

Project root: `/Users/arpita/Documents/project`

---

## Project Structure
```text
project/
├─ backend/
│  ├─ config/
│  ├─ scripts/
│  ├─ services/
│  ├─ tools/
│  ├─ utils/
│  ├─ workflows/
│  └─ venv/
├─ frontend/
│  ├─ public/
│  └─ src/
│     ├─ components/
│     └─ services/
├─ finetuning/
│  ├─ data/
│  │  ├─ raw/
│  │  └─ processed/
│  ├─ models/
│  ├─ results/
│  └─ scripts/
├─ db/
│  └─ postgres/            # local PostgreSQL data dir
├─ data/                   # CSV and other datasets
├─ logs/                   # backend.log, frontend.log
└─ scripts/                # setup/start/utility scripts
```

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

## 3) Download local MLX model
Install MLX requirements and download the quantized Mistral model to `finetuning/models/mistral-7b-instruct-4bit-mlx`.
```bash
# From project root
source backend/venv/bin/activate
pip install -r finetuning/requirements_mlx.txt

# Download the model from Hugging Face (default repo + destination are set)
python finetuning/scripts/download_mlx_model.py
```
You can customize the repo/destination if needed:
```bash
python finetuning/scripts/download_mlx_model.py \
  --repo mistralai/Mistral-7B-Instruct-v0.2-4bit-mlx \
  --dest finetuning/models/mistral-7b-instruct-4bit-mlx
```

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
 - `scripts/init_project_structure.sh`: bootstrap directory structure and placeholders

---

## Optional: Finetuning (MLX + LoRA)
You can fine-tune the Mistral 7B Instruct (4-bit MLX) model with LoRA on macOS (Apple Silicon) using MLX.

1) Prepare training data
```bash
source backend/venv/bin/activate
pip install -r finetuning/requirements_mlx.txt
python finetuning/scripts/build_dataset.py
```

2) Train (via config)
```bash
python finetuning/scripts/train_mlx.py --config finetuning/configs/train_mlx.yaml
```

Or train via CLI flags
```bash
python finetuning/scripts/train_mlx.py \
  --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
  --output-dir finetuning/models/mistral-mlx-lora \
  --batch-size 1 --epochs 3 --lr 2e-4 --max-length 1024 \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
```

3) Use finetuned adapters in backend
Set in `backend/.env`:
```env
MODEL_MODE=finetuned
MLX_MODEL=/Users/arpita/Documents/project/finetuning/models/mistral-7b-instruct-4bit-mlx
MLX_ADAPTER_PATH=/Users/arpita/Documents/project/finetuning/models/mistral-mlx-lora
```
