# PostgreSQL Migration Guide

This guide explains how to migrate from SQLite to PostgreSQL with vector embeddings support.

## Overview

The migration includes:
- ✅ PostgreSQL with pgvector extension for vector embeddings
- ✅ Data migration from SQLite to PostgreSQL
- ✅ Vector embedding generation for places and hotels
- ✅ Custom data directory: `db/postgres/`
- ✅ All existing functionality preserved

## Directory Structure

```
db/
├── travel_data.db          # Original SQLite database (kept for backup)
└── postgres/               # PostgreSQL data directory
    ├── .gitkeep           # Ensures directory is tracked by git
    ├── postgresql.conf    # PostgreSQL configuration
    └── [data files]       # PostgreSQL data files (ignored by git)
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt
```

### 2. Setup PostgreSQL

```bash
# Run the setup script
./scripts/setup_postgres.sh
```

This script will:
- Install PostgreSQL and pgvector (if not already installed)
- Create the `db/postgres/` directory
- Initialize PostgreSQL data directory
- Create the `travel_data` database
- Enable the pgvector extension

### 3. Migrate Data

```bash
# Migrate data from SQLite to PostgreSQL
python backend/scripts/migrate_to_postgres.py
```

### 4. Generate Vector Embeddings

```bash
# Populate vector embeddings for existing data
python backend/scripts/populate_embeddings.py
```

## Running PostgreSQL

### Start PostgreSQL
```bash
./scripts/start_postgres.sh
```

### Stop PostgreSQL
```bash
./scripts/stop_postgres.sh
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=travel_data
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DATA_DIR=../db/postgres

# Keep existing SQLite path for migration
DB_PATH=../db/travel_data.db
```

### Database Connection

The application will automatically use PostgreSQL when available. If PostgreSQL is not running, it will fall back to the original SQLite database.

## Vector Embeddings

### Features

- **Semantic Search**: Find similar places/hotels based on descriptions
- **AI-Powered Recommendations**: Suggest destinations based on user preferences
- **Content Matching**: Match user queries to relevant travel information

### Embedding Generation

Embeddings are generated for:
- **Places**: Based on name, description, significance, type, sustainability info
- **Hotels**: Based on name, description, amenities, location, price range

### Vector Search

The system supports:
- Cosine similarity search
- HNSW indexing for fast vector queries
- Batch embedding generation

## API Endpoints

All existing API endpoints continue to work unchanged:

- `GET /api/cities` - Get available cities
- `GET /api/places/{city}` - Get places in a city
- `GET /api/sustainable-places/{city}` - Get sustainable places
- `POST /api/chat` - AI chat with travel agent

## Troubleshooting

### PostgreSQL Not Starting

1. Check if the data directory exists:
   ```bash
   ls -la db/postgres/
   ```

2. Check PostgreSQL logs:
   ```bash
   tail -f db/postgres/log/postgresql-*.log
   ```

3. Verify permissions:
   ```bash
   chmod 700 db/postgres/
   ```

### Migration Issues

1. Ensure SQLite database exists:
   ```bash
   ls -la db/travel_data.db
   ```

2. Check PostgreSQL connection:
   ```bash
   psql -h localhost -p 5432 -U postgres -d travel_data
   ```

### Vector Embedding Issues

1. Verify OpenAI API key is set
2. Check embedding generation logs
3. Ensure pgvector extension is enabled:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

## Rollback

If you need to rollback to SQLite:

1. Stop PostgreSQL: `./scripts/stop_postgres.sh`
2. Update your application to use the original SQLite database
3. The original `travel_data.db` file is preserved

## Performance

### Vector Search Performance

- HNSW indexes provide fast similarity search
- Embeddings are cached in the database
- Batch processing for large datasets

### Database Performance

- PostgreSQL handles concurrent connections better than SQLite
- Better indexing and query optimization
- Support for larger datasets

## Security

- Database files are stored locally in `db/postgres/`
- Default credentials should be changed in production
- Vector embeddings don't contain sensitive information

## Next Steps

1. **Production Setup**: Change default passwords and configure SSL
2. **Monitoring**: Set up database monitoring and logging
3. **Backup**: Implement automated backup strategy
4. **Scaling**: Consider connection pooling for high traffic
