#!/usr/bin/env python3
"""
Migration script to transfer data from SQLite to PostgreSQL
"""

import sqlite3
import psycopg2
import pandas as pd
from pathlib import Path
import sys
import os

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

def get_sqlite_connection():
    """Get SQLite connection"""
    sqlite_path = Path(settings.DB_PATH)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}")
    return sqlite3.connect(str(sqlite_path))

def get_postgres_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD
    )

def migrate_table(sqlite_conn, postgres_conn, table_name, columns_mapping=None):
    """Migrate a single table from SQLite to PostgreSQL"""
    print(f"Migrating table: {table_name}")
    
    # Read data from SQLite
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute(f"SELECT * FROM {table_name}")
    rows = sqlite_cursor.fetchall()
    
    if not rows:
        print(f"  No data found in {table_name}")
        return
    
    # Get column names
    sqlite_cursor.execute(f"PRAGMA table_info({table_name})")
    sqlite_columns = [col[1] for col in sqlite_cursor.fetchall()]
    
    # Map columns if needed
    if columns_mapping:
        postgres_columns = [columns_mapping.get(col, col) for col in sqlite_columns]
    else:
        postgres_columns = sqlite_columns
    
    # Insert data into PostgreSQL
    postgres_cursor = postgres_conn.cursor()
    
    # Create placeholders for the INSERT statement
    placeholders = ', '.join(['%s'] * len(postgres_columns))
    columns_str = ', '.join(postgres_columns)
    
    # Use ON CONFLICT for all tables to handle duplicates
    insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    
    try:
        # Handle data type conversions
        converted_rows = []
        for row in rows:
            converted_row = []
            for i, value in enumerate(row):
                if i < len(sqlite_columns):
                    col_name = sqlite_columns[i]
                    # Convert boolean values
                    if col_name in ['is_active', 'is_sustainable'] and value is not None:
                        converted_row.append(bool(value))
                    # Convert establishment_year "Unknown" and non-numeric values to NULL
                    elif col_name == 'establishment_year' and (value == 'Unknown' or not str(value).isdigit()):
                        converted_row.append(None)
                    else:
                        converted_row.append(value)
                else:
                    converted_row.append(value)
            converted_rows.append(converted_row)
        
        postgres_cursor.executemany(insert_query, converted_rows)
        postgres_conn.commit()
        print(f"  Migrated {len(rows)} rows successfully")
    except Exception as e:
        print(f"  Error migrating {table_name}: {e}")
        postgres_conn.rollback()
        raise

def migrate_all_tables():
    """Migrate all tables from SQLite to PostgreSQL"""
    print("Starting migration from SQLite to PostgreSQL...")
    
    # Get connections
    sqlite_conn = get_sqlite_connection()
    postgres_conn = get_postgres_connection()
    
    try:
        # List of tables to migrate
        tables_to_migrate = [
            'users',
            'travel_modes', 
            'cities',
            'hotels',
            'places',
            'travel_data',
            'co2_emissions'
        ]
        
        # Check which tables exist in SQLite
        sqlite_cursor = sqlite_conn.cursor()
        sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in sqlite_cursor.fetchall()]
        
        print(f"Found tables in SQLite: {existing_tables}")
        
        # Define column mappings for schema differences
        column_mappings = {
            'users': {'password': 'password_hash'},
            'hotels': {'hotel_name': 'name', 'place': 'city'}
        }
        
        # Migrate each table
        for table in tables_to_migrate:
            if table in existing_tables:
                try:
                    mapping = column_mappings.get(table)
                    migrate_table(sqlite_conn, postgres_conn, table, mapping)
                except Exception as e:
                    print(f"Failed to migrate {table}: {e}")
                    continue
            else:
                print(f"Table {table} not found in SQLite, skipping...")
        
        print("Migration completed!")
        
    finally:
        sqlite_conn.close()
        postgres_conn.close()

def verify_migration():
    """Verify that the migration was successful"""
    print("\nVerifying migration...")
    
    sqlite_conn = get_sqlite_connection()
    postgres_conn = get_postgres_connection()
    
    try:
        tables_to_verify = ['users', 'travel_modes', 'cities', 'hotels', 'places']
        
        for table in tables_to_verify:
            # Count rows in SQLite
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table}")
            sqlite_count = sqlite_cursor.fetchone()[0]
            
            # Count rows in PostgreSQL
            postgres_cursor = postgres_conn.cursor()
            postgres_cursor.execute(f"SELECT COUNT(*) FROM {table}")
            postgres_count = postgres_cursor.fetchone()[0]
            
            status = "✅" if sqlite_count == postgres_count else "❌"
            print(f"{status} {table}: SQLite={sqlite_count}, PostgreSQL={postgres_count}")
    
    finally:
        sqlite_conn.close()
        postgres_conn.close()

if __name__ == "__main__":
    try:
        migrate_all_tables()
        verify_migration()
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
