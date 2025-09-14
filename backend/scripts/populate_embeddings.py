#!/usr/bin/env python3
"""
Script to populate vector embeddings for existing places and hotels
"""

import sys
from pathlib import Path
import logging

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.postgres_database import postgres_db_manager
from services.vector_service import vector_service

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def populate_place_embeddings():
    """Populate embeddings for all places"""
    logger.info("Starting to populate place embeddings...")
    
    try:
        with postgres_db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all places without embeddings
            cursor.execute("""
                SELECT id, name, description, significance, type, category, city, 
                       is_sustainable, sustainability_reason, best_time_to_visit
                FROM places 
                WHERE embedding IS NULL
            """)
            
            places = cursor.fetchall()
            logger.info(f"Found {len(places)} places without embeddings")
            
            if not places:
                logger.info("All places already have embeddings")
                return
            
            # Process places in batches
            batch_size = 10
            for i in range(0, len(places), batch_size):
                batch = places[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(places) + batch_size - 1)//batch_size}")
                
                for place in batch:
                    place_id = place[0]
                    place_data = {
                        'name': place[1],
                        'description': place[2],
                        'significance': place[3],
                        'type': place[4],
                        'category': place[5],
                        'city': place[6],
                        'is_sustainable': place[7],
                        'sustainability_reason': place[8],
                        'best_time_to_visit': place[9]
                    }
                    
                    # Generate embedding
                    embedding = vector_service.generate_place_embedding(place_data)
                    
                    if embedding:
                        # Update database
                        cursor.execute(
                            "UPDATE places SET embedding = %s WHERE id = %s",
                            (embedding, place_id)
                        )
                        logger.info(f"Updated embedding for place: {place_data['name']}")
                    else:
                        logger.warning(f"Failed to generate embedding for place: {place_data['name']}")
                
                # Commit batch
                conn.commit()
                logger.info(f"Committed batch {i//batch_size + 1}")
            
            logger.info("✅ Place embeddings populated successfully")
            
    except Exception as e:
        logger.error(f"Error populating place embeddings: {e}")
        raise

def populate_hotel_embeddings():
    """Populate embeddings for all hotels"""
    logger.info("Starting to populate hotel embeddings...")
    
    try:
        with postgres_db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all hotels without embeddings
            cursor.execute("""
                SELECT id, name, description, amenities, city, price_range, condition
                FROM hotels 
                WHERE embedding IS NULL
            """)
            
            hotels = cursor.fetchall()
            logger.info(f"Found {len(hotels)} hotels without embeddings")
            
            if not hotels:
                logger.info("All hotels already have embeddings")
                return
            
            # Process hotels in batches
            batch_size = 10
            for i in range(0, len(hotels), batch_size):
                batch = hotels[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(hotels) + batch_size - 1)//batch_size}")
                
                for hotel in batch:
                    hotel_id = hotel[0]
                    hotel_data = {
                        'name': hotel[1],
                        'description': hotel[2],
                        'amenities': hotel[3],
                        'city': hotel[4],
                        'price_range': hotel[5],
                        'condition': hotel[6]
                    }
                    
                    # Generate embedding
                    embedding = vector_service.generate_hotel_embedding(hotel_data)
                    
                    if embedding:
                        # Update database
                        cursor.execute(
                            "UPDATE hotels SET embedding = %s WHERE id = %s",
                            (embedding, hotel_id)
                        )
                        logger.info(f"Updated embedding for hotel: {hotel_data['name']}")
                    else:
                        logger.warning(f"Failed to generate embedding for hotel: {hotel_data['name']}")
                
                # Commit batch
                conn.commit()
                logger.info(f"Committed batch {i//batch_size + 1}")
            
            logger.info("✅ Hotel embeddings populated successfully")
            
    except Exception as e:
        logger.error(f"Error populating hotel embeddings: {e}")
        raise

def main():
    """Main function to populate all embeddings"""
    logger.info("🚀 Starting embedding population process...")
    
    try:
        # Populate place embeddings
        populate_place_embeddings()
        
        # Populate hotel embeddings
        populate_hotel_embeddings()
        
        logger.info("🎉 All embeddings populated successfully!")
        
    except Exception as e:
        logger.error(f"❌ Embedding population failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
