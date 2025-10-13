#!/usr/bin/env python3
"""
Script to populate vector embeddings using OpenAI Batch API
More efficient for large datasets
"""

import sys
from pathlib import Path
import logging
import time
from contextlib import nullcontext

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.postgres_database import postgres_db_manager
from services.batch_embedding_service import batch_embedding_service

# Optional progress bar
try:
    from tqdm import tqdm
    def _progress(iterable, total=None, desc=None):
        return tqdm(iterable, total=total, desc=desc)
except Exception:  # pragma: no cover
    def _progress(iterable, total=None, desc=None):
        return iterable

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def populate_place_embeddings_batch():
    """Populate embeddings for all places using Batch API"""
    logger.info("Starting batch embedding generation for places...")
    
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
                return {"success": True, "message": "No places to process"}
            
            # Convert to list of dicts
            places_data = []
            for place in places:
                places_data.append({
                    'id': place[0],
                    'name': place[1],
                    'description': place[2],
                    'significance': place[3],
                    'type': place[4],
                    'category': place[5],
                    'city': place[6],
                    'is_sustainable': place[7],
                    'sustainability_reason': place[8],
                    'best_time_to_visit': place[9]
                })
            
            # Generate embeddings using Batch API
            result = batch_embedding_service.create_batch_embeddings(places_data, "places")
            
            if result["success"]:
                # Update database with embeddings
                update_count = 0
                for item in _progress(result["results"], total=len(result["results"]), desc="Places embeddings"):
                    cursor.execute(
                        "UPDATE places SET embedding = %s WHERE id = %s",
                        (item["embedding"], item["id"])
                    )
                    update_count += 1
                
                conn.commit()
                logger.info(f"✅ Updated {update_count} place embeddings in database")
                
                return {
                    "success": True,
                    "batch_job_id": result["batch_job_id"],
                    "processed_items": update_count,
                    "total_items": len(places_data)
                }
            else:
                logger.error(f"❌ Batch embedding generation failed: {result.get('error')}")
                return result
                
    except Exception as e:
        logger.error(f"Error populating place embeddings: {e}")
        raise

def populate_hotel_embeddings_batch():
    """Populate embeddings for all hotels using Batch API"""
    logger.info("Starting batch embedding generation for hotels...")
    
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
                return {"success": True, "message": "No hotels to process"}
            
            # Convert to list of dicts
            hotels_data = []
            for hotel in hotels:
                hotels_data.append({
                    'id': hotel[0],
                    'name': hotel[1],
                    'description': hotel[2],
                    'amenities': hotel[3],
                    'city': hotel[4],
                    'price_range': hotel[5],
                    'condition': hotel[6]
                })
            
            # Generate embeddings using Batch API
            result = batch_embedding_service.create_batch_embeddings(hotels_data, "hotels")
            
            if result["success"]:
                # Update database with embeddings
                update_count = 0
                for item in _progress(result["results"], total=len(result["results"]), desc="Hotels embeddings"):
                    cursor.execute(
                        "UPDATE hotels SET embedding = %s WHERE id = %s",
                        (item["embedding"], item["id"])
                    )
                    update_count += 1
                
                conn.commit()
                logger.info(f"✅ Updated {update_count} hotel embeddings in database")
                
                return {
                    "success": True,
                    "batch_job_id": result["batch_job_id"],
                    "processed_items": update_count,
                    "total_items": len(hotels_data)
                }
            else:
                logger.error(f"❌ Batch embedding generation failed: {result.get('error')}")
                return result
                
    except Exception as e:
        logger.error(f"Error populating hotel embeddings: {e}")
        raise

def check_batch_job_status(batch_id: str):
    """Check the status of a batch job"""
    logger.info(f"Checking status of batch job: {batch_id}")
    
    try:
        status = batch_embedding_service.get_batch_job_status(batch_id)
        logger.info(f"Batch job status: {status}")
        return status
    except Exception as e:
        logger.error(f"Error checking batch job status: {e}")
        return {"error": str(e)}

def main():
    """Main function to populate all embeddings using Batch API"""
    logger.info("🚀 Starting batch embedding population process...")
    
    try:
        # Check if we want to process specific batch job
        if len(sys.argv) > 1 and sys.argv[1] == "status":
            if len(sys.argv) > 2:
                batch_id = sys.argv[2]
                status = check_batch_job_status(batch_id)
                print(json.dumps(status, indent=2))
                return
            else:
                logger.error("Please provide batch ID for status check")
                return
        
        # Populate place embeddings
        logger.info("📍 Processing places...")
        places_result = populate_place_embeddings_batch()
        
        if places_result["success"]:
            logger.info(f"✅ Places completed: {places_result}")
        else:
            logger.error(f"❌ Places failed: {places_result}")
        
        # Populate hotel embeddings
        logger.info("🏨 Processing hotels...")
        hotels_result = populate_hotel_embeddings_batch()
        
        if hotels_result["success"]:
            logger.info(f"✅ Hotels completed: {hotels_result}")
        else:
            logger.error(f"❌ Hotels failed: {hotels_result}")
        
        # Summary
        logger.info("🎉 Batch embedding population completed!")
        logger.info(f"Places: {places_result}")
        logger.info(f"Hotels: {hotels_result}")
        
    except Exception as e:
        logger.error(f"❌ Batch embedding population failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
