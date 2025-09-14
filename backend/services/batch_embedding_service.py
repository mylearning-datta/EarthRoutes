"""
OpenAI Batch API Service for Efficient Embedding Generation
Handles large-scale embedding generation using OpenAI's Batch API
"""

import openai
import json
import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import tempfile
import os
from config.settings import settings

logger = logging.getLogger(__name__)

class BatchEmbeddingService:
    """Service for generating embeddings using OpenAI Batch API"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536
        self.batch_size_limit = 50000  # OpenAI batch limit
        self.file_size_limit = 100 * 1024 * 1024  # 100MB limit
    
    def create_batch_embeddings(self, items: List[Dict], item_type: str = "hotels") -> Dict:
        """
        Create embeddings for a large number of items using OpenAI Batch API
        
        Args:
            items: List of items to generate embeddings for
            item_type: Type of items (hotels, places, etc.)
        
        Returns:
            Dict with batch job information and results
        """
        try:
            logger.info(f"Starting batch embedding generation for {len(items)} {item_type}")
            
            # Step 1: Create JSONL input file
            input_file_path = self._create_input_jsonl(items, item_type)
            logger.info(f"Created input file: {input_file_path}")
            
            # Step 2: Upload file to OpenAI
            uploaded_file = self._upload_file(input_file_path)
            logger.info(f"Uploaded file with ID: {uploaded_file.id}")
            
            # Step 3: Create batch job
            batch_job = self._create_batch_job(uploaded_file.id)
            logger.info(f"Created batch job with ID: {batch_job.id}")
            
            # Step 4: Monitor job progress
            final_job = self._monitor_batch_job(batch_job.id)
            
            # Step 5: Download and process results
            results = self._download_and_process_results(final_job.output_file_id, items)
            
            # Cleanup
            self._cleanup_files(input_file_path, uploaded_file.id)
            
            return {
                "success": True,
                "batch_job_id": batch_job.id,
                "total_items": len(items),
                "processed_items": len(results),
                "results": results,
                "item_type": item_type
            }
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_items": len(items),
                "item_type": item_type
            }
    
    def _create_input_jsonl(self, items: List[Dict], item_type: str) -> str:
        """Create JSONL input file for batch API"""
        temp_dir = tempfile.mkdtemp()
        input_file_path = os.path.join(temp_dir, f"{item_type}_embeddings_input.jsonl")
        
        with open(input_file_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(items):
                # Generate text for embedding based on item type
                if item_type == "hotels":
                    text = self._generate_hotel_embedding_text(item)
                elif item_type == "places":
                    text = self._generate_place_embedding_text(item)
                else:
                    text = str(item)
                
                # Create request object
                request = {
                    "custom_id": f"{item_type}_{item.get('id', i)}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.embedding_model,
                        "input": text,
                        "dimensions": self.embedding_dimensions
                    }
                }
                
                f.write(json.dumps(request) + '\n')
        
        return input_file_path
    
    def _generate_hotel_embedding_text(self, hotel: Dict) -> str:
        """Generate text for hotel embedding"""
        text_parts = []
        
        if hotel.get('name'):
            text_parts.append(f"Hotel: {hotel['name']}")
        
        if hotel.get('description'):
            text_parts.append(f"Description: {hotel['description']}")
        
        if hotel.get('amenities'):
            text_parts.append(f"Amenities: {hotel['amenities']}")
        
        if hotel.get('city'):
            text_parts.append(f"Location: {hotel['city']}")
        
        if hotel.get('price_range'):
            text_parts.append(f"Price: {hotel['price_range']}")
        
        if hotel.get('condition'):
            text_parts.append(f"Condition: {hotel['condition']}")
        
        return " | ".join(text_parts)
    
    def _generate_place_embedding_text(self, place: Dict) -> str:
        """Generate text for place embedding"""
        text_parts = []
        
        if place.get('name'):
            text_parts.append(f"Name: {place['name']}")
        
        if place.get('description'):
            text_parts.append(f"Description: {place['description']}")
        
        if place.get('significance'):
            text_parts.append(f"Significance: {place['significance']}")
        
        if place.get('type'):
            text_parts.append(f"Type: {place['type']}")
        
        if place.get('category'):
            text_parts.append(f"Category: {place['category']}")
        
        if place.get('city'):
            text_parts.append(f"Location: {place['city']}")
        
        if place.get('is_sustainable') and place.get('sustainability_reason'):
            text_parts.append(f"Sustainability: {place['sustainability_reason']}")
        
        if place.get('best_time_to_visit'):
            text_parts.append(f"Best time: {place['best_time_to_visit']}")
        
        return " | ".join(text_parts)
    
    def _upload_file(self, file_path: str):
        """Upload file to OpenAI"""
        with open(file_path, 'rb') as f:
            return self.openai_client.files.create(
                file=f,
                purpose="batch"
            )
    
    def _create_batch_job(self, file_id: str):
        """Create batch job"""
        return self.openai_client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/embeddings",
            completion_window="24h"
        )
    
    def _monitor_batch_job(self, batch_id: str, max_wait_time: int = 3600):
        """Monitor batch job progress"""
        logger.info(f"Monitoring batch job: {batch_id}")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            job = self.openai_client.batches.retrieve(batch_id)
            
            logger.info(f"Batch job status: {job.status}")
            
            if job.status == "completed":
                logger.info("Batch job completed successfully")
                return job
            elif job.status == "failed":
                logger.error(f"Batch job failed: {job.errors}")
                raise Exception(f"Batch job failed: {job.errors}")
            elif job.status in ["cancelled", "cancelling"]:
                logger.error("Batch job was cancelled")
                raise Exception("Batch job was cancelled")
            
            # Wait before checking again
            time.sleep(30)
        
        raise Exception("Batch job monitoring timed out")
    
    def _download_and_process_results(self, output_file_id: str, original_items: List[Dict]) -> List[Dict]:
        """Download and process batch results"""
        logger.info(f"Downloading results from file: {output_file_id}")
        
        # Download the results file
        result_file = self.openai_client.files.content(output_file_id)
        result_content = result_file.read().decode('utf-8')
        
        # Parse results
        results = []
        for line in result_content.strip().split('\n'):
            if line:
                result = json.loads(line)
                if result.get('response', {}).get('body', {}).get('data'):
                    # Extract embedding and custom_id
                    custom_id = result['custom_id']
                    embedding = result['response']['body']['data'][0]['embedding']
                    
                    # Find original item
                    item_id = custom_id.split('_', 1)[1] if '_' in custom_id else custom_id
                    original_item = next(
                        (item for item in original_items if str(item.get('id')) == item_id),
                        None
                    )
                    
                    if original_item:
                        results.append({
                            'id': original_item.get('id'),
                            'embedding': embedding,
                            'custom_id': custom_id
                        })
        
        logger.info(f"Processed {len(results)} embeddings from batch results")
        return results
    
    def _cleanup_files(self, input_file_path: str, uploaded_file_id: str):
        """Clean up temporary files"""
        try:
            # Delete local file
            if os.path.exists(input_file_path):
                os.remove(input_file_path)
                os.rmdir(os.path.dirname(input_file_path))
            
            # Delete uploaded file from OpenAI
            self.openai_client.files.delete(uploaded_file_id)
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def get_batch_job_status(self, batch_id: str) -> Dict:
        """Get status of a batch job"""
        try:
            job = self.openai_client.batches.retrieve(batch_id)
            return {
                "batch_id": batch_id,
                "status": job.status,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "failed_at": job.failed_at,
                "request_counts": job.request_counts,
                "errors": job.errors
            }
        except Exception as e:
            return {
                "batch_id": batch_id,
                "error": str(e)
            }
    
    def cancel_batch_job(self, batch_id: str) -> Dict:
        """Cancel a batch job"""
        try:
            job = self.openai_client.batches.cancel(batch_id)
            return {
                "success": True,
                "batch_id": batch_id,
                "status": job.status
            }
        except Exception as e:
            return {
                "success": False,
                "batch_id": batch_id,
                "error": str(e)
            }

# Global batch embedding service instance
batch_embedding_service = BatchEmbeddingService()
