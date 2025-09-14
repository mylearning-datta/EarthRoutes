"""
Vector Embedding Service for Travel App
Handles generation and management of vector embeddings for places and hotels
"""

import openai
import numpy as np
from typing import List, Dict, Optional, Any
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class VectorEmbeddingService:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"  # Cost-effective model
        self.embedding_dimensions = 1536
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a given text"""
        try:
            if not text or not text.strip():
                return None
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text.strip(),
                dimensions=self.embedding_dimensions
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_place_embedding(self, place: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding for a place based on its attributes"""
        try:
            # Combine relevant place information for embedding
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
            
            combined_text = " | ".join(text_parts)
            return self.generate_embedding(combined_text)
            
        except Exception as e:
            logger.error(f"Error generating place embedding: {e}")
            return None
    
    def generate_hotel_embedding(self, hotel: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding for a hotel based on its attributes"""
        try:
            # Combine relevant hotel information for embedding
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
            
            combined_text = " | ".join(text_parts)
            return self.generate_embedding(combined_text)
            
        except Exception as e:
            logger.error(f"Error generating hotel embedding: {e}")
            return None
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for a user query"""
        return self.generate_embedding(query)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batch"""
        try:
            if not texts:
                return []
            
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                return [None] * len(texts)
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=valid_texts,
                dimensions=self.embedding_dimensions
            )
            
            # Map embeddings back to original list positions
            embeddings = [None] * len(texts)
            valid_index = 0
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    embeddings[i] = response.data[valid_index].embedding
                    valid_index += 1
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return [None] * len(texts)

# Global vector service instance
vector_service = VectorEmbeddingService()
