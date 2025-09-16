"""
Semantic Search Service for Natural Language Queries
Provides AI-powered search capabilities for places and hotels
"""

from typing import List, Dict, Optional, Any
from services.vector_service import vector_service
from utils.postgres_database import postgres_db_manager
import logging

logger = logging.getLogger(__name__)

class SemanticSearchService:
    """Service for semantic search using natural language queries"""
    
    def __init__(self):
        self.vector_service = vector_service
        self.db_manager = postgres_db_manager
    
    def search_places_natural_language(self, query: str, city: Optional[str] = None, limit: int = 10) -> Dict:
        """
        Search places using natural language queries
        
        Args:
            query: Natural language search query (e.g., "historic monuments with spiritual significance")
            city: Optional city filter
            limit: Maximum number of results
        
        Returns:
            Dict with search results and metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.vector_service.generate_query_embedding(query)
            
            if not query_embedding:
                return {
                    "results": [],
                    "query": query,
                    "error": "Could not process query"
                }
            
            # Search for similar places
            if city:
                # Filter by city and use semantic search
                results = self._search_places_in_city_semantic(query_embedding, city, limit)
            else:
                # Global semantic search
                results = self.db_manager.search_similar_places(query_embedding, limit)
            
            # Enhance results with additional context
            enhanced_results = self._enhance_place_results(results, query)
            
            return {
                "results": enhanced_results,
                "query": query,
                "city_filter": city,
                "total_results": len(enhanced_results),
                "search_type": "semantic"
            }
            
        except Exception as e:
            logger.error(f"Error in semantic place search: {e}")
            return {
                "results": [],
                "query": query,
                "error": str(e)
            }
    
    def search_hotels_natural_language(self, query: str, city: Optional[str] = None, limit: int = 10) -> Dict:
        """
        Search hotels using natural language queries
        
        Args:
            query: Natural language search query (e.g., "luxury hotels with spa facilities")
            city: Optional city filter
            limit: Maximum number of results
        
        Returns:
            Dict with search results and metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.vector_service.generate_query_embedding(query)
            
            if not query_embedding:
                return {
                    "results": [],
                    "query": query,
                    "error": "Could not process query"
                }
            
            # Search for similar hotels
            if city:
                # Filter by city and use semantic search
                results = self._search_hotels_in_city_semantic(query_embedding, city, limit)
            else:
                # Global semantic search
                results = self.db_manager.search_similar_hotels(query_embedding, limit)
            
            # Enhance results with additional context
            enhanced_results = self._enhance_hotel_results(results, query)
            
            return {
                "results": enhanced_results,
                "query": query,
                "city_filter": city,
                "total_results": len(enhanced_results),
                "search_type": "semantic"
            }
            
        except Exception as e:
            logger.error(f"Error in semantic hotel search: {e}")
            return {
                "results": [],
                "query": query,
                "error": str(e)
            }
    
    def search_unified(self, query: str, city: Optional[str] = None, limit: int = 10) -> Dict:
        """
        Unified search across both places and hotels
        
        Args:
            query: Natural language search query
            city: Optional city filter
            limit: Maximum number of results per category
        
        Returns:
            Dict with combined results from places and hotels
        """
        try:
            # Search both places and hotels
            places_result = self.search_places_natural_language(query, city, limit)
            hotels_result = self.search_hotels_natural_language(query, city, limit)
            
            # Combine and rank results
            combined_results = self._combine_and_rank_results(
                places_result.get("results", []),
                hotels_result.get("results", []),
                query
            )
            
            return {
                "places": places_result.get("results", []),
                "hotels": hotels_result.get("results", []),
                "combined_ranking": combined_results,
                "query": query,
                "city_filter": city,
                "total_places": len(places_result.get("results", [])),
                "total_hotels": len(hotels_result.get("results", [])),
                "search_type": "unified_semantic"
            }
            
        except Exception as e:
            logger.error(f"Error in unified search: {e}")
            return {
                "places": [],
                "hotels": [],
                "combined_ranking": [],
                "query": query,
                "error": str(e)
            }
    
    def _search_places_in_city_semantic(self, query_embedding: List[float], city: str, limit: int) -> List[Dict]:
        """Search places in a specific city using semantic similarity"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                query = """
                SELECT id, name, city, type, significance, google_review_rating,
                       is_sustainable, sustainability_reason, description,
                       1 - (embedding <=> %s::vector) as similarity
                FROM places 
                WHERE city = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
                # Convert embedding to string format
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                cursor.execute(query, (embedding_str, city, embedding_str, limit))
                results = cursor.fetchall()
                
                # Convert to list of dicts
                columns = [desc[0] for desc in cursor.description]
                results_dict = []
                for row in results:
                    row_dict = dict(zip(columns, row))
                    # Replace any NaN values with None to prevent JSON serialization errors
                    for key, value in row_dict.items():
                        if value is not None and str(value).lower() == 'nan':
                            row_dict[key] = None
                    results_dict.append(row_dict)
                return results_dict
                
        except Exception as e:
            logger.error(f"Error searching places in city: {e}")
            return []
    
    def _search_hotels_in_city_semantic(self, query_embedding: List[float], city: str, limit: int) -> List[Dict]:
        """Search hotels in a specific city using semantic similarity"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                query = """
                SELECT id, name, city, rating, price_range, amenities, description,
                       1 - (embedding <=> %s::vector) as similarity
                FROM hotels 
                WHERE city = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
                # Convert embedding to string format
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                cursor.execute(query, (embedding_str, city, embedding_str, limit))
                results = cursor.fetchall()
                
                # Convert to list of dicts
                columns = [desc[0] for desc in cursor.description]
                results_dict = []
                for row in results:
                    row_dict = dict(zip(columns, row))
                    # Replace any NaN values with None to prevent JSON serialization errors
                    for key, value in row_dict.items():
                        if value is not None and str(value).lower() == 'nan':
                            row_dict[key] = None
                    results_dict.append(row_dict)
                return results_dict
                
        except Exception as e:
            logger.error(f"Error searching hotels in city: {e}")
            return []
    
    def _enhance_place_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Enhance place results with additional context and reasoning"""
        enhanced_results = []
        
        for result in results:
            enhanced_result = {
                **result,
                "ai_reasoning": self._generate_place_reasoning(result, query),
                "relevance_score": result.get("similarity", 0.0),
                "search_context": self._extract_search_context(query)
            }
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _enhance_hotel_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Enhance hotel results with additional context and reasoning"""
        enhanced_results = []
        
        for result in results:
            enhanced_result = {
                **result,
                "ai_reasoning": self._generate_hotel_reasoning(result, query),
                "relevance_score": result.get("similarity", 0.0),
                "search_context": self._extract_search_context(query)
            }
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _generate_place_reasoning(self, place: Dict, query: str) -> str:
        """Generate AI reasoning for why this place matches the query"""
        reasoning_parts = []
        
        similarity = place.get("similarity", 0.0)
        if similarity > 0.8:
            reasoning_parts.append("Highly relevant to your search")
        elif similarity > 0.6:
            reasoning_parts.append("Good match for your search")
        elif similarity > 0.4:
            reasoning_parts.append("Moderately relevant to your search")
        else:
            reasoning_parts.append("Somewhat relevant to your search")
        
        # Add specific reasoning based on place characteristics
        if place.get("is_sustainable") and "eco" in query.lower():
            reasoning_parts.append("Eco-friendly destination")
        
        if place.get("google_review_rating", 0) > 4.0 and "good" in query.lower():
            reasoning_parts.append("Highly rated by visitors")
        
        if place.get("type") and place["type"].lower() in query.lower():
            reasoning_parts.append(f"Matches your interest in {place['type']}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_hotel_reasoning(self, hotel: Dict, query: str) -> str:
        """Generate AI reasoning for why this hotel matches the query"""
        reasoning_parts = []
        
        similarity = hotel.get("similarity", 0.0)
        if similarity > 0.8:
            reasoning_parts.append("Highly relevant to your search")
        elif similarity > 0.6:
            reasoning_parts.append("Good match for your search")
        elif similarity > 0.4:
            reasoning_parts.append("Moderately relevant to your search")
        else:
            reasoning_parts.append("Somewhat relevant to your search")
        
        # Add specific reasoning based on hotel characteristics
        if hotel.get("rating", 0) > 4.0 and "luxury" in query.lower():
            reasoning_parts.append("High-rated luxury accommodation")
        
        if hotel.get("amenities") and "spa" in query.lower() and "spa" in hotel["amenities"].lower():
            reasoning_parts.append("Features spa facilities")
        
        if hotel.get("price_range") and "budget" in query.lower():
            reasoning_parts.append("Budget-friendly option")
        
        return ". ".join(reasoning_parts) + "."
    
    def _extract_search_context(self, query: str) -> Dict:
        """Extract search context from the query"""
        context = {
            "keywords": [],
            "intent": "general",
            "filters": []
        }
        
        query_lower = query.lower()
        
        # Extract keywords
        keywords = query.split()
        context["keywords"] = keywords
        
        # Determine intent
        if any(word in query_lower for word in ["find", "search", "look for"]):
            context["intent"] = "discovery"
        elif any(word in query_lower for word in ["recommend", "suggest", "best"]):
            context["intent"] = "recommendation"
        elif any(word in query_lower for word in ["compare", "vs", "versus"]):
            context["intent"] = "comparison"
        
        # Extract filters
        if "eco" in query_lower or "sustainable" in query_lower:
            context["filters"].append("eco_friendly")
        if "luxury" in query_lower or "premium" in query_lower:
            context["filters"].append("luxury")
        if "budget" in query_lower or "cheap" in query_lower:
            context["filters"].append("budget")
        
        return context
    
    def _combine_and_rank_results(self, places: List[Dict], hotels: List[Dict], query: str) -> List[Dict]:
        """Combine and rank results from places and hotels"""
        combined = []
        
        # Add places with type indicator
        for place in places:
            combined.append({
                **place,
                "type": "place",
                "category": "attraction"
            })
        
        # Add hotels with type indicator
        for hotel in hotels:
            combined.append({
                **hotel,
                "type": "hotel",
                "category": "accommodation"
            })
        
        # Sort by relevance score
        combined.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        
        return combined

# Global semantic search service instance
semantic_search_service = SemanticSearchService()
