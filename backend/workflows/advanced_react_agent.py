from typing import Dict, List, Any, Optional, Type
import logging
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from config.settings import settings
from tools.travel_tools import travel_tools
from services.vector_service import vector_service
from utils.postgres_database import postgres_db_manager
import json
import re
from datetime import datetime

# Module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def _safe_json_loads(maybe_json: str) -> dict:
    """Robustly parse JSON from agent tool inputs that may include backticks/code fences.
    - Strips wrapping backticks and ```json fences
    - Extracts the first {...} block if present
    - Returns {} on failure
    """
    try:
        if not isinstance(maybe_json, str):
            return {}
        s = maybe_json.strip()
        # Remove single-line backticks
        if s.startswith('`') and s.endswith('`'):
            s = s[1:-1].strip()
        # Remove fenced code blocks ``` or ```json
        if s.startswith("```"):
            # Strip first fence
            s = s.split("\n", 1)[1] if "\n" in s else s.replace("```", "", 1)
            # Strip trailing fence
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
            s = s.strip()
        # If still not pure JSON, try to extract first {...}
        if not (s.startswith('{') and s.endswith('}')):
            import re
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                s = m.group(0)
        import json as _json
        try:
            return _json.loads(s) if s else {}
        except Exception:
            # Fallback: parse Python-style single-quoted dicts
            try:
                import ast as _ast
                return _ast.literal_eval(s) if s else {}
            except Exception:
                return {}
    except Exception:
        return {}

class AdvancedTravelPlanningTool(BaseTool):
    """Advanced tool for comprehensive travel planning with reasoning"""
    
    name: str = "advanced_travel_planner"
    description: str = """Use this tool for comprehensive travel planning including route analysis, CO2 calculations, and eco-friendly recommendations.
    Input should be a JSON string with keys: source, destination, preferences (optional).
    preferences can include: budget, time_constraint, eco_priority, comfort_level"""
    
    def _run(self, query: str) -> str:
        try:
            logger.info("Enter AdvancedTravelPlanningTool._run")
            data = _safe_json_loads(query)
            source = data.get("source")
            destination = data.get("destination")
            preferences = data.get("preferences", {})
            
            if not source or not destination:
                return json.dumps({"error": "Source and destination are required"}, indent=2)
            
            # Get comprehensive travel data
            travel_data = travel_tools.get_travel_suggestions(source, destination)
            
            # Add reasoning and recommendations
            analysis = self._analyze_travel_options(travel_data, preferences)
            
            result = {
                "travel_data": travel_data,
                "analysis": analysis,
                "recommendations": self._generate_recommendations(travel_data, preferences),
                "timestamp": datetime.now().isoformat()
            }
            
            output = json.dumps(result, indent=2)
            logger.info("Exit AdvancedTravelPlanningTool._run")
            return output
                
        except Exception as e:
            logger.exception("Error in AdvancedTravelPlanningTool._run")
            return json.dumps({"error": str(e)}, indent=2)
    
    def _analyze_travel_options(self, travel_data: Dict, preferences: Dict) -> Dict:
        """Enhanced analysis using semantic similarity and AI-powered recommendations"""
        logger.info("Enter AdvancedTravelPlanningTool._analyze_travel_options")
        analysis = {
            "eco_friendly_ranking": [],
            "time_efficient_ranking": [],
            "cost_effective_ranking": [],
            "comfort_ranking": [],
            "semantic_ranking": [],
            "personalized_recommendations": []
        }
        
        if "travel_comparison" in travel_data and "options" in travel_data["travel_comparison"]:
            options = travel_data["travel_comparison"]["options"]
            
            # Traditional sorting with type-safe keys
            def _co2_key(opt: Dict) -> float:
                try:
                    return float(opt.get("co2Emissions", float("inf")))
                except Exception:
                    return float("inf")
            analysis["eco_friendly_ranking"] = sorted(options, key=_co2_key)
            analysis["time_efficient_ranking"] = sorted(options, key=lambda x: self._parse_duration(x.get("duration", 24)))
            
            # Enhanced AI-powered analysis
            analysis["semantic_ranking"] = self._semantic_travel_ranking(options, preferences)
            analysis["personalized_recommendations"] = self._generate_ai_recommendations(options, preferences)
            
        logger.info("Exit AdvancedTravelPlanningTool._analyze_travel_options")
        return analysis
    
    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string to hours for comparison"""
        try:
            if isinstance(duration_str, (int, float)):
                return float(duration_str)
            if not isinstance(duration_str, str):
                return 24
            if "days" in duration_str:
                parts = duration_str.split()
                days = int(parts[0])
                hours = int(parts[2]) if len(parts) > 2 else 0
                return days * 24 + hours
            elif "hrs" in duration_str:
                parts = duration_str.split()
                try:
                    hours = int(parts[0])
                except Exception:
                    hours = 0
                try:
                    minutes = int(parts[2]) if len(parts) > 2 and "mins" in parts[2] else 0
                except Exception:
                    minutes = 0
                return hours + minutes / 60
            elif "mins" in duration_str:
                try:
                    minutes = int(duration_str.split()[0])
                except Exception:
                    minutes = 0
                return minutes / 60
            else:
                return 24  # Default fallback
        except:
            return 24
    
    def _semantic_travel_ranking(self, options: List[Dict], preferences: Dict) -> List[Dict]:
        """Rank travel options using semantic similarity"""
        try:
            logger.info("Enter AdvancedTravelPlanningTool._semantic_travel_ranking")
            # Create preference context for semantic matching
            preference_context = self._build_preference_context(preferences)
            
            # Generate embedding for user preferences
            preference_embedding = vector_service.generate_query_embedding(preference_context)
            
            if not preference_embedding:
                return options  # Fallback to original order
            
            # Score each option based on semantic similarity
            scored_options = []
            for option in options:
                # Create option description for embedding
                option_description = self._build_option_description(option)
                option_embedding = vector_service.generate_query_embedding(option_description)
                
                if option_embedding:
                    similarity = vector_service.calculate_similarity(preference_embedding, option_embedding)
                    scored_options.append({
                        **option,
                        "semantic_score": similarity,
                        "ai_reasoning": self._generate_ai_reasoning(option, preferences, similarity)
                    })
                else:
                    scored_options.append({**option, "semantic_score": 0.0})
            
            # Sort by semantic similarity
            result = sorted(scored_options, key=lambda x: x["semantic_score"], reverse=True)
            logger.info("Exit AdvancedTravelPlanningTool._semantic_travel_ranking")
            return result
            
        except Exception as e:
            print(f"Error in semantic ranking: {e}")
            return options
    
    def _generate_ai_recommendations(self, options: List[Dict], preferences: Dict) -> List[Dict]:
        """Generate AI-powered personalized recommendations"""
        try:
            logger.info("Enter AdvancedTravelPlanningTool._generate_ai_recommendations")
            recommendations = []
            
            # Analyze user preferences for context
            preference_context = self._build_preference_context(preferences)
            
            # Find similar places/hotels in database for context
            similar_places = self._find_similar_database_items(preference_context)
            
            # Generate recommendations based on semantic analysis
            for option in options[:3]:  # Top 3 options
                recommendation = {
                    "option": option,
                    "ai_analysis": self._analyze_option_with_ai(option, preferences, similar_places),
                    "personalization_score": self._calculate_personalization_score(option, preferences),
                    "contextual_insights": self._generate_contextual_insights(option, similar_places)
                }
                recommendations.append(recommendation)
            
            logger.info("Exit AdvancedTravelPlanningTool._generate_ai_recommendations")
            return recommendations
            
        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
            return []
    
    def _build_preference_context(self, preferences: Dict) -> str:
        """Build a natural language context from user preferences"""
        context_parts = []
        
        if preferences.get("eco_priority"):
            context_parts.append("environmentally conscious travel with low carbon footprint")
        
        if preferences.get("budget"):
            budget = preferences["budget"]
            if budget < 1000:
                context_parts.append("budget-friendly travel options")
            elif budget > 5000:
                context_parts.append("luxury travel experiences")
            else:
                context_parts.append("mid-range travel options")
        
        if preferences.get("time_constraint"):
            time = preferences["time_constraint"]
            if time < 24:
                context_parts.append("quick and efficient travel")
            else:
                context_parts.append("leisurely travel with time to explore")
        
        if preferences.get("comfort_level"):
            comfort = preferences["comfort_level"]
            if comfort == "high":
                context_parts.append("comfortable and premium travel experience")
            elif comfort == "low":
                context_parts.append("adventure and budget travel")
        
        return " ".join(context_parts) if context_parts else "general travel preferences"
    
    def _build_option_description(self, option: Dict) -> str:
        """Build a description of travel option for embedding"""
        description_parts = []
        
        if option.get("mode"):
            description_parts.append(f"travel mode: {option['mode']}")
        
        if option.get("duration"):
            description_parts.append(f"duration: {option['duration']}")
        
        if option.get("co2Emissions") is not None:
            try:
                emissions = float(option.get("co2Emissions"))
                if emissions < 1:
                    description_parts.append("very low carbon emissions")
                elif emissions < 5:
                    description_parts.append("low carbon emissions")
                elif emissions < 20:
                    description_parts.append("moderate carbon emissions")
                else:
                    description_parts.append("high carbon emissions")
            except Exception:
                pass
        
        if option.get("cost"):
            description_parts.append(f"cost: {option['cost']}")
        
        return " ".join(description_parts)
    
    def _generate_ai_reasoning(self, option: Dict, preferences: Dict, similarity: float) -> str:
        """Generate AI reasoning for why this option matches user preferences"""
        reasoning_parts = []
        
        if similarity > 0.8:
            reasoning_parts.append("Highly matches your preferences")
        elif similarity > 0.6:
            reasoning_parts.append("Good match for your preferences")
        elif similarity > 0.4:
            reasoning_parts.append("Moderately matches your preferences")
        else:
            reasoning_parts.append("Limited match with your preferences")
        
        # Add specific reasoning based on option characteristics
        if option.get("co2Emissions", 0) < 1 and preferences.get("eco_priority"):
            reasoning_parts.append("Excellent for eco-conscious travel")
        
        if option.get("duration") and preferences.get("time_constraint"):
            reasoning_parts.append("Fits your time constraints")
        
        return ". ".join(reasoning_parts) + "."
    
    def _find_similar_database_items(self, preference_context: str) -> Dict:
        """Find similar places and hotels in database for context"""
        try:
            logger.info("Enter AdvancedTravelPlanningTool._find_similar_database_items")
            # Generate embedding for preference context
            query_embedding = vector_service.generate_query_embedding(preference_context)
            
            if not query_embedding:
                return {"places": [], "hotels": []}
            
            # Search for similar items
            similar_places = postgres_db_manager.search_similar_places(query_embedding, limit=5)
            similar_hotels = postgres_db_manager.search_similar_hotels(query_embedding, limit=5)
            
            result = {
                "places": similar_places,
                "hotels": similar_hotels
            }
            logger.info("Exit AdvancedTravelPlanningTool._find_similar_database_items")
            return result
            
        except Exception as e:
            print(f"Error finding similar database items: {e}")
            return {"places": [], "hotels": []}
    
    def _analyze_option_with_ai(self, option: Dict, preferences: Dict, similar_items: Dict) -> str:
        """Analyze travel option using AI and database context"""
        analysis_parts = []
        
        # Analyze based on similar items in database
        if similar_items.get("places"):
            analysis_parts.append("Similar to popular destinations in our database")
        
        if similar_items.get("hotels"):
            analysis_parts.append("Matches preferences of similar travelers")
        
        # Analyze option characteristics
        if option.get("co2Emissions", 0) < 1:
            analysis_parts.append("Environmentally sustainable choice")
        
        if option.get("duration"):
            duration_hours = self._parse_duration(option["duration"])
            if duration_hours < 2:
                analysis_parts.append("Quick and efficient option")
            elif duration_hours > 12:
                analysis_parts.append("Leisurely travel experience")
        
        return ". ".join(analysis_parts) + "." if analysis_parts else "Standard travel option."
    
    def _calculate_personalization_score(self, option: Dict, preferences: Dict) -> float:
        """Calculate how well this option matches user preferences"""
        score = 0.5  # Base score
        
        # Eco-friendly scoring
        if preferences.get("eco_priority") and option.get("co2Emissions", 0) < 5:
            score += 0.3
        
        # Time constraint scoring
        if preferences.get("time_constraint") and option.get("duration"):
            duration_hours = self._parse_duration(option["duration"])
            time_preference = preferences["time_constraint"]
            if abs(duration_hours - time_preference) < 2:
                score += 0.2
        
        # Budget scoring (simplified)
        if preferences.get("budget") and option.get("cost"):
            # This would need actual cost data to be accurate
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_contextual_insights(self, option: Dict, similar_items: Dict) -> List[str]:
        """Generate contextual insights based on similar database items"""
        insights = []
        
        if similar_items.get("places"):
            insights.append(f"Similar to {len(similar_items['places'])} popular destinations")
        
        if similar_items.get("hotels"):
            insights.append(f"Matches preferences of travelers who chose {len(similar_items['hotels'])} similar accommodations")
        
        # Add insights based on option characteristics
        if option.get("co2Emissions", 0) < 1:
            insights.append("This option aligns with sustainable travel trends")
        
        return insights
    
    def _generate_recommendations(self, travel_data: Dict, preferences: Dict) -> List[Dict]:
        """Generate personalized recommendations"""
        logger.info("Enter AdvancedTravelPlanningTool._generate_recommendations")
        recommendations = []
        
        if "travel_comparison" in travel_data and "options" in travel_data["travel_comparison"]:
            options = travel_data["travel_comparison"]["options"]
            
            # Eco-friendly recommendation
            eco_option = min(options, key=lambda x: x["co2Emissions"])
            recommendations.append({
                "type": "eco_friendly",
                "title": "Most Eco-Friendly Option",
                "option": eco_option,
                "reasoning": f"Lowest CO2 emissions at {eco_option['co2Emissions']:.3f} kg, requiring {eco_option['treesNeeded']} trees to offset"
            })
            
            # Time-efficient recommendation
            time_option = min(options, key=lambda x: self._parse_duration(x["duration"]))
            recommendations.append({
                "type": "time_efficient",
                "title": "Fastest Option",
                "option": time_option,
                "reasoning": f"Shortest travel time of {time_option['duration']}"
            })
            
            # Balanced recommendation (optimize tradeoff between CO2 and duration, prefer electric train on ties)
            if len(options) >= 1:
                # Prepare normalized metrics
                durations_hours = [self._parse_duration(opt["duration"]) for opt in options]
                co2_values = [opt["co2Emissions"] for opt in options]
                min_dur, max_dur = min(durations_hours), max(durations_hours)
                min_co2, max_co2 = min(co2_values), max(co2_values)

                def normalize(value: float, lo: float, hi: float) -> float:
                    if hi <= lo:
                        return 0.0
                    return (value - lo) / (hi - lo)

                # Equal weights for balanced tradeoff
                scores = []
                for idx, opt in enumerate(options):
                    nd = normalize(durations_hours[idx], min_dur, max_dur)
                    nc = normalize(co2_values[idx], min_co2, max_co2)
                    score = 0.5 * nd + 0.5 * nc
                    scores.append(score)

                # Baseline best
                best_idx = min(range(len(options)), key=lambda i: scores[i])
                best_score = scores[best_idx]

                # Prefer electric train if within epsilon of best
                epsilon = 0.1
                electric_idx = next((i for i, opt in enumerate(options) if opt.get("id") == "train_electric" or opt.get("name") == "Electric Train"), None)
                if electric_idx is not None and scores[electric_idx] <= best_score + epsilon:
                    best_idx = electric_idx

                balanced_option = options[best_idx]
                recommendations.append({
                    "type": "balanced",
                    "title": "Balanced Option",
                    "option": balanced_option,
                    "reasoning": f"Good balance between time ({balanced_option['duration']}) and environmental impact ({balanced_option['co2Emissions']:.3f} kg CO2)"
                })
        
        # Add sustainable places recommendation
        if "sustainable_places" in travel_data and travel_data["sustainable_places"]:
            sustainable_places = travel_data["sustainable_places"]
            recommendations.append({
                "type": "sustainable_destinations",
                "title": "Eco-Friendly Places to Visit",
                "places": sustainable_places[:5],  # Top 5 sustainable places
                "reasoning": f"Found {len(sustainable_places)} sustainable places in your destination. These locations have minimal environmental impact and promote eco-friendly tourism."
            })
        
        logger.info("Exit AdvancedTravelPlanningTool._generate_recommendations")
        return recommendations
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class EnvironmentalImpactTool(BaseTool):
    """Tool for detailed environmental impact analysis"""
    
    name: str = "environmental_analyzer"
    description: str = """Use this tool to analyze environmental impact of travel choices and suggest offsetting strategies.
    Input should be a JSON string with keys: travel_mode, distance_km, or route_data"""
    
    def _run(self, query: str) -> str:
        try:
            logger.info("Enter EnvironmentalImpactTool._run")
            data = _safe_json_loads(query)
            
            if "route_data" in data:
                # Analyze entire route
                route_data = data["route_data"]
                analysis = self._analyze_route_impact(route_data)
            elif "travel_mode" in data and "distance_km" in data:
                # Analyze single mode
                co2_data = travel_tools.calculate_co2_emissions(data["distance_km"], data["travel_mode"])
                analysis = self._analyze_single_mode_impact(co2_data, data["distance_km"])
            else:
                return json.dumps({"error": "Invalid input format"}, indent=2)
            
            output = json.dumps(analysis, indent=2)
            logger.info("Exit EnvironmentalImpactTool._run")
            return output
                
        except Exception as e:
            logger.exception("Error in EnvironmentalImpactTool._run")
            return json.dumps({"error": str(e)}, indent=2)
    
    def _analyze_route_impact(self, route_data: Dict) -> Dict:
        """Analyze environmental impact of a route"""
        logger.info("Enter EnvironmentalImpactTool._analyze_route_impact")
        analysis = {
            "total_impact": {},
            "comparison": {},
            "offsetting_strategies": [],
            "eco_alternatives": []
        }
        
        if "options" in route_data:
            options = route_data["options"]
            
            # Calculate total impact
            total_co2 = sum(option["co2Emissions"] for option in options)
            total_trees = sum(option["treesNeeded"] for option in options)
            
            analysis["total_impact"] = {
                "total_co2_kg": total_co2,
                "total_trees_needed": total_trees,
                "carbon_footprint_category": self._categorize_footprint(total_co2)
            }
            
            # Generate offsetting strategies
            analysis["offsetting_strategies"] = self._generate_offsetting_strategies(total_co2)
            
            # Suggest eco alternatives
            analysis["eco_alternatives"] = self._suggest_eco_alternatives(options)
        
        logger.info("Exit EnvironmentalImpactTool._analyze_route_impact")
        return analysis
    
    def _analyze_single_mode_impact(self, co2_data: Dict, distance_km: float) -> Dict:
        """Analyze impact of a single travel mode"""
        logger.info("Enter EnvironmentalImpactTool._analyze_single_mode_impact")
        result = {
            "emissions": co2_data,
            "impact_category": self._categorize_footprint(co2_data["total_emissions"]),
            "offsetting_strategies": self._generate_offsetting_strategies(co2_data["total_emissions"]),
            "distance_km": distance_km
        }
        logger.info("Exit EnvironmentalImpactTool._analyze_single_mode_impact")
        return result
    
    def _categorize_footprint(self, co2_kg: float) -> str:
        """Categorize carbon footprint"""
        if co2_kg < 1:
            return "Very Low"
        elif co2_kg < 5:
            return "Low"
        elif co2_kg < 20:
            return "Medium"
        elif co2_kg < 50:
            return "High"
        else:
            return "Very High"
    
    def _generate_offsetting_strategies(self, co2_kg: float) -> List[Dict]:
        """Generate carbon offsetting strategies"""
        strategies = []
        
        trees_needed = max(1, round(co2_kg / 22))
        
        strategies.append({
            "type": "tree_planting",
            "description": f"Plant {trees_needed} trees",
            "impact": f"Offset {co2_kg:.2f} kg CO2 over 1 year",
            "cost_estimate": f"${trees_needed * 10}-${trees_needed * 25}"
        })
        
        if co2_kg > 10:
            strategies.append({
                "type": "renewable_energy",
                "description": "Invest in renewable energy credits",
                "impact": f"Offset {co2_kg:.2f} kg CO2",
                "cost_estimate": f"${co2_kg * 0.5:.2f}-${co2_kg * 2:.2f}"
            })
        
        return strategies
    
    def _suggest_eco_alternatives(self, options: List[Dict]) -> List[Dict]:
        """Suggest eco-friendly alternatives"""
        logger.info("Enter EnvironmentalImpactTool._suggest_eco_alternatives")
        alternatives = []
        
        # Find the most eco-friendly options
        eco_options = sorted(options, key=lambda x: x["co2Emissions"])[:3]
        
        for option in eco_options:
            if option["co2Emissions"] < 1:  # Very low emissions
                alternatives.append({
                    "option": option,
                    "reason": "Excellent environmental choice",
                    "benefits": ["Zero or minimal CO2 emissions", "Promotes active transportation", "Health benefits"]
                })
        
        logger.info("Exit EnvironmentalImpactTool._suggest_eco_alternatives")
        return alternatives
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class CityInformationTool(BaseTool):
    """Tool for getting city information and available cities"""
    
    name: str = "city_info"
    description: str = """Use this tool to get information about available cities, hotels, and attractions.
    Input must be a JSON string with keys: action, location (preferred) or city (legacy).
    action must be EXACTLY one of: 'list_cities', 'get_hotels', 'get_attractions'.
    Do NOT invent new action names. Use these exact strings.

    When to use which action:
    - 'list_cities': Use when you need the full list of supported cities. No 'location' is required.
    - 'get_hotels': Use to fetch hotels for a given location. Provide 'location' (city/state/zone). Returns 'hotels'.
    - 'get_attractions': Use to fetch places/attractions for a given location. Provide 'location' (city/state/zone). Returns 'attractions'.

    Notes:
    - Prefer 'location' over 'city'. Matching is case-insensitive and accepts city, state, or zone.
    - If 'action' is omitted but 'location' is provided, the tool defaults to 'get_attractions'."""
    
    def _run(self, query: str) -> str:
        try:
            logger.info("Enter CityInformationTool._run")
            data = _safe_json_loads(query)
            action_raw = data.get("action")
            city = data.get("city")
            location = data.get("location") or city
            # Normalize and alias actions; default to attractions when location is given
            action = (action_raw or "").strip().lower()
            if not action and location:
                action = "get_attractions"
            if action in {"list_sustainable_places", "list_places", "get_places", "attractions", "list_attractions"}:
                action = "get_attractions"
            elif action in {"hotels", "list_hotels"}:
                action = "get_hotels"
            logger.info(f"CityInformationTool request: action={action or action_raw} location={location}")
            
            if action == "list_cities":
                cities = travel_tools.get_available_cities()
                output = json.dumps({"cities": cities}, indent=2)
                try:
                    logger.info(f"CityInformationTool list_cities: count={len(cities)} sample={cities[:5]}")
                except Exception:
                    pass
                logger.info("Exit CityInformationTool._run (list_cities)")
                return output
            elif action == "get_hotels" and location:
                from utils.postgres_database import postgres_db_manager as db_manager
                # Try flexible location resolution: city -> state -> zone
                hotels = db_manager.get_hotels_by_location(location)
                output = json.dumps({"hotels": hotels}, indent=2)
                try:
                    hotel_names = [h.get("name") for h in hotels[:5]] if isinstance(hotels, list) else []
                    logger.info(f"CityInformationTool get_hotels: location={location} count={len(hotels)} sample={hotel_names}")
                except Exception:
                    pass
                logger.info("Exit CityInformationTool._run (get_hotels)")
                return output
            elif action == "get_attractions" and location:
                from utils.postgres_database import postgres_db_manager as db_manager
                # Try flexible location resolution: city -> state -> zone
                places = db_manager.get_places_by_location(location)
                output = json.dumps({"attractions": places}, indent=2)
                try:
                    place_names = [p.get("name") for p in places[:5]] if isinstance(places, list) else []
                    logger.info(f"CityInformationTool get_attractions: location={location} count={len(places)} sample={place_names}")
                except Exception:
                    pass
                logger.info("Exit CityInformationTool._run (get_attractions)")
                return output
            else:
                return json.dumps({"error": "Invalid action or missing city", "received_action": action_raw, "supported_actions": ["list_cities", "get_hotels", "get_attractions"]}, indent=2)
                
        except Exception as e:
            logger.exception("Error in CityInformationTool._run")
            return json.dumps({"error": str(e)}, indent=2)
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class FineTunedServiceLLM(LLM):
    """LangChain LLM wrapper around our fine-tuned model service.
    This lets us reuse the same ReAct toolchain while swapping only the LLM backend.
    Note: LLM is a Pydantic model; fields must be declared on the class, not set in __init__.
    """
    max_new_tokens: int = 1500
    # 'variant' accepts 'community' | 'finetuned' | None (None => env defaults)
    variant: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "finetuned_service"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None, **kwargs: Any) -> str:  # type: ignore[override]
        """Generate next output token(s) for the ReAct prompt using the MLX-backed model.
        This preserves the ReAct format (Thought/Action/Observation) so tools can run.
        """
        try:
            logger.info("Enter FineTunedServiceLLM._call")
            from services.finetuned_model_service import get_finetuned_model_service
            # Normalize variant and fetch the appropriate loaded service instance
            def norm_variant(v: Optional[str]) -> Optional[str]:
                v = (v or "").strip().lower()
                if v in {"community", "finetuned"}:
                    return v
                return None

            desired_variant = norm_variant(self.variant)
            svc = get_finetuned_model_service(desired_variant)

            # Prefer MLX generation path for consistent, low-latency token gen
            if getattr(svc, "using_mlx", False) and getattr(svc, "model", None) is not None and getattr(svc, "tokenizer", None) is not None:
                try:
                    # Import locally to avoid hard dependency if MLX not installed
                    from mlx_lm import generate as mlx_generate  # type: ignore
                except Exception as _e:
                    return f"[finetuned_service_error] MLX not available: {_e}"

                # Only stop on Observation to preserve 'Final Answer:' for the agent parser
                observation_token: str = "\nObservation:"
                stop_markers: List[str] = [observation_token]

                try:
                    text = mlx_generate(
                        svc.model,
                        svc.tokenizer,
                        prompt=prompt,
                        max_tokens=self.max_new_tokens,
                        stop=stop_markers,
                        verbose=False
                    )
                except TypeError:
                    # Older mlx_generate without stop support
                    text = mlx_generate(
                        svc.model,
                        svc.tokenizer,
                        prompt=prompt,
                        max_tokens=self.max_new_tokens,
                        verbose=False
                    )

                # Some versions return full prompt + completion; trim if needed
                completion = text[len(prompt):] if text.startswith(prompt) else text
                # Ensure completion does not include stop markers (trim at Observation only)
                try:
                    idx = completion.find(observation_token)
                    if idx != -1:
                        completion = completion[:idx].rstrip()
                except Exception:
                    pass
                logger.info("Exit FineTunedServiceLLM._call (MLX path)")
                return completion

            # Fallback: use the Transformers generate through the service if available
            try:
                import torch  # type: ignore
            except Exception:
                return "[finetuned_service_error] No generation backend available"

            if getattr(svc, "model", None) is None or getattr(svc, "tokenizer", None) is None:
                return "[finetuned_service_error] Model not loaded"

            inputs = svc.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = svc.model.generate(
                    **inputs,
                    max_new_tokens=max(128, min(self.max_new_tokens, 1024)),
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                    pad_token_id=svc.tokenizer.eos_token_id,
                    eos_token_id=svc.tokenizer.eos_token_id,
                    use_cache=True
                )
            full = svc.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = full[len(prompt):] if full.startswith(prompt) else full
            # Apply stop tokens client-side if provided; avoid trimming Final Answer content
            if stop:
                for token in stop:
                    if token and token in completion and token.strip() == "Observation:":
                        completion = completion.split(token, 1)[0]
            print("completion=====", completion)
            logger.info("Exit FineTunedServiceLLM._call (Transformers path)")
            return completion
        except Exception as e:
            logger.exception("Error in FineTunedServiceLLM._call")
            return f"[finetuned_service_error] {e}"


def create_advanced_react_agent(custom_llm: Optional[Any] = None):
    """Create an advanced ReAct agent with memory and enhanced reasoning"""
    
    logger.info("Enter create_advanced_react_agent")
    if custom_llm is not None:
        llm = custom_llm
    else:
        # Initialize the LLM with OpenAI (default path)
        # Some OpenAI o-series models (e.g., o3/o4 family) only support the default temperature (1).
        _model = settings.AGENT_MODEL
        _temp = settings.AGENT_TEMPERATURE
        if _model and (_model.startswith("o3") or _model.startswith("o4")):
            _temp = 1.0
        llm = ChatOpenAI(
            model=_model,
            temperature=_temp,
            max_tokens=settings.AGENT_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY
        )
    
    # Create advanced tools
    tools = [
        AdvancedTravelPlanningTool(),
        EnvironmentalImpactTool(),
        CityInformationTool()  # From the previous implementation
    ]
    
    # Create memory for conversation context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Enhanced ReAct prompt with reasoning capabilities (aligned with finetuned model prompt)
    react_prompt = PromptTemplate.from_template("""
You are a helpful travel and sustainability assistant. You provide personalized, accurate, and environmentally conscious travel advice. 

You have access to the following tools:
{tools}

IMPORTANT: Always include ALL the provided context details in your response:
- If distance information is provided, mention the exact distance and travel time
- If CO2 emissions data is provided, show the comparison for different travel modes with specific emission values
- If places are suggested, include their names, locations, and descriptions
- If hotels are suggested, include their names, ratings, and amenities
- Always provide specific, actionable recommendations with concrete details

Formatting requirement: Whenever you present options, lists, alternatives, itineraries, recommendations, or comparisons, format them as bulleted points.

Use the provided context to give comprehensive, detailed responses that include all relevant information.

You should use a systematic approach to solve travel planning problems:

1. **Understand**: Analyze the user's request and identify key requirements
2. **Plan**: Determine what information you need and which tools to use
3. **Execute**: Use tools to gather data and perform calculations
4. **Analyze**: Process the results and identify patterns/insights
5. **Recommend**: Provide personalized recommendations with clear reasoning
6. **Explain**: Help the user understand the environmental and practical implications

Formatting requirement: Whenever you present options, lists, alternatives, itineraries, recommendations, or comparisons, format them as bulleted points.

IMPORTANT INSTRUCTIONS FOR YOUR RESPONSE:
1. Start your response by mentioning the travel distance and CO2 emissions if provided
2. Include specific place names, ratings, and descriptions from the context
3. Include specific hotel names, ratings, and amenities from the context
4. Always mention the exact CO2 values for different travel modes
5. Provide actionable recommendations with concrete details
6. Present any options, alternatives, or recommendations as bulleted points
7. For eco-friendly or sustainable place requests, only include places where is_sustainable is true in CityInformationTool results; do NOT output places where is_sustainable is false.
8. If a tool returns empty or insufficient data, DO NOT call the tool again with the same parameters. Immediately proceed to Final Answer using best-effort general knowledge and any available context.
9. Avoid repeated tool calls for the same query. After one successful tool call, if you can answer the user, write the Final Answer.
10. For intercity or long-distance travel, do NOT recommend walking or bicycling as the primary mode. Prefer practical motorized modes (e.g., electric train, shared bus) when selecting the most sustainable option.

Use the following format:

Question: the input question you must answer
Thought: I need to think step by step about this problem. Let me break it down:
1. What is the user asking for?
2. What information do I need?
3. Which tools should I use?
4. What analysis should I perform?

Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: Based on this result, I can see that... Now I should...
Action: the next action to take
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to provide a comprehensive answer. Let me synthesize the findings and provide recommendations.
Final Answer: [Provide a detailed, well-reasoned response with specific recommendations, environmental impact analysis, and actionable advice. Present any options, alternatives, comparisons, or recommendations as bulleted points]

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
    
    # Create the advanced ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    
    # Create agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        max_execution_time=settings.AGENT_MAX_EXECUTION_SECS,
        early_stopping_method=settings.AGENT_EARLY_STOPPING_METHOD,
        return_intermediate_steps=True
    )
    
    logger.info("Exit create_advanced_react_agent")
    return agent_executor

def process_travel_query_advanced_react(query: str) -> Dict[str, Any]:
    """Process a travel query using advanced ReAct agent"""
    try:
        logger.info("Enter process_travel_query_advanced_react")
        # Create agent
        agent = create_advanced_react_agent()
        
        # Process the query
        result = agent.invoke({"input": query})
        
        output = {
            "response": result["output"],
            "travel_data": None,  # Advanced agent handles data internally
            "error": None,
            "intermediate_steps": result.get("intermediate_steps", [])
        }
        logger.info("Exit process_travel_query_advanced_react")
        return output
        
    except Exception as e:
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
            "travel_data": None,
            "error": str(e),
            "intermediate_steps": []
        }


def process_travel_query_advanced_react_finetuned(query: str, variant: Optional[str] = None) -> Dict[str, Any]:
    """Process a travel query using the same ReAct toolchain but with the fine-tuned LLM backend."""
    try:
        logger.info("Enter process_travel_query_advanced_react_finetuned")
        # Create agent with finetuned service LLM; pass variant (community|finetuned|None)
        ft_llm = FineTunedServiceLLM(max_new_tokens=settings.FINETUNED_MAX_NEW_TOKENS, variant=variant)
        agent = create_advanced_react_agent(custom_llm=ft_llm)
        result = agent.invoke({"input": query})
        output = {
            "response": result["output"],
            "travel_data": None,
            "error": None,
            "intermediate_steps": result.get("intermediate_steps", [])
        }
        logger.info("Exit process_travel_query_advanced_react_finetuned")
        return output
    except Exception as e:
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
            "travel_data": None,
            "error": str(e),
            "intermediate_steps": []
        }

# Global advanced ReAct agent instance
advanced_react_travel_agent = create_advanced_react_agent()
