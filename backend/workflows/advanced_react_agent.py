from typing import Dict, List, Any, Optional, Type
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from config.settings import settings
from tools.travel_tools import travel_tools
import json
import re
from datetime import datetime

class AdvancedTravelPlanningTool(BaseTool):
    """Advanced tool for comprehensive travel planning with reasoning"""
    
    name: str = "advanced_travel_planner"
    description: str = """Use this tool for comprehensive travel planning including route analysis, CO2 calculations, and eco-friendly recommendations.
    Input should be a JSON string with keys: source, destination, preferences (optional).
    preferences can include: budget, time_constraint, eco_priority, comfort_level"""
    
    def _run(self, query: str) -> str:
        try:
            data = json.loads(query)
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
            
            return json.dumps(result, indent=2)
                
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    def _analyze_travel_options(self, travel_data: Dict, preferences: Dict) -> Dict:
        """Analyze travel options based on preferences"""
        analysis = {
            "eco_friendly_ranking": [],
            "time_efficient_ranking": [],
            "cost_effective_ranking": [],
            "comfort_ranking": []
        }
        
        if "travel_comparison" in travel_data and "options" in travel_data["travel_comparison"]:
            options = travel_data["travel_comparison"]["options"]
            
            # Sort by different criteria
            analysis["eco_friendly_ranking"] = sorted(options, key=lambda x: x["co2Emissions"])
            analysis["time_efficient_ranking"] = sorted(options, key=lambda x: self._parse_duration(x["duration"]))
            
        return analysis
    
    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string to hours for comparison"""
        try:
            if "days" in duration_str:
                parts = duration_str.split()
                days = int(parts[0])
                hours = int(parts[2]) if len(parts) > 2 else 0
                return days * 24 + hours
            elif "hrs" in duration_str:
                parts = duration_str.split()
                hours = int(parts[0])
                minutes = int(parts[2]) if len(parts) > 2 and "mins" in parts[2] else 0
                return hours + minutes / 60
            elif "mins" in duration_str:
                minutes = int(duration_str.split()[0])
                return minutes / 60
            else:
                return 24  # Default fallback
        except:
            return 24
    
    def _generate_recommendations(self, travel_data: Dict, preferences: Dict) -> List[Dict]:
        """Generate personalized recommendations"""
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
            
            # Balanced recommendation (middle ground)
            if len(options) >= 3:
                balanced_option = options[len(options)//2]
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
            data = json.loads(query)
            
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
            
            return json.dumps(analysis, indent=2)
                
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    def _analyze_route_impact(self, route_data: Dict) -> Dict:
        """Analyze environmental impact of a route"""
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
        
        return analysis
    
    def _analyze_single_mode_impact(self, co2_data: Dict, distance_km: float) -> Dict:
        """Analyze impact of a single travel mode"""
        return {
            "emissions": co2_data,
            "impact_category": self._categorize_footprint(co2_data["total_emissions"]),
            "offsetting_strategies": self._generate_offsetting_strategies(co2_data["total_emissions"]),
            "distance_km": distance_km
        }
    
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
        
        return alternatives
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class CityInformationTool(BaseTool):
    """Tool for getting city information and available cities"""
    
    name: str = "city_info"
    description: str = """Use this tool to get information about available cities, hotels, and attractions.
    Input should be a JSON string with keys: action, city (optional).
    action can be: 'list_cities', 'get_hotels', 'get_attractions'"""
    
    def _run(self, query: str) -> str:
        try:
            data = json.loads(query)
            action = data.get("action")
            city = data.get("city")
            
            if action == "list_cities":
                cities = travel_tools.get_available_cities()
                return json.dumps({"cities": cities}, indent=2)
            elif action == "get_hotels" and city:
                from utils.database import db_manager
                hotels = db_manager.get_hotels_in_city(city)
                return json.dumps({"hotels": hotels}, indent=2)
            elif action == "get_attractions" and city:
                from utils.database import db_manager
                places = db_manager.get_places_in_city(city)
                return json.dumps({"attractions": places}, indent=2)
            else:
                return json.dumps({"error": "Invalid action or missing city"}, indent=2)
                
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

def create_advanced_react_agent():
    """Create an advanced ReAct agent with memory and enhanced reasoning"""
    
    # Initialize the LLM with GPT-4o
    llm = ChatOpenAI(
        model=settings.AGENT_MODEL,
        temperature=settings.AGENT_TEMPERATURE,
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
    
    # Enhanced ReAct prompt with reasoning capabilities
    react_prompt = PromptTemplate.from_template("""
You are an advanced travel planning assistant with deep expertise in sustainable travel, environmental impact analysis, and personalized recommendations.

You have access to the following tools:
{tools}

You should use a systematic approach to solve travel planning problems:

1. **Understand**: Analyze the user's request and identify key requirements
2. **Plan**: Determine what information you need and which tools to use
3. **Execute**: Use tools to gather data and perform calculations
4. **Analyze**: Process the results and identify patterns/insights
5. **Recommend**: Provide personalized recommendations with clear reasoning
6. **Explain**: Help the user understand the environmental and practical implications

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
Final Answer: [Provide a detailed, well-reasoned response with specific recommendations, environmental impact analysis, and actionable advice]

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
        max_iterations=8,  # More iterations for complex reasoning
        early_stopping_method="generate",
        return_intermediate_steps=True
    )
    
    return agent_executor

def process_travel_query_advanced_react(query: str) -> Dict[str, Any]:
    """Process a travel query using advanced ReAct agent"""
    try:
        # Create agent
        agent = create_advanced_react_agent()
        
        # Process the query
        result = agent.invoke({"input": query})
        
        return {
            "response": result["output"],
            "travel_data": None,  # Advanced agent handles data internally
            "error": None,
            "intermediate_steps": result.get("intermediate_steps", [])
        }
        
    except Exception as e:
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
            "travel_data": None,
            "error": str(e),
            "intermediate_steps": []
        }

# Global advanced ReAct agent instance
advanced_react_travel_agent = create_advanced_react_agent()
