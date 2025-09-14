#!/usr/bin/env python3
"""
Test script for natural language inputs - simulating real user queries.
Shows how the model should handle conversational travel requests.
"""

import json
import re
from typing import Dict, List, Tuple


def parse_natural_query(query: str) -> Dict:
    """Parse natural language query to extract intent and parameters."""
    query_lower = query.lower()
    
    # Extract origin and destination - improved regex
    from_match = re.search(r'from\s+([a-zA-Z\s]+?)(?:\s+to|\s+and)', query_lower)
    
    # Special case for "from X to Y and provide" pattern
    destination = None
    if 'and provide' in query_lower:
        and_match = re.search(r'to\s+([a-zA-Z\s]+?)\s+and\s+provide', query_lower)
        if and_match:
            destination = and_match.group(1).strip()
    else:
        to_match = re.search(r'to\s+([a-zA-Z\s]+?)(?:\s+and|\s+provide|\s+give|\s+show|\s+with|$)', query_lower)
        destination = to_match.group(1).strip() if to_match else None
    
    origin = from_match.group(1).strip() if from_match else None
    
    # Determine task type and target city
    target_city = None
    
    if any(word in query_lower for word in ['sustainable', 'eco', 'green', 'places', 'visit', 'attractions']):
        task = "SUSTAINABLE_POIS"
        # For POI queries, target city is usually the destination or mentioned city
        if destination and not any(word in destination for word in ['travel', 'provide', 'give', 'show']):
            target_city = destination
        elif origin and not any(word in origin for word in ['travel', 'provide', 'give', 'show']):
            target_city = origin
        else:
            # Try to find city names in the query
            cities = ['bangalore', 'chennai', 'mumbai', 'delhi', 'hyderabad', 'pune', 'kolkata', 'ahmedabad', 'jaipur', 'mysore', 'agra', 'lonavala']
            for city in cities:
                if city in query_lower:
                    target_city = city
                    break
    elif any(word in query_lower for word in ['travel', 'go', 'transport', 'mode', 'way', 'how']):
        task = "MODE_CHOICE"
    else:
        task = "UNKNOWN"
    
    return {
        "original_query": query,
        "task": task,
        "origin": origin,
        "destination": destination,
        "target_city": target_city
    }


def simulate_natural_response(parsed_query: Dict) -> Dict:
    """Simulate model response to natural language query."""
    task = parsed_query["task"]
    original_query = parsed_query["original_query"]
    
    if task == "MODE_CHOICE":
        origin = parsed_query["origin"]
        destination = parsed_query["destination"]
        
        if not origin or not destination:
            return {
                "error": "Could not identify origin and destination. Please specify 'from [city] to [city]'."
            }
        
        # Simulate distance and mode selection
        distance = 500  # Mock distance
        preferred_mode = "train_electric"
        emissions = distance * 0.041
        
        response_text = f"For traveling from {origin.title()} to {destination.title()}, I recommend taking the train. It's the most sustainable option with only {emissions:.1f} kg CO2e emissions for the {distance}km journey."
        
        return {
            "task": "MODE_CHOICE",
            "original_query": original_query,
            "response": response_text,
            "response_json": {
                "preferred_mode": preferred_mode,
                "emissions_kg_co2e": round(emissions, 3),
                "alternatives": [
                    {"mode": "electric_car", "emissions_kg_co2e": round(distance * 0.053, 3)},
                    {"mode": "bus_shared", "emissions_kg_co2e": round(distance * 0.089, 3)}
                ],
                "justification": f"Train is the lowest-carbon option for this {distance}km route.",
                "tips": [
                    "Book tickets in advance for better rates",
                    "Consider overnight trains to save on accommodation"
                ]
            }
        }
    
    elif task == "SUSTAINABLE_POIS":
        city = parsed_query["target_city"]
        
        if not city:
            return {
                "error": "Could not identify the destination city. Please specify where you're visiting."
            }
        
        # Mock sustainable places for the city
        city_places = {
            "chennai": [
                {"name": "Marina Beach", "type": "beach", "reason": "Natural coastline, walking-friendly"},
                {"name": "Guindy National Park", "type": "national park", "reason": "Urban wildlife conservation"}
            ],
            "bangalore": [
                {"name": "Lalbagh Botanical Garden", "type": "botanical garden", "reason": "Biodiversity conservation, green space"},
                {"name": "Cubbon Park", "type": "park", "reason": "Carbon sequestration, walking trails"}
            ],
            "mumbai": [
                {"name": "Sanjay Gandhi National Park", "type": "national park", "reason": "Wildlife conservation, forest trails"},
                {"name": "Marine Drive", "type": "promenade", "reason": "Pedestrian-friendly, low infrastructure"}
            ],
            "delhi": [
                {"name": "Lodhi Garden", "type": "park", "reason": "Historic green space, walking paths"},
                {"name": "Akshardham Temple", "type": "temple", "reason": "Spiritual site, low energy usage"}
            ]
        }
        
        places = city_places.get(city.lower(), [
            {"name": "Local Park", "type": "park", "reason": "Green space, low environmental impact"},
            {"name": "Heritage Site", "type": "historical", "reason": "Cultural preservation, passive tourism"}
        ])
        
        recommendations = []
        for place in places:
            recommendations.append({
                "name": place["name"],
                "type": place["type"],
                "sustainability_tags": [place["type"]],
                "certifications": [],
                "why_sustainable": place["reason"],
                "best_season": "All"
            })
        
        response_text = f"Here are some sustainable places to visit in {city.title()}: {', '.join([p['name'] for p in places])}. These locations have minimal environmental impact and promote eco-friendly tourism."
        
        return {
            "task": "SUSTAINABLE_POIS",
            "original_query": original_query,
            "response": response_text,
            "response_json": {
                "recommendations": recommendations,
                "tips": [
                    "Use public transport or walk between locations",
                    "Carry a reusable water bottle",
                    "Support local eco-friendly businesses"
                ],
                "sources": ["sustainability database", "local tourism board"]
            }
        }
    
    else:
        return {
            "error": "I can help with travel mode recommendations or sustainable place suggestions. Please specify your request more clearly."
        }


def test_natural_queries():
    """Test various natural language queries."""
    print("NATURAL LANGUAGE INPUT TEST")
    print("=" * 70)
    
    test_queries = [
        "I want to travel from Bangalore to Chennai and provide me with suitable sustainable options",
        "How should I go from Mumbai to Delhi in an eco-friendly way?",
        "I'm visiting Bangalore, suggest some sustainable places to visit",
        "What's the best green way to travel from Pune to Lonavala?",
        "I'm going to Chennai, what are some eco-friendly attractions?",
        "From Delhi to Agra, what's the most sustainable transport option?",
        "Show me sustainable places in Mumbai",
        "I need to go from Bangalore to Mysore, what's the greenest way?",
        "Visiting Delhi, recommend eco-friendly places",
        "Travel from Hyderabad to Bangalore sustainably"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. QUERY: \"{query}\"")
        print("-" * 50)
        
        # Parse the query
        parsed = parse_natural_query(query)
        print(f"Parsed: {parsed}")
        
        # Generate response
        response = simulate_natural_response(parsed)
        
        if "error" in response:
            print(f"ERROR: {response['error']}")
        else:
            print(f"TASK: {response['task']}")
            print(f"RESPONSE: {response['response']}")
            
            if "response_json" in response:
                json_resp = response["response_json"]
                if response["task"] == "MODE_CHOICE":
                    print(f"PREFERRED MODE: {json_resp.get('preferred_mode', 'N/A')}")
                    print(f"EMISSIONS: {json_resp.get('emissions_kg_co2e', 'N/A')} kg CO2e")
                elif response["task"] == "SUSTAINABLE_POIS":
                    recs = json_resp.get('recommendations', [])
                    print(f"RECOMMENDATIONS: {len(recs)} sustainable places")
                    for rec in recs[:2]:  # Show first 2
                        print(f"  - {rec['name']}: {rec['why_sustainable']}")
    
    print("\n" + "=" * 70)
    print("NATURAL LANGUAGE TEST COMPLETE")
    print("=" * 70)


def show_expected_training_format():
    """Show how natural queries would be formatted for training."""
    print("\n\nEXPECTED TRAINING FORMAT")
    print("=" * 70)
    
    sample_query = "I want to travel from Bangalore to Chennai and provide me with suitable sustainable options"
    parsed = parse_natural_query(sample_query)
    response = simulate_natural_response(parsed)
    
    print("Natural Query:", sample_query)
    print("\nTraining Format:")
    print("-" * 30)
    
    training_prompt = f"""### TASK: {response['task']}
### INSTRUCTION:
{sample_query}
### RESPONSE:
{response['response']}
### RESPONSE_JSON:
{json.dumps(response['response_json'], indent=2)}"""
    
    print(training_prompt)


if __name__ == "__main__":
    test_natural_queries()
    show_expected_training_format()
