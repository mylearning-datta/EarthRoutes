#!/usr/bin/env python3
"""
Simple inference test to demonstrate expected model responses.
This simulates what the fine-tuned model should output.
"""

import json
from typing import Dict


def simulate_mode_choice_response(origin: str, destination: str, distance_km: float) -> Dict:
    """Simulate a MODE_CHOICE response."""
    # Simple heuristic: prefer train for distances > 100km, electric car for shorter
    if distance_km > 100:
        preferred_mode = "train_electric"
        emissions = distance_km * 0.041  # kg CO2e/km
        alternatives = [
            {"mode": "train_diesel", "emissions_kg_co2e": round(distance_km * 0.041, 3)},
            {"mode": "electric_car", "emissions_kg_co2e": round(distance_km * 0.053, 3)},
            {"mode": "bus_shared", "emissions_kg_co2e": round(distance_km * 0.089, 3)}
        ]
    else:
        preferred_mode = "electric_car"
        emissions = distance_km * 0.053
        alternatives = [
            {"mode": "bicycle", "emissions_kg_co2e": 0.0},
            {"mode": "bus_shared", "emissions_kg_co2e": round(distance_km * 0.089, 3)},
            {"mode": "petrol_car", "emissions_kg_co2e": round(distance_km * 0.192, 3)}
        ]
    
    return {
        "task": "MODE_CHOICE",
        "instruction": f"From {origin} to {destination}, which low-carbon mode should I take?",
        "response": f"{preferred_mode.replace('_', ' ').title()} is the lowest-carbon feasible mode for this route.",
        "response_json": {
            "preferred_mode": preferred_mode,
            "emissions_kg_co2e": round(emissions, 3),
            "alternatives": alternatives,
            "justification": f"For {distance_km:.0f}km, {preferred_mode} offers the best carbon efficiency.",
            "warnings": [],
            "sources": ["emission_factors_2024_v1"]
        }
    }


def simulate_sustainable_pois_response(city: str) -> Dict:
    """Simulate a SUSTAINABLE_POIS response."""
    # Mock sustainable places for different cities
    city_places = {
        "Delhi": [
            {"name": "Lodhi Garden", "type": "park", "reason": "Green space, carbon sequestration"},
            {"name": "Akshardham Temple", "type": "temple", "reason": "Low energy usage, spiritual activities"}
        ],
        "Mumbai": [
            {"name": "Sanjay Gandhi National Park", "type": "national park", "reason": "Preserved ecosystem, conservation"},
            {"name": "Marine Drive", "type": "promenade", "reason": "Encourages walking, low infrastructure"}
        ],
        "Bangalore": [
            {"name": "Lalbagh Botanical Garden", "type": "botanical garden", "reason": "Supports biodiversity, green space"},
            {"name": "Cubbon Park", "type": "park", "reason": "Green space, carbon sequestration"}
        ]
    }
    
    places = city_places.get(city, [
        {"name": "Local Park", "type": "park", "reason": "Green space, low emissions"},
        {"name": "Heritage Site", "type": "historical", "reason": "Passive tourism, preserved"}
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
    
    return {
        "task": "SUSTAINABLE_POIS",
        "instruction": f"I'm visiting {city}. Suggest sustainable places to visit.",
        "response": "Here are low-impact options with sustainability reasons.",
        "response_json": {
            "recommendations": recommendations,
            "tips": [
                "Use public transport or walk between nearby POIs",
                "Carry a reusable bottle",
                "Choose local seasonal food"
            ],
            "sources": [
                "place_type_sustainability_with_reasons.csv",
                "Top Indian Places to Visit.csv"
            ]
        }
    }


def main():
    """Test inference simulation."""
    print("INFERENCE SIMULATION TEST")
    print("=" * 60)
    
    # Test MODE_CHOICE
    print("\n1. MODE_CHOICE Examples:")
    print("-" * 40)
    
    test_routes = [
        ("Mumbai", "Delhi", 1400),
        ("Bangalore", "Chennai", 350),
        ("Pune", "Lonavala", 65)
    ]
    
    for origin, dest, distance in test_routes:
        response = simulate_mode_choice_response(origin, dest, distance)
        print(f"\nRoute: {origin} → {dest} ({distance}km)")
        print(f"Response: {response['response']}")
        print(f"Preferred mode: {response['response_json']['preferred_mode']}")
        print(f"Emissions: {response['response_json']['emissions_kg_co2e']} kg CO2e")
    
    # Test SUSTAINABLE_POIS
    print("\n\n2. SUSTAINABLE_POIS Examples:")
    print("-" * 40)
    
    test_cities = ["Delhi", "Mumbai", "Bangalore"]
    
    for city in test_cities:
        response = simulate_sustainable_pois_response(city)
        print(f"\nCity: {city}")
        print(f"Response: {response['response']}")
        print("Recommendations:")
        for rec in response['response_json']['recommendations']:
            print(f"  - {rec['name']} ({rec['type']}): {rec['why_sustainable']}")
    
    print("\n" + "=" * 60)
    print("INFERENCE SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
