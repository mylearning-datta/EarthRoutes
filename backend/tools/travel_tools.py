import requests
import json
from typing import Dict, List, Optional, Any
from config.settings import settings
from utils.postgres_database import postgres_db_manager as db_manager
from services.google_maps import GoogleMapsService
from services.co2_service import CO2EmissionService

class TravelTools:
    def __init__(self):
        self.google_maps_service = GoogleMapsService()
        self.co2_service = CO2EmissionService()
    
    def calculate_distance(self, source: str, destination: str, mode: str = "driving") -> Dict:
        """Calculate distance between two cities using Google Maps API"""
        try:
            if not self.google_maps_api_key:
                return self._estimate_distance(source, destination)
            
            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                "origins": source,
                "destinations": destination,
                "mode": mode,
                "key": self.google_maps_api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data["status"] == "OK" and data["rows"][0]["elements"][0]["status"] == "OK":
                element = data["rows"][0]["elements"][0]
                return {
                    "distance": element["distance"],
                    "duration": element["duration"],
                    "source": source,
                    "destination": destination,
                    "mode": mode
                }
            else:
                return self._estimate_distance(source, destination)
                
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return self._estimate_distance(source, destination)
    
    def _estimate_distance(self, source: str, destination: str) -> Dict:
        """Fallback distance estimation"""
        # Simple distance estimation based on major Indian cities
        city_distances = {
            'Delhi-Mumbai': 1400,
            'Mumbai-Bangalore': 850,
            'Delhi-Bangalore': 2150,
            'Mumbai-Hyderabad': 710,
            'Delhi-Hyderabad': 1600,
            'Bangalore-Hyderabad': 570,
            'Delhi-Kolkata': 1500,
            'Mumbai-Kolkata': 2000,
            'Bangalore-Kolkata': 1900,
            'Hyderabad-Kolkata': 1200
        }
        
        route = f"{source}-{destination}"
        reverse_route = f"{destination}-{source}"
        
        distance_km = city_distances.get(route) or city_distances.get(reverse_route) or 500
        
        return {
            "distance": {"text": f"{distance_km} km", "value": distance_km * 1000},
            "duration": {"text": "Estimated", "value": 0},
            "source": source,
            "destination": destination,
            "mode": "estimated"
        }
    
    def calculate_co2_emissions(self, distance_km: float, travel_mode: str) -> Dict:
        """Calculate CO2 emissions for a given distance and travel mode"""
        try:
            travel_modes = db_manager.get_travel_modes()
            if travel_mode in travel_modes:
                mode_data = travel_modes[travel_mode]
                emission_factor = mode_data["emission_factor"]
                total_emissions = distance_km * emission_factor
                
                # Calculate trees needed (1 tree absorbs ~22 kg CO2 per year)
                trees_needed = max(1, round(total_emissions / 22))
                
                return {
                    "travel_mode": travel_mode,
                    "mode_name": mode_data["name"],
                    "category": mode_data["category"],
                    "distance_km": distance_km,
                    "emission_factor": emission_factor,
                    "total_emissions": total_emissions,
                    "trees_needed": trees_needed
                }
            else:
                return {"error": f"Unknown travel mode: {travel_mode}"}
                
        except Exception as e:
            print(f"Error calculating CO2 emissions: {e}")
            return {"error": str(e)}
    
    def compare_travel_modes(self, source: str, destination: str) -> Dict:
        """Compare different travel modes for a route"""
        try:
            # Get distance using the new service
            distance_data = self.google_maps_service.get_distance(source, destination)
            distance_km = distance_data["distance"]["value"] / 1000
            
            # Calculate emissions for all travel modes using the new service
            travel_options = []
            
            travel_modes = db_manager.get_travel_modes()
            for mode_id in travel_modes:
                co2_data = self.co2_service.calculate_emissions(distance_km, mode_id)
                
                # Calculate duration based on mode
                duration = self._calculate_duration(distance_km, mode_id)
                
                travel_options.append({
                    "id": mode_id,
                    "name": self._get_mode_name(mode_id),
                    "category": self._get_mode_category(mode_id),
                    "distance": distance_km,
                    "co2Emissions": co2_data["totalEmissions"],
                    "emissionFactor": co2_data["emissionFactor"],
                    "treesNeeded": co2_data["equivalentMetrics"]["treesNeeded"],
                    "duration": duration
                })
            
            # Sort by CO2 emissions (lowest first)
            travel_options.sort(key=lambda x: x["co2Emissions"])
            
            return {
                "source": source,
                "destination": destination,
                "distance": distance_data["distance"]["text"],
                "options": travel_options
            }
            
        except Exception as e:
            print(f"Error comparing travel modes: {e}")
            return {"error": str(e)}
    
    def _calculate_duration(self, distance_km: float, mode: str) -> str:
        """Calculate estimated duration for different travel modes"""
        if mode == "bicycle":
            hours = distance_km / 15  # 15 km/h average
        elif mode == "walking":
            hours = distance_km / 5   # 5 km/h average
        elif mode == "flight":
            hours = (distance_km / 500) + 2  # 500 km/h + 2 hours for airport procedures
        elif mode in ["diesel_car", "petrol_car", "electric_car"]:
            hours = distance_km / 60  # 60 km/h average including stops
        elif mode in ["train_diesel", "train_electric"]:
            hours = distance_km / 80  # 80 km/h average including stops
        elif mode == "bus_shared":
            hours = distance_km / 50  # 50 km/h average including stops
        else:
            hours = distance_km / 60  # Default
        
        return self._format_duration(hours)
    
    def _format_duration(self, hours: float) -> str:
        """Format duration in a readable format"""
        if hours < 1:
            minutes = round(hours * 60)
            return f"{minutes} mins"
        elif hours < 24:
            whole_hours = int(hours)
            minutes = round((hours - whole_hours) * 60)
            if minutes == 0:
                return f"{whole_hours} hrs"
            return f"{whole_hours} hrs {minutes} mins"
        else:
            days = int(hours / 24)
            remaining_hours = int(hours % 24)
            if remaining_hours == 0:
                return f"{days} days"
            return f"{days} days {remaining_hours} hrs"
    
    def get_travel_suggestions(self, source: str, destination: str) -> Dict:
        """Get comprehensive travel suggestions including hotels and places"""
        try:
            # Get travel mode comparison
            travel_comparison = self.compare_travel_modes(source, destination)
            
            # Get database suggestions
            db_suggestions = db_manager.get_travel_suggestions(source, destination)
            
            return {
                "travel_comparison": travel_comparison,
                "hotels": db_suggestions["hotels"],
                "places": db_suggestions["places"],
                "sustainable_places": db_suggestions["sustainable_places"],
                "source": source,
                "destination": destination
            }
            
        except Exception as e:
            print(f"Error getting travel suggestions: {e}")
            return {"error": str(e)}
    
    def get_available_cities(self) -> List[str]:
        """Get list of available cities"""
        return db_manager.get_cities()
    
    def _get_mode_name(self, mode_id: str) -> str:
        """Get human-readable name for travel mode"""
        mode_names = {
            "flight": "Flight",
            "diesel_car": "Diesel Car",
            "petrol_car": "Petrol Car", 
            "electric_car": "Electric Car",
            "train_diesel": "Diesel Train",
            "bus_shared": "Shared Bus",
            "train_electric": "Electric Train",
            "bicycle": "Bicycle",
            "walking": "Walking"
        }
        return mode_names.get(mode_id, mode_id.title())
    
    def _get_mode_category(self, mode_id: str) -> str:
        """Get category for travel mode"""
        categories = {
            "flight": "Air",
            "diesel_car": "Road",
            "petrol_car": "Road",
            "electric_car": "Road",
            "train_diesel": "Rail",
            "bus_shared": "Road",
            "train_electric": "Rail",
            "bicycle": "Active",
            "walking": "Active"
        }
        return categories.get(mode_id, "Other")

# Global travel tools instance
travel_tools = TravelTools()
