from typing import Dict, List
import requests
from config.settings import settings


class GoogleMapsService:
    def __init__(self) -> None:
        self.api_key = settings.GOOGLE_MAPS_API_KEY
        self.base_url = "https://maps.googleapis.com/maps/api"

    def _estimate_distance_km(self, source: str, destination: str) -> float:
        """Estimate distance using Haversine formula for major cities"""
        # Get city coordinates from settings
        city_coords = settings.CITY_COORDINATES
        
        # Normalize city names
        source_norm = source.lower().strip()
        dest_norm = destination.lower().strip()
        
        # Try exact match first
        if source_norm in city_coords and dest_norm in city_coords:
            return self._haversine_distance(city_coords[source_norm], city_coords[dest_norm])
        
        # Try partial matching
        for city, coords in city_coords.items():
            if city in source_norm or source_norm in city:
                source_coords = coords
                break
        else:
            # Default fallback coordinates (Delhi)
            source_coords = (28.6139, 77.2090)
            
        for city, coords in city_coords.items():
            if city in dest_norm or dest_norm in city:
                dest_coords = coords
                break
        else:
            # Default fallback coordinates (Mumbai)
            dest_coords = (19.0760, 72.8777)
            
        return self._haversine_distance(source_coords, dest_coords)
    
    def _haversine_distance(self, coord1: tuple, coord2: tuple) -> float:
        """Calculate distance between two coordinates using Haversine formula"""
        import math
        
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return round(c * r, 2)

    def get_distance(self, source: str, destination: str, mode: str = "driving") -> Dict:
        if not self.api_key:
            km = self._estimate_distance_km(source, destination)
            return {
                "distance": {"text": f"{km} km", "value": km * 1000},
                "duration": {"text": "Estimated", "value": 0},
                "origin": source,
                "destination": destination,
                "mode": "estimated",
            }

        url = f"{self.base_url}/distancematrix/json"
        params = {"origins": source, "destinations": destination, "mode": mode, "key": self.api_key}
        try:
            resp = requests.get(url, params=params, timeout=20)
            data = resp.json()
            if data.get("status") == "OK":
                el = data["rows"][0]["elements"][0]
                if el.get("status") == "OK":
                    return {
                        "distance": el["distance"],
                        "duration": el["duration"],
                        "origin": source,
                        "destination": destination,
                        "mode": mode,
                    }
        except Exception:
            pass
        km = self._estimate_distance_km(source, destination)
        return {
            "distance": {"text": f"{km} km", "value": km * 1000},
            "duration": {"text": "Estimated", "value": 0},
            "origin": source,
            "destination": destination,
            "mode": "estimated",
        }

    def get_multiple_distances(self, origins: List[str], destinations: List[str], mode: str = "driving") -> List[Dict]:
        results: List[Dict] = []
        for o in origins:
            for d in destinations:
                results.append(self.get_distance(o, d, mode))
        return results
    
    def add_city_coordinates(self, city_name: str, latitude: float, longitude: float):
        """Add new city coordinates to settings (runtime addition)"""
        settings.CITY_COORDINATES[city_name.lower()] = (latitude, longitude)
    
    def get_available_cities(self) -> List[str]:
        """Get list of cities with known coordinates"""
        return list(settings.CITY_COORDINATES.keys())


