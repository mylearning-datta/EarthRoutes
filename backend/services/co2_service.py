import math
from typing import Dict, List
from utils.postgres_database import postgres_db_manager


class CO2EmissionService:
    def __init__(self) -> None:
        # Load emission factors from database when available, fallback otherwise
        try:
            modes = postgres_db_manager.get_travel_modes()
            self.emission_factors = {mode_id: data.get("emission_factor", 0.1) for mode_id, data in modes.items()}
        except Exception:
            # Safe fallback
            self.emission_factors: Dict[str, float] = {
                "flight": 0.255,
                "diesel_car": 0.169,
                "petrol_car": 0.155,
                "electric_car": 0.07,
                "train_diesel": 0.06,
                "bus_shared": 0.08,
                "train_electric": 0.03,
                "bicycle": 0.0,
                "walking": 0.0,
            }

    def calculate_emissions(self, distance_km: float, travel_mode: str, options: Dict = {}) -> Dict:
        emission_factor = self.emission_factors.get(travel_mode, 0.1)
        total_emissions = float(distance_km) * emission_factor

        trees_needed = math.ceil(total_emissions / 22)  # ~22kg CO2/year per tree
        daily_avg_pct = (total_emissions / 16.4) * 100  # ~16.4kg/day avg person

        return {
            "distanceKm": distance_km,
            "travelMode": travel_mode,
            "emissionFactor": emission_factor,
            "totalEmissions": total_emissions,
            "equivalentMetrics": {
                "treesNeeded": trees_needed,
                "dailyAveragePercentage": daily_avg_pct,
            },
        }

    def compare_emissions(self, distance_km: float, travel_modes: List[str]) -> List[Dict]:
        results = [self.calculate_emissions(distance_km, mode) for mode in travel_modes]
        return sorted(results, key=lambda x: x["totalEmissions"])  # lowest first

    def calculate_savings(self, distance_km: float, from_mode: str, to_mode: str) -> Dict:
        from_em = self.calculate_emissions(distance_km, from_mode)
        to_em = self.calculate_emissions(distance_km, to_mode)
        savings = from_em["totalEmissions"] - to_em["totalEmissions"]
        pct = (savings / from_em["totalEmissions"]) * 100 if from_em["totalEmissions"] > 0 else 0
        return {
            "distanceKm": distance_km,
            "fromMode": from_mode,
            "toMode": to_mode,
            "fromEmissions": from_em["totalEmissions"],
            "toEmissions": to_em["totalEmissions"],
            "savings": savings,
            "savingsPercentage": pct,
        }

    def get_all_emission_factors(self) -> Dict[str, float]:
        return self.emission_factors


