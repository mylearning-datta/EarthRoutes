import math
from typing import Dict, List


class CO2EmissionService:
    def __init__(self) -> None:
        # kg CO2 per km factors (aligned with original JS service)
        self.emission_factors: Dict[str, float] = {
            "flight": 0.255,
            "diesel_car": 0.171,
            "petrol_car": 0.192,
            "electric_car": 0.053,
            "train_diesel": 0.041,
            "bus_shared": 0.089,
            "train_electric": 0.041,
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


