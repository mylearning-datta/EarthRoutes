#!/usr/bin/env python3
"""
Build finetuning datasets for two tasks:
  - MODE_CHOICE: choose lowest-carbon feasible transport mode between cities
  - SUSTAINABLE_POIS: recommend sustainable places to visit in a destination city

Inputs (produced by extract_facts.py):
  finetuning/data/processed/places.jsonl
  finetuning/data/processed/hotels.jsonl (optional)
  finetuning/data/processed/sustainability_index.json

Outputs:
  finetuning/data/processed/mode_choice.jsonl
  finetuning/data/processed/sustainable_pois.jsonl

Notes:
  - Uses backend services for distance and emission factors when available.
  - Contains simple feasibility heuristics for modes based on distance.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "finetuning" / "data" / "processed"
ASSETS_DIR = PROJECT_ROOT / "finetuning" / "assets"

# Make backend importable
import sys
sys.path.append(str((PROJECT_ROOT / "backend").resolve()))

try:
    from services.co2_service import CO2EmissionService
    from services.google_maps import GoogleMapsService
except Exception:
    # Fallback minimal implementations if backend imports fail
    class CO2EmissionService:  # type: ignore
        def __init__(self) -> None:
            self.emission_factors = {
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
            ef = self.emission_factors.get(travel_mode, 0.1)
            return {"totalEmissions": distance_km * ef, "emissionFactor": ef}

    class GoogleMapsService:  # type: ignore
        def __init__(self) -> None:
            pass

        def get_distance(self, source: str, destination: str, mode: str = "driving") -> Dict:
            # Very rough fallback: pretend 1 degree ~ 111 km using simple hash on names
            rng = random.Random(hash(source + destination) & 0xFFFFFFFF)
            km = max(5.0, rng.uniform(10, 1500))
            return {"distance": {"text": f"{km:.1f} km", "value": km * 1000}}


@dataclass
class ModeChoiceExample:
    task: str
    instruction: str
    context: str
    response: str
    response_json: Dict


@dataclass
class SustainablePoisExample:
    task: str
    instruction: str
    context: str
    response: str
    response_json: Dict


def read_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_emission_factors() -> Dict[str, float]:
    # Prefer backend service factors; fallback to assets file if present
    co2 = CO2EmissionService()
    factors = getattr(co2, "emission_factors", {}) or {}
    assets_file = ASSETS_DIR / "emission_factors.json"
    if assets_file.exists():
        try:
            with assets_file.open("r", encoding="utf-8") as f:
                asset_factors = json.load(f)
                factors.update(asset_factors)
        except Exception:
            pass
    return factors


def format_factors_for_context(factors: Dict[str, float], modes: List[str]) -> str:
    parts = [f"{m}: {factors.get(m, 0.0):.3f} kgCO2e/km" for m in modes]
    return ", ".join(parts)


def feasible_modes_for_distance(distance_km: float) -> List[str]:
    modes: List[str] = []
    if distance_km <= 3:
        modes.extend(["walking", "bicycle"])  # hyper-local
    if distance_km <= 30:
        modes.append("bicycle")
    if distance_km <= 150:
        modes.extend(["bus_shared", "train_electric", "train_diesel", "electric_car", "petrol_car", "diesel_car"])
    elif distance_km <= 800:
        modes.extend(["bus_shared", "train_electric", "train_diesel", "electric_car", "petrol_car", "diesel_car"])
    else:
        modes.extend(["train_electric", "train_diesel", "bus_shared", "electric_car", "petrol_car", "diesel_car", "flight"])

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for m in modes:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def build_mode_choice(max_examples: int = 1000, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    places = read_jsonl(PROCESSED_DIR / "places.jsonl")
    if not places:
        print("No places.jsonl found. Run extract_facts.py first.")
        return []

    # Gather unique cities with a few representatives
    cities = sorted({p["city"] for p in places if p.get("city")})
    if len(cities) < 2:
        return []

    maps = GoogleMapsService()
    co2 = CO2EmissionService()
    ef = load_emission_factors()

    examples: List[Dict] = []
    pairs: List[Tuple[str, str]] = []
    # Sample pairs by shuffling and taking prefixes
    shuffled = cities[:]
    random.shuffle(shuffled)
    for i, src in enumerate(shuffled):
        # choose up to 5 destinations per source
        dests = shuffled[i + 1 : i + 6]
        for dst in dests:
            if src == dst:
                continue
            pairs.append((src, dst))
            if len(pairs) >= max_examples * 2:  # build more pairs than needed; later trim
                break
        if len(pairs) >= max_examples * 2:
            break

    for (src, dst) in pairs:
        try:
            dist_info = maps.get_distance(src, dst)
            meters = float(dist_info["distance"]["value"])
            distance_km = max(0.0, meters / 1000.0)
        except Exception:
            continue

        available_modes = feasible_modes_for_distance(distance_km)
        # Compute emissions per mode
        per_mode: List[Tuple[str, float]] = []
        for m in available_modes:
            res = co2.calculate_emissions(distance_km, m)
            per_mode.append((m, float(res["totalEmissions"])))

        per_mode.sort(key=lambda t: t[1])
        if not per_mode:
            continue
        preferred_mode, preferred_emissions = per_mode[0]
        alternatives = [
            {"mode": m, "emissions_kg_co2e": round(e, 3)} for m, e in per_mode[1:4]
        ]

        instruction = f"From {src} to {dst}, which low-carbon mode should I take?"
        factors_str = format_factors_for_context(ef, available_modes[:6])
        context = f"Distance ≈ {distance_km:.0f} km. Modes: {', '.join(available_modes[:6])}. Factors: {factors_str}."
        response = f"{preferred_mode.replace('_', ' ').title()} is the lowest-carbon feasible mode for this route."
        response_json = {
            "preferred_mode": preferred_mode,
            "emissions_kg_co2e": round(preferred_emissions, 3),
            "alternatives": alternatives,
            "justification": f"Among available modes for ~{distance_km:.0f} km, {preferred_mode} has the lowest intensity.",
            "warnings": [],
            "sources": ["emission_factors_2024_v1"],
        }

        examples.append({
            "task": "MODE_CHOICE",
            "instruction": instruction,
            "context": context,
            "response": response,
            "response_json": response_json,
        })
        if len(examples) >= max_examples:
            break

    return examples


def build_sustainable_pois(max_examples_per_city: int = 1) -> List[Dict]:
    places = read_jsonl(PROCESSED_DIR / "places.jsonl")
    if not places:
        print("No places.jsonl found. Run extract_facts.py first.")
        return []

    # Group sustainable POIs by city
    by_city: Dict[str, List[Dict]] = {}
    for p in places:
        if not p.get("is_sustainable"):
            continue
        city = p.get("city")
        if not city:
            continue
        by_city.setdefault(city, []).append(p)

    examples: List[Dict] = []
    for city, pois in by_city.items():
        if not pois:
            continue
        # Select top-k by variety of place_type
        pois_sorted = sorted(pois, key=lambda x: (x.get("place_type") or "", x.get("name") or ""))
        chosen: List[Dict] = []
        seen_types: set = set()
        for poi in pois_sorted:
            if poi.get("place_type") in seen_types and len(chosen) >= 5:
                continue
            chosen.append(poi)
            seen_types.add(poi.get("place_type"))
            if len(chosen) >= 6:
                break

        if not chosen:
            continue

        instruction = f"I'm visiting {city}. Suggest sustainable places to visit."
        ctx_snippets = []
        for c in chosen[:4]:
            t = c.get("place_type") or ""
            r = c.get("sustainability_reason") or ""
            if t:
                ctx_snippets.append(f"{t}: {r}")
        context = " | ".join(ctx_snippets)
        response = "Here are low-impact options with reasons."

        recommendations = []
        for c in chosen:
            recommendations.append({
                "name": c.get("name"),
                "type": c.get("place_type"),
                "sustainability_tags": [c.get("place_type")] if c.get("place_type") else [],
                "certifications": [],
                "why_sustainable": c.get("sustainability_reason") or "",
                "best_season": c.get("best_time") or None,
            })

        response_json = {
            "recommendations": recommendations,
            "tips": ["Use public transport or walk between nearby POIs", "Carry a reusable bottle"],
            "sources": [
                "place_type_sustainability_with_reasons.csv",
                "Top Indian Places to Visit.csv",
            ],
        }

        examples.append({
            "task": "SUSTAINABLE_POIS",
            "instruction": instruction,
            "context": context,
            "response": response,
            "response_json": response_json,
        })

    return examples


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mode_choice = build_mode_choice(max_examples=500)
    sustainable_pois = build_sustainable_pois()

    out_mode = PROCESSED_DIR / "mode_choice.jsonl"
    out_pois = PROCESSED_DIR / "sustainable_pois.jsonl"

    write_jsonl(out_mode, mode_choice)
    write_jsonl(out_pois, sustainable_pois)

    print(f"Wrote: {out_mode} ({len(mode_choice)})")
    print(f"Wrote: {out_pois} ({len(sustainable_pois)})")


if __name__ == "__main__":
    main()


