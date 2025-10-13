#!/usr/bin/env python3
"""
Normalize raw place and hotel data into structured JSONL tables for finetuning tasks.

Inputs:
- data/Top Indian Places to Visit.csv
- data/place_type_sustainability_with_reasons.csv
- data/hotel_details.csv (optional)

Outputs (in finetuning/data/processed):
- places.jsonl: normalized POIs with sustainability flags and reasons
- hotels.jsonl: normalized hotels with sustainability features (if available)
- sustainability_index.json: mapping of place types -> {is_sustainable, reason}
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "finetuning" / "data" / "processed"


@dataclass
class PlaceRecord:
    zone: str
    state: str
    city: str
    name: str
    place_type: str
    is_sustainable: bool
    sustainability_reason: str
    best_time: Optional[str] = None


@dataclass
class HotelRecord:
    city: str
    name: str
    sustainability_tags: List[str]
    certifications: List[str]


def load_sustainability_index(path: Path) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys like "Place Type" and values
            place_type = (row.get("Place Type") or "").strip().lower()
            is_sustainable = (row.get("is_sustainable") or "No").strip().lower() in {"yes", "true", "1"}
            reason = (row.get("Reason") or "").strip()
            if place_type:
                index[place_type] = {
                    "is_sustainable": is_sustainable,
                    "reason": reason,
                }
    return index


def normalize_place_type(value: str) -> str:
    v = (value or "").strip().lower()
    # Simple harmonization rules for frequent variants
    synonyms = {
        "religious shrine": "religious site",
        "gurudwara": "religious site",
        "church": "church",
        "temple": "temple",
        "tombs": "tomb",
        "mausoleum": "mausoleum",
        "national park": "national park",
        "wildlife sanctuary": "wildlife sanctuary",
        "bird sanctuary": "bird sanctuary",
        "park": "park",
        "botanical garden": "botanical garden",
        "museum": "museum",
        "fort": "fort",
        "palace": "palace",
        "lake": "lake",
        "beach": "beach",
        "valley": "valley",
        "cave": "cave",
        "viewpoint": "viewpoint",
        "hill": "hill",
        "site": "site",
        "monument": "monument",
        "market": "market",
        "promenade": "promenade",
        "observatory": "observatory",
        "science": "science",
        "zoo": "zoo",
        "entertainment": "entertainment",
        "amusement park": "amusement park",
        "shopping": "mall",
        "mall": "mall",
        "vineyard": "vineyard",
        "scenic area": "scenic area",
        "scenic point": "scenic point",
        "waterfall": "waterfall",
        "bridge": "bridge",
        "dam": "dam",
        "race track": "race track",
        "aquarium": "aquarium",
        "monastery": "monastery",
        "spiritual center": "spiritual center",
        "government building": "government building",
        "commercial complex": "commercial complex",
        "cultural": "cultural",
        "historical": "historical",
        "prehistory": "prehistoric site",
        "prehistoric site": "prehistoric site",
        "township": "township",
        "natural feature": "natural feature",
    }
    return synonyms.get(v, v)


def iter_places(path: Path, sustainability_index: Dict[str, Dict[str, str]]) -> Iterable[PlaceRecord]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zone = (row.get("Zone") or "").strip()
            state = (row.get("State") or "").strip()
            city = (row.get("City") or "").strip()
            name = (row.get("Name") or "").strip()
            place_type_raw = (row.get("Type") or row.get("Place Type") or "").strip()
            place_type = normalize_place_type(place_type_raw)
            best_time = (row.get("Best Time to visit") or row.get("Best Time to visit") or row.get("Best Time to visit") or row.get("Best Time to visit") or row.get("Best Time to visit") )
            if isinstance(best_time, str):
                best_time = best_time.strip() or None

            idx_key = place_type.lower()
            sust = sustainability_index.get(idx_key)
            is_sust = bool(sust["is_sustainable"]) if sust else False
            reason = sust["reason"] if sust else ""

            if not city or not name:
                continue

            yield PlaceRecord(
                zone=zone,
                state=state,
                city=city,
                name=name,
                place_type=place_type,
                is_sustainable=is_sust,
                sustainability_reason=reason,
                best_time=best_time,
            )


def iter_hotels_if_any(path: Path) -> Iterable[HotelRecord]:
    if not path.exists():
        return []
    hotels: List[HotelRecord] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            city = (row.get("City") or row.get("city") or "").strip()
            name = (row.get("Hotel Name") or row.get("name") or "").strip()
            tags_raw = (row.get("Sustainability Features") or row.get("sustainability_tags") or "").strip()
            certs_raw = (row.get("Certifications") or row.get("certifications") or "").strip()
            if not city or not name:
                continue
            tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []
            certs = [c.strip() for c in certs_raw.split(";") if c.strip()] if certs_raw else []
            hotels.append(HotelRecord(city=city, name=name, sustainability_tags=tags, certifications=certs))
    return hotels


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    places_csv = RAW_DIR / "Top Indian Places to Visit.csv"
    sustainability_csv = RAW_DIR / "place_type_sustainability_with_reasons.csv"
    hotels_csv = RAW_DIR / "hotel_details.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sustainability_index = load_sustainability_index(sustainability_csv)

    places_out = OUT_DIR / "places.jsonl"
    hotels_out = OUT_DIR / "hotels.jsonl"
    sustainability_index_out = OUT_DIR / "sustainability_index.json"

    # Write sustainability index
    with sustainability_index_out.open("w", encoding="utf-8") as f:
        json.dump(sustainability_index, f, ensure_ascii=False, indent=2)

    # Write places
    places_iter = iter_places(places_csv, sustainability_index)
    write_jsonl(places_out, (asdict(p) for p in places_iter))

    # Write hotels if any
    hotels_iter = iter_hotels_if_any(hotels_csv)
    write_jsonl(hotels_out, (asdict(h) for h in hotels_iter))

    print(f"Wrote: {places_out}")
    print(f"Wrote: {hotels_out}")
    print(f"Wrote: {sustainability_index_out}")


if __name__ == "__main__":
    main()


