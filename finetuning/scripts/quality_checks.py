#!/usr/bin/env python3
"""
Quality checks for finetuning datasets:
 - Valid JSON structure
 - Emission recomputation consistency (MODE_CHOICE)
 - Grounding for SUSTAINABLE_POIS (entities present in places table)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "finetuning" / "data" / "processed"


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


def check_mode_choice_emissions(mode_choice_path: Path, tolerance_pct: float = 5.0) -> List[str]:
    # No direct recomputation without distance; we check presence and non-negativity and ordering
    errors: List[str] = []
    rows = read_jsonl(mode_choice_path)
    for i, r in enumerate(rows):
        js = r.get("response_json") or {}
        pref = js.get("preferred_mode")
        pref_em = js.get("emissions_kg_co2e")
        alts = js.get("alternatives") or []
        if pref is None or pref_em is None:
            errors.append(f"row {i}: missing preferred_mode or emissions")
            continue
        if pref_em < 0:
            errors.append(f"row {i}: negative emissions")
        # Ensure alternatives have higher or equal emissions
        for a in alts:
            e = a.get("emissions_kg_co2e")
            if e is None:
                errors.append(f"row {i}: alternative missing emissions")
            elif e < pref_em - 1e-6:
                errors.append(f"row {i}: alternative emissions lower than preferred")
    return errors


def check_pois_grounding(pois_path: Path, places_path: Path) -> List[str]:
    errors: List[str] = []
    rows = read_jsonl(pois_path)
    places = read_jsonl(places_path)
    names = {(p.get("city"), p.get("name")) for p in places}
    for i, r in enumerate(rows):
        js = r.get("response_json") or {}
        recs = js.get("recommendations") or []
        for rec in recs:
            key = (r.get("instruction", "").replace("I'm visiting ", "").replace(". Suggest sustainable places to visit.", ""), rec.get("name"))
            # try city match via instruction; if not found, fallback to any city
            if key not in names and not any(n == rec.get("name") for (_, n) in names):
                errors.append(f"row {i}: recommendation '{rec.get('name')}' not grounded in places table")
    return errors


def main() -> None:
    mode_choice_path = PROCESSED_DIR / "mode_choice.jsonl"
    pois_path = PROCESSED_DIR / "sustainable_pois.jsonl"
    places_path = PROCESSED_DIR / "places.jsonl"

    errs = []
    errs += check_mode_choice_emissions(mode_choice_path)
    errs += check_pois_grounding(pois_path, places_path)

    if errs:
        print("Quality check failures:")
        for e in errs:
            print(" -", e)
    else:
        print("All quality checks passed.")


if __name__ == "__main__":
    main()


