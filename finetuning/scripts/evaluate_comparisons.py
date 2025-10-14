#!/usr/bin/env python3
"""
Evaluate quality/consistency of multi-model comparisons from model_comparison_all.json.

Checks:
- MODE_CHOICE: consistency of preferred mode across models (normalized to canonical ids).
- SUSTAINABLE_POIS: overlap of recommended place names across models (Jaccard and exact match).

Usage:
  python finetuning/scripts/evaluate_comparisons.py \
    --input finetuning/results/model_comparison_local.json \
    --output finetuning/results/model_quality_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --- MODE NORMALIZATION ---

_MODE_CANONICAL = {
    "train_electric",
    "train_diesel",
    "bus_shared",
    "electric_car",
    "diesel_car",
    "petrol_car",
    "flight",
    "bicycle",
    "walking",
}

_MODE_SYNONYMS: List[Tuple[str, str]] = [
    ("train electric", "train_electric"),
    ("electric train", "train_electric"),
    ("electric railway", "train_electric"),
    ("train diesel", "train_diesel"),
    ("diesel train", "train_diesel"),
    ("train", "train_electric"),  # default train → electric variant
    ("shared bus", "bus_shared"),
    ("coach", "bus_shared"),
    ("bus", "bus_shared"),
    ("ev", "electric_car"),
    ("electric car", "electric_car"),
    ("petrol car", "petrol_car"),
    ("diesel car", "diesel_car"),
    ("plane", "flight"),
    ("flight", "flight"),
    ("air", "flight"),
    ("walk", "walking"),
    ("walking", "walking"),
    ("bicycle", "bicycle"),
    ("bike", "bicycle"),
]


def normalize_mode(text: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", text or "").strip().lower()
    # prefer longest phrases first
    for key, canon in sorted(_MODE_SYNONYMS, key=lambda kv: len(kv[0]), reverse=True):
        if key in s:
            return canon
    return None


def extract_mode_from_response(resp: str) -> Optional[str]:
    if not resp:
        return None
    # Prefer mass-transit / motorized modes over human-powered when multiple are present
    preferred_order = [
        "train_electric", "train_diesel", "bus_shared",
        "electric_car", "petrol_car", "diesel_car", "flight",
        "bicycle", "walking",
    ]
    # Try explicit JSON-like key matches
    m = re.search(r"preferred[_\s-]*mode[^:]*[:\-]\s*\"?([a-zA-Z_\s]+)\"?", resp, flags=re.I)
    if m:
        nm = normalize_mode(m.group(1))
        if nm:
            return nm
    # Try common phrasing: "I recommend <mode>" or "<Mode> is the most ..."
    m2 = re.search(r"recommend(?:ing)?\s+(?:the\s+)?([a-zA-Z\s]+)", resp, flags=re.I)
    if m2:
        nm = normalize_mode(m2.group(1))
        if nm:
            return nm
    m3 = re.search(r"^\s*([A-Z][a-zA-Z\s]+?)\s+is\s+the\s+most\s+", resp, flags=re.I | re.M)
    if m3:
        nm = normalize_mode(m3.group(1))
        if nm:
            return nm
    # Fallback: scan for all candidates, then pick by preferred_order
    candidates: Set[str] = set()
    s = re.sub(r"\s+", " ", resp or "").lower()
    for key, canon in sorted(_MODE_SYNONYMS, key=lambda kv: len(kv[0]), reverse=True):
        if key in s:
            candidates.add(canon)
    for mode in preferred_order:
        if mode in candidates:
            return mode
    return None


# --- PLACE NAME EXTRACTION ---

def _clean_place_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name or "").strip()
    # strip trailing punctuation
    name = re.sub(r"[\s\-–—:,;.]+$", "", name)
    # strip markdown bold/italic/code
    name = re.sub(r"(\*\*|__|_+|`)(.*?)\1", r"\2", name)
    return name


def extract_places_from_response(resp: str, top_k: int = 5) -> List[str]:
    if not resp:
        return []
    places: List[str] = []
    # Common field labels to ignore when bolded or listed
    stop_labels: Set[str] = {
        "type",
        "rating",
        "google review rating",
        "sustainability reason",
        "time needed",
        "entrance fee",
        "best time to visit",
        "dslr allowed",
        "google reviews lakhs",
        "establishment year",
        "zone",
        "state",
        "category",
        "description",
        "sustainability",
        "additional attractions",
        "recommendations",
        "recommendation",
        "location",
        "opening hours",
    }
    
    # Phrases that indicate this is not a place name
    stop_phrases = {
        "these places",
        "visit early",
        "plan for",
        "enjoy your",
        "these recommendations",
        "offer unique",
        "contribute less",
        "environmental degradation",
        "compared to other",
        "attractions",
    }
    
    lines = resp.splitlines()

    # 1) Bolded names (common style: **Name**)
    for line in lines:
        for m in re.finditer(r"\*\*([^*]{2,})\*\*", line):
            cand = _clean_place_name(m.group(1))
            if not cand:
                continue
            if cand.lower() in stop_labels:
                continue
            # Skip if it's a long sentence (>6 words) or contains stop phrases
            if len(cand.split()) > 6:
                continue
            if any(phrase in cand.lower() for phrase in stop_phrases):
                continue
            places.append(cand)

    # 2) Enumerated/bulleted lists: require name-first schema; capture segment before ':' or ' - '
    bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[\).])\s+(.+)")
    for line in lines:
        b = bullet_re.match(line)
        if b:
            item = _clean_place_name(b.group(1))
            # stop at colon or dash
            item = re.split(r"\s*[:\-–—]\s*", item, maxsplit=1)[0]
            # truncate parens
            item = re.split(r"\s*\(.*\)$", item)[0]
            item = _clean_place_name(item)
            if not item:
                continue
            if item.lower() in stop_labels:
                continue
            # Skip long sentences or those with stop phrases
            if len(item.split()) > 6:
                continue
            if any(phrase in item.lower() for phrase in stop_phrases):
                continue
            # discard bullets that start with common non-name phrases
            if re.match(r"^(i'm sorry|if you're interested|to reach|here (are|is)|bulleted points|these places|for more information)\b", item, flags=re.I):
                continue
            # discard known lead-in phrases that appear as pseudo-names
            if re.match(r"^for eco(\s|-)?friendly\b", item, flags=re.I) or re.match(r"^for eco\b", item, flags=re.I):
                continue
            places.append(item)

    # 3) Fallback: look for capitalized multi-word phrases (very heuristic)
    if not places:
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", resp):
            cand = m.group(1)
            if len(cand.split()) >= 2 and len(cand.split()) <= 5:  # prefer multi-word but not too long
                cleaned = _clean_place_name(cand)
                # Skip if contains stop phrases
                if any(phrase in cleaned.lower() for phrase in stop_phrases):
                    continue
                places.append(cleaned)

    # Deduplicate keeping order
    seen: Set[str] = set()
    uniq: List[str] = []
    for p in places:
        key = p.lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
        if len(uniq) >= top_k:
            break
    return uniq


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union > 0 else 0.0


def evaluate(input_path: Path, output_path: Path) -> Dict:
    data = read_json(input_path)

    prompts_map: Dict[str, str] = data.get("prompts", {})
    # Accept multiple aliases across file variants
    candidate_keys = [
        "gpt",
        "mistral_community",
        "mistral_lora",
        "mistral_finetuned",
        "finetuned",
        "mlx",
    ]
    models = [k for k in candidate_keys if k in data]

    report: Dict = {
        "input": str(input_path),
        "models": models,
        "summary": {
            "mode_choice_total": 0,
            "mode_choice_consistent": 0,
            "pois_total": 0,
            "pois_exact_match": 0,
            "pois_avg_jaccard": 0.0,
        },
        "details": []
    }

    pois_jaccards: List[float] = []

    # Iterate prompts by index order
    for i_key in sorted(prompts_map.keys(), key=lambda x: int(x.split("_")[-1])):
        prompt_text = prompts_map[i_key]
        task = "MODE_CHOICE" if "MODE_CHOICE" in prompt_text else ("SUSTAINABLE_POIS" if "SUSTAINABLE_POIS" in prompt_text else "UNKNOWN")

        entry: Dict = {
            "prompt_id": i_key,
            "task": task,
            "prompt": prompt_text,
            "models": {}
        }

        # Collect model responses
        for m in models:
            m_obj = data.get(m, {}).get(i_key, {})
            resp = m_obj.get("response", "")
            time_s = m_obj.get("time")
            entry["models"][m] = {"time": time_s, "response": resp}

        # Evaluate by task
        if task == "MODE_CHOICE":
            report["summary"]["mode_choice_total"] += 1
            modes: Dict[str, Optional[str]] = {}
            for m in models:
                modes[m] = extract_mode_from_response(entry["models"][m].get("response", ""))
                entry["models"][m]["extracted_mode"] = modes[m]

            present = [mv for mv in modes.values() if mv]
            consistent = len(set(present)) == 1 and len(present) >= 2  # consistent across ≥2 models
            entry["mode_consistent_all"] = consistent
            entry["mode_values"] = modes
            if consistent:
                report["summary"]["mode_choice_consistent"] += 1

        elif task == "SUSTAINABLE_POIS":
            report["summary"]["pois_total"] += 1
            places_sets: Dict[str, Set[str]] = {}
            for m in models:
                places = extract_places_from_response(entry["models"][m].get("response", ""), top_k=5)
                entry["models"][m]["extracted_places"] = places
                places_sets[m] = {p.lower() for p in places if p}

            # Pairwise Jaccard and exact match flag
            pairs = []
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    mi, mj = models[i], models[j]
                    ji = jaccard(places_sets[mi], places_sets[mj])
                    pairs.append(ji)
            avg_j = sum(pairs) / len(pairs) if pairs else 0.0
            pois_jaccards.append(avg_j)
            entry["pois_avg_jaccard"] = avg_j
            exact_match = all(ps == places_sets[models[0]] for ps in places_sets.values()) if models else False
            entry["pois_exact_match_all"] = exact_match
            if exact_match:
                report["summary"]["pois_exact_match"] += 1

        # Append entry
        report["details"].append(entry)

    # Aggregate
    if pois_jaccards:
        report["summary"]["pois_avg_jaccard"] = round(sum(pois_jaccards) / len(pois_jaccards), 3)

    write_json(output_path, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate multi-model comparison outputs")
    p.add_argument("--input", default="finetuning/results/compare_models_via_api.json")
    p.add_argument("--output", default="finetuning/results/model_quality_report.json")
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    report = evaluate(inp, outp)
    # Console summary
    s = report["summary"]
    print("Quality Summary")
    print("=" * 40)
    print(f"MODE_CHOICE: {s['mode_choice_consistent']}/{s['mode_choice_total']} prompts consistent across models")
    print(f"SUSTAINABLE_POIS exact matches: {s['pois_exact_match']}/{s['pois_total']}")
    print(f"SUSTAINABLE_POIS avg Jaccard: {s['pois_avg_jaccard']:.3f}")
    print(f"Saved report to: {outp}")


if __name__ == "__main__":
    main()


