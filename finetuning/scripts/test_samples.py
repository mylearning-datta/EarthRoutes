#!/usr/bin/env python3
"""
Test script to show sample responses from the generated datasets.
Demonstrates the training format and response quality.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "finetuning" / "data" / "processed"


def read_jsonl(path: Path) -> List[Dict]:
    """Read JSONL file and return list of dictionaries."""
    items: List[Dict] = []
    if not path.exists():
        print(f"File not found: {path}")
        return items
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def format_training_example(example: Dict) -> str:
    """Format example in training format (instruction + response)."""
    task = example.get("task", "UNKNOWN")
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")
    response_json = example.get("response_json", {})
    
    # Create training format
    prompt = f"### TASK: {task}\n### INSTRUCTION:\n{instruction}"
    if context:
        prompt += f"\n### CONTEXT:\n{context}"
    prompt += f"\n### RESPONSE:\n{response}"
    
    if response_json:
        prompt += f"\n### RESPONSE_JSON:\n{json.dumps(response_json, indent=2)}"
    
    return prompt


def show_mode_choice_samples(count: int = 3):
    """Show sample MODE_CHOICE examples."""
    print("=" * 80)
    print("MODE_CHOICE SAMPLES")
    print("=" * 80)
    
    examples = read_jsonl(PROCESSED_DIR / "mode_choice.jsonl")
    if not examples:
        print("No MODE_CHOICE examples found!")
        return
    
    # Show random samples
    samples = random.sample(examples, min(count, len(examples)))
    
    for i, example in enumerate(samples, 1):
        print(f"\n--- SAMPLE {i} ---")
        print(format_training_example(example))
        print()


def show_sustainable_pois_samples(count: int = 3):
    """Show sample SUSTAINABLE_POIS examples."""
    print("=" * 80)
    print("SUSTAINABLE_POIS SAMPLES")
    print("=" * 80)
    
    examples = read_jsonl(PROCESSED_DIR / "sustainable_pois.jsonl")
    if not examples:
        print("No SUSTAINABLE_POIS examples found!")
        return
    
    # Show random samples
    samples = random.sample(examples, min(count, len(examples)))
    
    for i, example in enumerate(samples, 1):
        print(f"\n--- SAMPLE {i} ---")
        print(format_training_example(example))
        print()


def show_dataset_stats():
    """Show dataset statistics."""
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    mode_choice = read_jsonl(PROCESSED_DIR / "mode_choice.jsonl")
    sustainable_pois = read_jsonl(PROCESSED_DIR / "sustainable_pois.jsonl")
    places = read_jsonl(PROCESSED_DIR / "places.jsonl")
    
    print(f"MODE_CHOICE examples: {len(mode_choice)}")
    print(f"SUSTAINABLE_POIS examples: {len(sustainable_pois)}")
    print(f"Total places extracted: {len(places)}")
    
    # Show task distribution
    if mode_choice:
        print(f"\nMODE_CHOICE task distribution:")
        print(f"  - All examples: {len(mode_choice)}")
    
    if sustainable_pois:
        print(f"\nSUSTAINABLE_POIS task distribution:")
        print(f"  - All examples: {len(sustainable_pois)}")
    
    # Show emission factor ranges
    if mode_choice:
        emissions = []
        for ex in mode_choice:
            json_resp = ex.get("response_json", {})
            if "emissions_kg_co2e" in json_resp:
                emissions.append(json_resp["emissions_kg_co2e"])
        
        if emissions:
            print(f"\nEmission ranges in MODE_CHOICE:")
            print(f"  - Min: {min(emissions):.2f} kg CO2e")
            print(f"  - Max: {max(emissions):.2f} kg CO2e")
            print(f"  - Avg: {sum(emissions)/len(emissions):.2f} kg CO2e")


def show_response_quality():
    """Show response quality analysis."""
    print("=" * 80)
    print("RESPONSE QUALITY ANALYSIS")
    print("=" * 80)
    
    mode_choice = read_jsonl(PROCESSED_DIR / "mode_choice.jsonl")
    sustainable_pois = read_jsonl(PROCESSED_DIR / "sustainable_pois.jsonl")
    
    # Analyze MODE_CHOICE
    if mode_choice:
        print("MODE_CHOICE Quality:")
        valid_json = 0
        has_alternatives = 0
        has_justification = 0
        
        for ex in mode_choice:
            json_resp = ex.get("response_json", {})
            if json_resp:
                valid_json += 1
                if "alternatives" in json_resp and json_resp["alternatives"]:
                    has_alternatives += 1
                if "justification" in json_resp and json_resp["justification"]:
                    has_justification += 1
        
        print(f"  - Valid JSON responses: {valid_json}/{len(mode_choice)} ({100*valid_json/len(mode_choice):.1f}%)")
        print(f"  - Has alternatives: {has_alternatives}/{len(mode_choice)} ({100*has_alternatives/len(mode_choice):.1f}%)")
        print(f"  - Has justification: {has_justification}/{len(mode_choice)} ({100*has_justification/len(mode_choice):.1f}%)")
    
    # Analyze SUSTAINABLE_POIS
    if sustainable_pois:
        print("\nSUSTAINABLE_POIS Quality:")
        valid_json = 0
        has_recommendations = 0
        has_tips = 0
        
        for ex in sustainable_pois:
            json_resp = ex.get("response_json", {})
            if json_resp:
                valid_json += 1
                if "recommendations" in json_resp and json_resp["recommendations"]:
                    has_recommendations += 1
                if "tips" in json_resp and json_resp["tips"]:
                    has_tips += 1
        
        print(f"  - Valid JSON responses: {valid_json}/{len(sustainable_pois)} ({100*valid_json/len(sustainable_pois):.1f}%)")
        print(f"  - Has recommendations: {has_recommendations}/{len(sustainable_pois)} ({100*has_recommendations/len(sustainable_pois):.1f}%)")
        print(f"  - Has tips: {has_tips}/{len(sustainable_pois)} ({100*has_tips/len(sustainable_pois):.1f}%)")


def main():
    """Main test function."""
    print("FINETUNING DATASET TEST SCRIPT")
    print("=" * 80)
    
    # Set random seed for reproducible samples
    random.seed(42)
    
    # Show statistics
    show_dataset_stats()
    
    # Show quality analysis
    show_response_quality()
    
    # Show sample responses
    show_mode_choice_samples(2)
    show_sustainable_pois_samples(2)
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
