#!/usr/bin/env python3
"""
Status check script for the finetuning pipeline.
Shows what's implemented, what's missing, and current state.
"""

import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "finetuning" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "finetuning" / "models"


def check_datasets() -> Dict:
    """Check dataset status."""
    status = {
        "places": {"exists": False, "count": 0},
        "hotels": {"exists": False, "count": 0},
        "mode_choice": {"exists": False, "count": 0},
        "sustainable_pois": {"exists": False, "count": 0},
        "sustainability_index": {"exists": False}
    }
    
    # Check places
    places_file = PROCESSED_DIR / "places.jsonl"
    if places_file.exists():
        with places_file.open("r") as f:
            status["places"]["count"] = sum(1 for line in f if line.strip())
        status["places"]["exists"] = True
    
    # Check hotels
    hotels_file = PROCESSED_DIR / "hotels.jsonl"
    if hotels_file.exists():
        with hotels_file.open("r") as f:
            status["hotels"]["count"] = sum(1 for line in f if line.strip())
        status["hotels"]["exists"] = True
    
    # Check mode_choice
    mode_choice_file = PROCESSED_DIR / "mode_choice.jsonl"
    if mode_choice_file.exists():
        with mode_choice_file.open("r") as f:
            status["mode_choice"]["count"] = sum(1 for line in f if line.strip())
        status["mode_choice"]["exists"] = True
    
    # Check sustainable_pois
    pois_file = PROCESSED_DIR / "sustainable_pois.jsonl"
    if pois_file.exists():
        with pois_file.open("r") as f:
            status["sustainable_pois"]["count"] = sum(1 for line in f if line.strip())
        status["sustainable_pois"]["exists"] = True
    
    # Check sustainability_index
    index_file = PROCESSED_DIR / "sustainability_index.json"
    status["sustainability_index"]["exists"] = index_file.exists()
    
    return status


def check_models() -> Dict:
    """Check model status."""
    status = {
        "trained_model": {"exists": False, "path": None},
        "training_config": {"exists": False}
    }
    
    # Check for trained model
    model_path = MODELS_DIR / "travel-sustainability-lora"
    if model_path.exists():
        status["trained_model"]["exists"] = True
        status["trained_model"]["path"] = str(model_path)
        
        # Check for training config
        config_file = model_path / "training_config.json"
        if config_file.exists():
            status["training_config"]["exists"] = True
            with config_file.open("r") as f:
                config = json.load(f)
                status["training_config"]["details"] = config
    
    return status


def check_scripts() -> Dict:
    """Check script implementation status."""
    scripts_dir = PROJECT_ROOT / "finetuning" / "scripts"
    
    status = {
        "extract_facts": {"exists": False, "implemented": False},
        "build_dataset": {"exists": False, "implemented": False},
        "quality_checks": {"exists": False, "implemented": False},
        "train_sft": {"exists": False, "implemented": False},
        "inference": {"exists": False, "implemented": False},
        "test_samples": {"exists": False, "implemented": False},
        "test_natural_inputs": {"exists": False, "implemented": False}
    }
    
    for script_name in status.keys():
        script_file = scripts_dir / f"{script_name}.py"
        status[script_name]["exists"] = script_file.exists()
        
        if script_file.exists():
            # Check if it's more than just a placeholder
            with script_file.open("r") as f:
                content = f.read()
                # Simple check: if it has more than 50 lines and contains actual logic
                status[script_name]["implemented"] = (
                    len(content.split('\n')) > 50 and 
                    ('def ' in content or 'class ' in content) and
                    'pass' not in content
                )
    
    return status


def print_status():
    """Print comprehensive status report."""
    print("FINETUNING PIPELINE STATUS")
    print("=" * 60)
    
    # Dataset status
    print("\n📊 DATASETS:")
    print("-" * 30)
    dataset_status = check_datasets()
    
    for name, info in dataset_status.items():
        if name == "sustainability_index":
            status_icon = "✅" if info["exists"] else "❌"
            print(f"  {status_icon} {name}: {'Found' if info['exists'] else 'Missing'}")
        else:
            status_icon = "✅" if info["exists"] else "❌"
            count = info["count"] if info["exists"] else 0
            print(f"  {status_icon} {name}: {count} examples")
    
    # Model status
    print("\n🤖 MODELS:")
    print("-" * 30)
    model_status = check_models()
    
    if model_status["trained_model"]["exists"]:
        print(f"  ✅ Trained model: {model_status['trained_model']['path']}")
        if model_status["training_config"]["exists"]:
            config = model_status["training_config"]["details"]
            print(f"    - Examples trained on: {config.get('total_examples', 'Unknown')}")
            print(f"    - Epochs: {config.get('num_epochs', 'Unknown')}")
            print(f"    - Learning rate: {config.get('learning_rate', 'Unknown')}")
    else:
        print("  ❌ No trained model found")
    
    # Scripts status
    print("\n📝 SCRIPTS:")
    print("-" * 30)
    script_status = check_scripts()
    
    for name, info in script_status.items():
        if info["exists"] and info["implemented"]:
            print(f"  ✅ {name}.py: Implemented")
        elif info["exists"]:
            print(f"  ⚠️  {name}.py: Placeholder only")
        else:
            print(f"  ❌ {name}.py: Missing")
    
    # Next steps
    print("\n🚀 NEXT STEPS:")
    print("-" * 30)
    
    if not dataset_status["mode_choice"]["exists"]:
        print("  1. Run: python finetuning/scripts/extract_facts.py")
        print("  2. Run: python finetuning/scripts/build_dataset.py")
    elif not model_status["trained_model"]["exists"]:
        print("  1. Install dependencies: pip install transformers peft datasets torch")
        print("  2. Run: python finetuning/scripts/train_sft.py")
    else:
        print("  1. Test model: python finetuning/scripts/inference.py --mode test")
        print("  2. Interactive mode: python finetuning/scripts/inference.py --mode interactive")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_status()
