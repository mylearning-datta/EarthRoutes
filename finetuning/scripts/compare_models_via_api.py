#!/usr/bin/env python3
"""
Compare models by directly calling backend workflow functions (no HTTP).

Outputs a unified JSON with identical prompts evaluated across:
- GPT/ReAct path
- Mistral community (base_4bit)
- Mistral fine-tuned (colab_finetune)

Usage:
  python finetuning/scripts/compare_models_via_api.py \
    --out finetuning/results/model_comparison_local.json \
    --pairs 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List
import contextlib
import io
import logging

from dotenv import load_dotenv


def load_env(project_root: Path) -> None:
    try:
        be = project_root / "backend" / ".env"
        if be.exists():
            load_dotenv(dotenv_path=str(be), override=False)
        root_env = project_root / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except Exception:
        pass


def get_indian_cities_local() -> List[str]:
    # Local fallback city list
    return [
        "Delhi", "Mumbai", "Bangalore", "Hyderabad", "Kolkata",
        "Chennai", "Pune", "Ahmedabad", "Jaipur", "Surat",
    ]


def build_indian_prompts(cities: List[str], n_pairs: int = 1) -> List[str]:
    rng = random.Random(42)
    #if len(cities) < 3:
    cities_1 = ["Delhi", "Mumbai", "Bangalore","Chennai","Pune","Ahmedabad", "Jaipur","Hyderabad"]
    cities_places = ["Kolkata", "Hyderabad", "Mumbai", "Kerala","Bengaluru","Ahmedabad"]
    prompts: List[str] = []
    for _ in range(max(1, n_pairs)):
        a, b = rng.sample(cities_1, 2)
        # MODE_CHOICE
        prompts.append(
            f"### TASK: MODE_CHOICE\n### INSTRUCTION:\nChoose the most sustainable transport mode from {a} to {b}.\n### RESPONSE:"
        )
        # SUSTAINABLE_POIS
        c = rng.choice(cities_places)
        prompts.append(
            f"### TASK: SUSTAINABLE_POIS\n### INSTRUCTION:\nRecommend eco-friendly places in {c}.\n### RESPONSE:"
        )
    return prompts


def call_chat(*args, **kwargs) -> Dict:  # Backwards-compat shim (should not be used now)
    return {"ok": False, "time": 0.0, "body": {"error": "HTTP path removed; use local calls"}}


def call_local(project_root: Path, message: str, finetuned: bool = False, variant: str | None = None) -> Dict:
    """Call backend workflow functions directly (no HTTP)."""
    import time as _time
    start = _time.time()
    try:
        import sys as _sys
        import os as _os
        import importlib as _importlib
        import importlib.util as _importlib_util
        # Ensure backend is importable
        backend_path = str(project_root / "backend")
        if backend_path not in _sys.path:
            _sys.path.insert(0, backend_path)
        module = None
        try:
            module = _importlib.import_module("workflows.advanced_react_agent")
        except Exception:
            # Fallback to loading by file path if package import fails
            file_path = str(project_root / "backend" / "workflows" / "advanced_react_agent.py")
            spec = _importlib_util.spec_from_file_location("advanced_react_agent", file_path)
            if spec and spec.loader:
                module = _importlib_util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
            else:
                raise ImportError(f"Cannot load module from {file_path}")

        process_travel_query_advanced_react = getattr(module, "process_travel_query_advanced_react")
        process_travel_query_advanced_react_finetuned = getattr(module, "process_travel_query_advanced_react_finetuned")
        if finetuned:
            body = process_travel_query_advanced_react_finetuned(message, variant=variant)
        else:
            body = process_travel_query_advanced_react(message)
        elapsed = _time.time() - start
        return {"ok": True, "time": elapsed, "body": body}
    except Exception as e:
        elapsed = _time.time() - start
        return {"ok": False, "time": elapsed, "body": {"error": str(e)}}


def main() -> None:
    p = argparse.ArgumentParser(description="Compare models via backend APIs")
    p.add_argument("--out", default="finetuning/results/model_comparison_local.json")
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--log", default="logs/compare_models.log", help="Path to write detailed run logs")
    args = p.parse_args()

    run_start_ts = time.time()

    project_root = Path(__file__).resolve().parents[2]
    load_env(project_root)

    # Configure file logging
    try:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
            handlers=[
                logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
            ]
        )
        run_logger = logging.getLogger("compare_models_via_api")
        run_logger.info("==== New compare_models run started ====")
        run_logger.info(f"Output file: {args.out} | Pairs: {args.pairs}")
    except Exception:
        run_logger = logging.getLogger("compare_models_via_api")

    # Always local now
    cities = get_indian_cities_local()
    prompts = build_indian_prompts(cities, n_pairs=args.pairs)

    results: Dict[str, Dict] = {
        "prompts": {f"prompt_{i}": p for i, p in enumerate(prompts)},
        "gpt": {},
        "mistral_community": {},
        "mistral_finetuned": {}
    }

    total = len(prompts)
    for i, prompt in enumerate(prompts):
        task = "MODE_CHOICE" if "MODE_CHOICE" in prompt else ("SUSTAINABLE_POIS" if "SUSTAINABLE_POIS" in prompt else "UNKNOWN")
        print(f"[{i+1}/{total}] Processing prompt ({task})...")
        try:
            run_logger.info(f"Processing prompt {i+1}/{total} ({task})")
        except Exception:
            pass

        # Prepare suppression of noisy output from backend/agents
        loggers_to_quiet = [
            "workflows.advanced_react_agent",
            "utils.postgres_database",
            "services.finetuned_model_service",
            "httpx",
            "langchain",
            "langchain_core",
        ]
        previous_levels: Dict[str, int] = {}
        for ln in loggers_to_quiet:
            lg = logging.getLogger(ln)
            previous_levels[ln] = lg.level
            lg.setLevel(logging.ERROR)

        # GPT/ReAct path (local)
        print("  - Calling GPT/ReAct ...", end="", flush=True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            gpt_res = call_local(project_root, prompt, finetuned=False)
        print(f" done ({'OK' if gpt_res.get('ok') else 'FAIL'}, {gpt_res.get('time', 0.0):.2f}s)")
        try:
            run_logger.info(
                f"[timer] GPT/ReAct time={gpt_res.get('time', 0.0):.3f}s ok={gpt_res.get('ok')}"
            )
        except Exception:
            pass
        if not gpt_res.get("ok"):
            err = gpt_res.get("body", {}).get("error")
            if err:
                print(f"    Error: {err}")

        # Mistral community (base_4bit) (local)
        print("  - Calling Mistral community (base_4bit) ...", end="", flush=True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            comm_res = call_local(project_root, prompt, finetuned=True, variant="base_4bit")
        print(f" done ({'OK' if comm_res.get('ok') else 'FAIL'}, {comm_res.get('time', 0.0):.2f}s)")
        try:
            run_logger.info(
                f"[timer] Mistral community time={comm_res.get('time', 0.0):.3f}s ok={comm_res.get('ok')}"
            )
        except Exception:
            pass
        if not comm_res.get("ok"):
            err = comm_res.get("body", {}).get("error")
            if err:
                print(f"    Error: {err}")

        # Mistral fine-tuned (colab_finetune) (local)
        print("  - Calling Mistral fine-tuned (colab_finetune) ...", end="", flush=True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ft_res = call_local(project_root, prompt, finetuned=True, variant="colab_finetune")
        print(f" done ({'OK' if ft_res.get('ok') else 'FAIL'}, {ft_res.get('time', 0.0):.2f}s)")
        try:
            run_logger.info(
                f"[timer] Mistral finetuned time={ft_res.get('time', 0.0):.3f}s ok={ft_res.get('ok')}"
            )
        except Exception:
            pass
        if not ft_res.get("ok"):
            err = ft_res.get("body", {}).get("error")
            if err:
                print(f"    Error: {err}")

        # Restore logger levels
        for ln, lvl in previous_levels.items():
            logging.getLogger(ln).setLevel(lvl)

        # Normalize bodies (store only response + time)
        def extract_text(body: Dict) -> str:
            if not isinstance(body, dict):
                return str(body)
            if "response" in body:
                return body.get("response") or ""
            return json.dumps(body, ensure_ascii=False)

        results["gpt"][f"prompt_{i}"] = {
            "time": round(gpt_res.get("time", 0.0), 3),
            "response": extract_text(gpt_res.get("body", {})),
            "ok": gpt_res.get("ok", False),
        }
        results["mistral_community"][f"prompt_{i}"] = {
            "time": round(comm_res.get("time", 0.0), 3),
            "response": extract_text(comm_res.get("body", {})),
            "ok": comm_res.get("ok", False),
        }
        results["mistral_finetuned"][f"prompt_{i}"] = {
            "time": round(ft_res.get("time", 0.0), 3),
            "response": extract_text(ft_res.get("body", {})),
            "ok": ft_res.get("ok", False),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved comparison to: {out_path}")
    try:
        run_logger.info(f"Saved comparison to: {out_path}")
        run_logger.info(f"[timer] total_run_time={time.time() - run_start_ts:.3f}s")
        run_logger.info("==== compare_models run completed ====")
    except Exception:
        pass


if __name__ == "__main__":
    main()


