#!/usr/bin/env python3
"""
Download MLX-quantized Mistral model from Hugging Face into finetuning/models.

Default target:
  finetuning/models/mistral-7b-instruct-4bit-mlx

Usage:
  python finetuning/scripts/download_mlx_model.py \
      --repo mistralai/Mistral-7B-Instruct-v0.2-4bit-mlx \
      --dest finetuning/models/mistral-7b-instruct-4bit-mlx

Requires: huggingface_hub
"""
import argparse
from pathlib import Path
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print("huggingface_hub is required. Install via: pip install huggingface_hub", file=sys.stderr)
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MLX model from Hugging Face")
    parser.add_argument(
        "--repo",
        default="mistralai/Mistral-7B-Instruct-v0.2-4bit-mlx",
        help="Hugging Face repo id to download",
    )
    parser.add_argument(
        "--dest",
        default="finetuning/models/mistral-7b-instruct-4bit-mlx",
        help="Destination directory to store the model",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional allow patterns for files to download",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision / commit / tag to pin",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest_path = Path(args.dest).expanduser().resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.repo}' to '{dest_path}' ...")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest_path),
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_patterns,
        revision=args.revision,
        resume_download=True,
    )
    print("Download complete.")


if __name__ == "__main__":
    main()


