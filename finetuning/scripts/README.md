## MLX Mistral fine-tuning, testing, and comparison

This guide explains how to fine-tune Mistral with MLX on Apple Silicon, test the trained adapter/model, and compare models using the scripts in this directory.

### Prerequisites
- macOS with Apple Silicon (MLX requirement)
- Python 3.10+ virtual environment activated
- Project root: `/Users/arpita/Documents/project`
- Install MLX requirements (from project root):
```bash
pip install -r finetuning/requirements_mlx.txt
```

## 1) Train (recommended path)
The recommended entrypoint is `run_mlx_training.sh`, which prepares data if needed and launches `train_mlx.py` with the config at `finetuning/configs/train_mlx.yaml`.

Run from the project root:
```bash
bash finetuning/scripts/run_mlx_training.sh
```
What it does:
- Verifies MLX installation and Apple Silicon
- Ensures processed datasets exist via `finetuning/scripts/build_dataset.py`
- Starts MLX LoRA training via `finetuning/scripts/train_mlx.py --config finetuning/configs/train_mlx.yaml`

Outputs (by default):
- Trained MLX LoRA adapter under `finetuning/models/mistral-mlx-lora`
- A `training_config.json` describing the run

### Alternate: Train directly with train_mlx.py
Use the config (recommended):
```bash
python finetuning/scripts/train_mlx.py --config finetuning/configs/train_mlx.yaml
```
Or customize via CLI flags (examples):
```bash
python finetuning/scripts/train_mlx.py \
  --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
  --output-dir finetuning/models/mistral-mlx-lora \
  --epochs 3 --lr 2e-4 --batch-size 1 --lora-r 8 --lora-alpha 16
```

### Optional: Colab-equivalent flow
If you want a script that mirrors an older Colab pipeline in MLX, see:
```bash
python finetuning/scripts/colab_to_mlx.py
```
It will prepare MLX data, run MLX LoRA training via the MLX CLI, and test the result.

## 2) Use the trained model in the backend (optional)
If you want the backend to use the MLX model:
1) Edit `backend/.env` and set:
```
USE_MLX=true
MLX_MODEL=finetuning/models/mistral-mlx-lora
```
2) Restart your backend service.

## 3) Test the trained MLX model
Basic load and sample inference test:
```bash
python finetuning/scripts/test_mlx_model.py --model-path finetuning/models/mistral-mlx-lora
```
Add a simple performance check (optional):
```bash
python finetuning/scripts/test_mlx_model.py \
  --model-path finetuning/models/mistral-mlx-lora \
  --benchmark
```

## 4) Compare models (GPT/ReAct vs Mistral variants)
Generate a side-by-side comparison on a set of structured prompts (runs the backend workflows locally, no HTTP):
```bash
python finetuning/scripts/compare_models_via_api.py \
  --out finetuning/results/model_comparison_local.json \
  --pairs 2
```
Notes:
- Produces `finetuning/results/model_comparison_local.json` containing prompts and per-model responses/timings.
- Variants included: GPT/ReAct path, Mistral community (base_4bit), and fine-tuned Mistral.

## 5) Evaluate the comparisons
Compute consistency/overlap metrics across models for mode choice and sustainable POIs:
```bash
python finetuning/scripts/evaluate_comparisons.py \
  --input finetuning/results/model_comparison_local.json \
  --output finetuning/results/model_quality_report.json
```
This writes a concise report to `finetuning/results/model_quality_report.json` and prints a short console summary.

## Optional utilities
- Parse MLX training logs to CSV and PNG plot:
```bash
python finetuning/scripts/parse_mlx_log.py <path/to/training_mlx.log>
```
- Watch/inspect a running MLX training process (informational):
  - `finetuning/scripts/watch_training.py`
  - `finetuning/scripts/monitor_training.py`
  - `finetuning/scripts/live_progress.py`
  - `finetuning/scripts/follow_training.py`
  - `finetuning/scripts/real_time_progress.py`

## Script map (what you likely need)
- Training: `run_mlx_training.sh`, `train_mlx.py`, `build_dataset.py`
- Test: `test_mlx_model.py`
- Compare: `compare_models_via_api.py`
- Evaluate: `evaluate_comparisons.py`
- Optional helpers: `colab_to_mlx.py`, `parse_mlx_log.py`
