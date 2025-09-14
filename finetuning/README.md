Finetuning Workspace (MODE_CHOICE, SUSTAINABLE_POIS)

This workspace builds datasets and trains a lightweight LoRA model for two atomic tasks:
- MODE_CHOICE: choose the lowest-carbon feasible transport mode between cities.
- SUSTAINABLE_POIS: recommend sustainable places to visit in a destination.

Quick start
- Extract facts: `python finetuning/scripts/extract_facts.py`
- Build datasets: `python finetuning/scripts/build_dataset.py`
- Run quality checks: `python finetuning/scripts/quality_checks.py`

Outputs are written to `finetuning/data/processed`.

