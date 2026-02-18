# OSAIL KL Baseline (Knee X-ray)

Baseline pipeline for KL grading (0-4) using PyTorch + EfficientNet.

## Project structure
- `src/` scripts (splits, dataset test, baseline training)
- `data/` (ignored) labeled/unlabeled images + generated CSVs (not committed)
- `runs/` (ignored) saved models

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
