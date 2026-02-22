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

Final Model Results
Problem Definition

This project predicts Kellgren–Lawrence (KL) grade (0–4) from knee X-ray images.
Since KL grades are ordered, the problem was modeled as an ordinal classification task rather than standard multiclass classification.

Model Architecture
Final Model: Ordinal CNN

Backbone: EfficientNet-B0 (pretrained)

Output: 4 ordinal logits (y>0, y>1, y>2, y>3)

Loss: BCEWithLogitsLoss

Optimizer: AdamW (with weight decay)

LR Scheduler: ReduceLROnPlateau

Early Stopping: patience = 3

Class imbalance handled using WeightedRandomSampler

Threshold tuning performed on validation set

Final Test Performance
Metric	Value
Accuracy	0.6961
Macro F1	0.6517

After threshold tuning, the model shows:

Strong performance on KL3 and KL4

Errors mostly occur between neighboring KL grades

No extreme misclassifications (e.g., KL0 → KL4)

This behavior is consistent with the ordinal nature of the problem.

Baseline Comparison
Model	Accuracy	Macro F1
CNN Embeddings + XGBoost	~0.49	~0.43
Ordinal CNN (Final)	~0.70	~0.65

Using an ordinal modeling approach significantly improved performance compared to the embedding + XGBoost baseline.

Next Steps

Fine-tune deeper EfficientNet layers

Add Quadratic Weighted Kappa metric

Integrate clinical/tabular features (hybrid model)

Explore regression-based KL severity prediction