from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------
# Ayarlar
# -----------------------
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

val_df  = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
print("Val:", len(val_df))
print("Test:", len(test_df))

eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

class KneeDataset(Dataset):
    def __init__(self, df, transform=None, base_dir: Path = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        label = int(row["label"])

        if not img_path.is_absolute() and self.base_dir is not None:
            img_path = self.base_dir / img_path

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Okunamayan görüntü: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), str(img_path)

val_loader  = DataLoader(KneeDataset(val_df,  eval_tf, BASE_DIR),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(KneeDataset(test_df, eval_tf, BASE_DIR),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -----------------------
# Model
# -----------------------
class OrdinalNet(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b0", pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, NUM_CLASSES - 1)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

def probs_to_class(probs: torch.Tensor, thresholds=(0.5,0.5,0.5,0.5)) -> torch.Tensor:
    thr = torch.tensor(thresholds, device=probs.device).view(1, -1)
    return (probs > thr).sum(dim=1)

@torch.no_grad()
def collect_probs(loader):
    model.eval()
    all_probs, all_y, all_paths = [], [], []
    for images, labels, paths in loader:
        images = images.to(device)
        logits = model(images)          # (B,4)
        probs = torch.sigmoid(logits)   # (B,4)
        all_probs.append(probs.cpu().numpy())
        all_y.append(labels.numpy())
        all_paths.extend(paths)
    return np.concatenate(all_probs), np.concatenate(all_y), all_paths

def macro_f1_from_cm(cm):
    # macro F1 hesapla (sklearn kullanmadan hızlı)
    eps = 1e-9
    f1s = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    return float(np.mean(f1s))

def tune_thresholds(val_probs, val_y):
    """
    Çok basit ve etkili: her threshold için 0.2..0.8 arası tarıyoruz
    (ordinalde genelde yeterli)
    """
    grid = np.linspace(0.2, 0.8, 13)  # 0.2,0.25,...0.8
    best_thr = (0.5,0.5,0.5,0.5)
    best_score = -1.0

    for t0 in grid:
        for t1 in grid:
            for t2 in grid:
                for t3 in grid:
                    thr = (t0,t1,t2,t3)
                    pred = (val_probs > np.array(thr).reshape(1,-1)).sum(axis=1)
                    cm = confusion_matrix(val_y, pred, labels=list(range(NUM_CLASSES)))
                    score = macro_f1_from_cm(cm)
                    if score > best_score:
                        best_score = score
                        best_thr = thr
    return best_thr, best_score

# -----------------------
# Load checkpoint
# -----------------------
ckpt_path = RUNS_DIR / "best_ordinal.pt"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint yok: {ckpt_path} (Önce 06_train_ordinal.py çalışmalı)")

model = OrdinalNet(pretrained=False).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
print("Loaded:", ckpt_path)

# -----------------------
# 1) Val üzerinde threshold tuning
# -----------------------
val_probs, val_y, _ = collect_probs(val_loader)
best_thr, best_val_macro_f1 = tune_thresholds(val_probs, val_y)
print("\nBest thresholds (val):", best_thr)
print("Best val macro_f1 (approx):", round(best_val_macro_f1, 4))

# -----------------------
# 2) Test evaluate (tuned thresholds)
# -----------------------
test_probs, test_y, test_paths = collect_probs(test_loader)
test_pred = (test_probs > np.array(best_thr).reshape(1,-1)).sum(axis=1)

print("\n=== Classification Report (Test / Ordinal, tuned thresholds) ===")
print(classification_report(test_y, test_pred, digits=4, zero_division=0))

cm = confusion_matrix(test_y, test_pred, labels=list(range(NUM_CLASSES)))
print("\n=== Confusion Matrix (raw counts) ===")
print(cm)

disp = ConfusionMatrixDisplay.from_predictions(
    test_y,
    test_pred,
    labels=list(range(NUM_CLASSES)),
    display_labels=[f"KL{i}" for i in range(NUM_CLASSES)],
    normalize="true",
    xticks_rotation=45
)
plt.title("Ordinal Model - Confusion Matrix (Normalized, Tuned Thresholds)")
plt.tight_layout()

out_fig = RUNS_DIR / "confusion_ordinal_tuned.png"
plt.savefig(out_fig, dpi=200)
print("\nSaved figure:", out_fig)

# misclassified list
wrong_idx = np.where(test_y != test_pred)[0]
wrong_df = pd.DataFrame({
    "path": [test_paths[i] for i in wrong_idx],
    "y_true": test_y[wrong_idx],
    "y_pred": test_pred[wrong_idx],
})
out_csv = RUNS_DIR / "ordinal_misclassified_tuned.csv"
wrong_df.to_csv(out_csv, index=False)
print("Saved misclassified list:", out_csv)

plt.show()