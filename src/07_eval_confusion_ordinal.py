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
# Ayarlar (06 ile aynı)
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

test_df = pd.read_csv(DATA_DIR / "test.csv")
print("Test:", len(test_df))

# -----------------------
# Eval transform (06 ile aynı)
# -----------------------
eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -----------------------
# Dataset (relative path robust)
# -----------------------
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

        # CSV path relative ise proje köküne göre tamamla
        if not img_path.is_absolute() and self.base_dir is not None:
            img_path = self.base_dir / img_path

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Okunamayan görüntü: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), str(img_path)

test_ds = KneeDataset(test_df, transform=eval_tf, base_dir=BASE_DIR)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================================================
# ORDINAL MODEL (06 ile aynı)
# =========================================================
class OrdinalNet(nn.Module):
    """
    EfficientNet-B0 backbone + 4 ordinal logit (y>0, y>1, y>2, y>3)
    """
    def __init__(self, backbone_name="tf_efficientnet_b0", pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, NUM_CLASSES - 1)  # 4 output

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

def ordinal_logits_to_class(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,4)
    probs = sigmoid(logits)
    sınıf = kaç tane threshold geçti? (prob>0.5) sayısı -> 0..4
    """
    probs = torch.sigmoid(logits)
    passed = (probs > 0.5).sum(dim=1)
    return passed

# -----------------------
# Checkpoint yükle
# -----------------------
ckpt_path = RUNS_DIR / "best_ordinal.pt"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint yok: {ckpt_path} (Önce 06_train_ordinal.py çalışmalı)")

model = OrdinalNet(pretrained=False).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

print("Loaded:", ckpt_path)

# -----------------------
# Inference
# -----------------------
all_y_true = []
all_y_pred = []
all_paths  = []

with torch.no_grad():
    for images, labels, paths in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)                    # (B,4)
        preds = ordinal_logits_to_class(logits)   # (B,)

        all_y_true.append(labels.cpu().numpy())
        all_y_pred.append(preds.cpu().numpy())
        all_paths.extend(paths)

y_true = np.concatenate(all_y_true)
y_pred = np.concatenate(all_y_pred)

# -----------------------
# Raporlar
# -----------------------
print("\n=== Classification Report (Test / Ordinal) ===")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
print("\n=== Confusion Matrix (raw counts) ===")
print(cm)

# Normalized confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    display_labels=[f"KL{i}" for i in range(NUM_CLASSES)],
    normalize="true",
    xticks_rotation=45
)
plt.title("Ordinal Model - Confusion Matrix (Normalized)")
plt.tight_layout()

out_fig = RUNS_DIR / "confusion_ordinal_normalized.png"
plt.savefig(out_fig, dpi=200)
print("\nSaved figure:", out_fig)

# İstersen yanlış örnekleri kaydet
wrong_idx = np.where(y_true != y_pred)[0]
wrong_df = pd.DataFrame({
    "path": [all_paths[i] for i in wrong_idx],
    "y_true": y_true[wrong_idx],
    "y_pred": y_pred[wrong_idx],
})
out_csv = RUNS_DIR / "ordinal_misclassified.csv"
wrong_df.to_csv(out_csv, index=False)
print("Saved misclassified list:", out_csv)

plt.show()