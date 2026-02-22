from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

# -----------------------
# Ayarlar
# -----------------------
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
LR = 3e-4
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

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")
print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

# -----------------------
# Transformlar (seninkiyle aynı)
# -----------------------
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
])

eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -----------------------
# Dataset (seninkiyle aynı)
# -----------------------
class KneeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["label"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Okunamayan görüntü: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

train_ds = KneeDataset(train_df, transform=train_tf)
val_ds   = KneeDataset(val_df, transform=eval_tf)
test_ds  = KneeDataset(test_df, transform=eval_tf)

# -----------------------
# Weighted sampler (train kodunla aynı)
# -----------------------
label_counts = train_df["label"].value_counts().sort_index()
print("Train class counts:", label_counts.to_dict())

class_weights = 1.0 / label_counts.values
sample_weights = class_weights[train_df["label"].values]
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

# =========================================================
# ✅ ORDINAL MODEL: EfficientNet backbone + ordinal head
# =========================================================
class OrdinalNet(nn.Module):
    """
    EfficientNet-B0 backbone + 4 adet ordinal logit (y>0, y>1, y>2, y>3)
    """
    def __init__(self, backbone_name="tf_efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)  # feature extractor
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, NUM_CLASSES - 1)  # 4 output

    def forward(self, x):
        feats = self.backbone(x)     # (B, feat_dim)
        logits = self.head(feats)    # (B, 4)
        return logits

model = OrdinalNet(pretrained=True).to(device)

# =========================================================
# ✅ ORDINAL TARGET + LOSS (4 adet BCE)
# =========================================================
bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def labels_to_ordinal_targets(labels: torch.Tensor, num_classes=5) -> torch.Tensor:
    """
    labels: (B,) 0..4
    return: (B, num_classes-1) 0/1 target matrix
    örn KL2 -> [1,1,0,0]
    """
    # thresholds: 0,1,2,3
    thresholds = torch.arange(num_classes - 1, device=labels.device)  # (4,)
    # labels[:, None] > thresholds[None, :]
    targets = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
    return targets

def ordinal_logits_to_class(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,4)
    probs = sigmoid(logits)
    sınıf = kaç tane threshold geçti? (prob>0.5) sayısı
    """
    probs = torch.sigmoid(logits)
    passed = (probs > 0.5).sum(dim=1)   # 0..4
    return passed

# Metrics: bizim prediction sınıf 0..4 olacak
acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
f1_metric  = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)

def run_one_epoch(loader, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    acc_metric.reset()
    f1_metric.reset()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        targets = labels_to_ordinal_targets(labels, num_classes=NUM_CLASSES)  # (B,4)

        with torch.set_grad_enabled(train):
            logits = model(images)                # (B,4)
            loss = bce(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = ordinal_logits_to_class(logits)   # (B,)
        acc_metric.update(preds, labels)
        f1_metric.update(preds, labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = acc_metric.compute().item()
    f1  = f1_metric.compute().item()
    return avg_loss, acc, f1

# -----------------------
# Train loop
# -----------------------
best_val_f1 = -1.0
best_path = RUNS_DIR / "best_ordinal.pt"

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_f1 = run_one_epoch(train_loader, train=True)
    va_loss, va_acc, va_f1 = run_one_epoch(val_loader, train=False)

    print(f"Epoch {epoch:02d} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
          f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")

    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), best_path)
        print("  ✅ best saved:", best_path.name, "val_f1=", round(best_val_f1, 4))

# -----------------------
# Test
# -----------------------
model.load_state_dict(torch.load(best_path, map_location=device))
te_loss, te_acc, te_f1 = run_one_epoch(test_loader, train=False)
print("\nTEST RESULTS (ORDINAL)")
print("loss:", round(te_loss, 4), "acc:", round(te_acc, 4), "macro_f1:", round(te_f1, 4))
print("✅ checkpoint:", best_path)