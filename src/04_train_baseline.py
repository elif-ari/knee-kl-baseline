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
# Ayarlar (CPU için optimize)
# -----------------------
IMG_SIZE = 224          # CPU için 384 yerine 224
BATCH_SIZE = 8          # CPU'da 8 iyi başlangıç
EPOCHS = 8              # önce kısa deneme
LR = 3e-4
NUM_CLASSES = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

device = torch.device("cpu")  # sende CPU

print("Device:", device)
print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

# -----------------------
# Dataset
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
# Weighted sampler (KL4 az diye)
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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -----------------------
# Model (EfficientNet-B0)
# -----------------------
model = timm.create_model("tf_efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
f1_metric  = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)

def run_one_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    acc_metric.reset()
    f1_metric.reset()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        acc_metric.update(preds, labels)
        f1_metric.update(preds, labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = acc_metric.compute().item()
    f1  = f1_metric.compute().item()
    return avg_loss, acc, f1

# -----------------------
# Train loop (best val F1 ile kaydet)
# -----------------------
best_val_f1 = -1.0
best_path = BASE_DIR / "runs" / "best_baseline.pt"
best_path.parent.mkdir(parents=True, exist_ok=True)

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
# Test (en iyi modeli yükle)
# -----------------------
model.load_state_dict(torch.load(best_path, map_location=device))
te_loss, te_acc, te_f1 = run_one_epoch(test_loader, train=False)
print("\nTEST RESULTS")
print("loss:", round(te_loss, 4), "acc:", round(te_acc, 4), "macro_f1:", round(te_f1, 4))

# -----------------------
# Basit örnek inference: KL + yüzde
# -----------------------
def severity_percent_from_pred(pred_kl: int) -> float:
    return (pred_kl / 4.0) * 100.0

model.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    logits = model(images.to(device))
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).cpu().numpy()

print("\nÖrnek tahminler (ilk 8):")
for i in range(min(8, len(preds))):
    print(f"GT={int(labels[i])}  PRED={int(preds[i])}  severity%={severity_percent_from_pred(int(preds[i])):.1f}")
