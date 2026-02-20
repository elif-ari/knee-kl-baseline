from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -----------------------
# Ayarlar (train ile uyumlu)
# -----------------------
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 5

CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
CKPT_PATH = RUNS_DIR / "best_baseline.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Eval transform (train kodunla aynı)
# -----------------------
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

# -----------------------
# CM çizdirme
# -----------------------
def plot_cm(cm, class_names, normalize=False, title="Confusion Matrix"):
    cm_plot = cm.astype(float)

    if normalize:
        cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True).clip(min=1e-12)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_plot)
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            txt = f"{cm_plot[i, j]:.2f}" if normalize else f"{int(cm_plot[i, j])}"
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.show()

# -----------------------
# Tahminleri topla
# -----------------------
@torch.no_grad()
def collect_preds(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    return y_true, y_pred, y_prob

def main():
    print("Device:", DEVICE)

    # -----------------------
    # CSV oku
    # -----------------------
    val_df  = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("Val/Test:", len(val_df), len(test_df))

    # -----------------------
    # Loader’lar
    # -----------------------
    val_ds  = KneeDataset(val_df, transform=eval_tf)
    test_ds = KneeDataset(test_df, transform=eval_tf)

    val_loader  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -----------------------
    # Model kur + checkpoint yükle
    # -----------------------
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {CKPT_PATH}")

    model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    print("✅ Loaded checkpoint:", CKPT_PATH)

    # -----------------------
    # VAL değerlendirme
    # -----------------------
    y_true, y_pred, y_prob = collect_preds(model, val_loader)

    cm = confusion_matrix(y_true, y_pred)
    print("\n==== VAL Confusion Matrix (raw) ====\n", cm)

    print("\n==== VAL Classification Report ====\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # ordinal hata metrikleri
    dist = np.abs(y_true - y_pred)
    print("\nVAL Mean |error|:", round(dist.mean(), 4))
    print("VAL >=2 away errors ratio:", round(float((dist >= 2).mean()), 4))

    # En çok karışan çiftler
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    pairs = []
    for i in range(cm_off.shape[0]):
        for j in range(cm_off.shape[1]):
            if cm_off[i, j] > 0:
                pairs.append((cm_off[i, j], i, j))
    pairs.sort(reverse=True)

    print("\nTop confusions (VAL):")
    for count, i, j in pairs[:10]:
        print(f"{CLASS_NAMES[i]} -> {CLASS_NAMES[j]} : {int(count)}")

    # Çok emin ama yanlış örnekler (index bazlı)
    conf = y_prob.max(axis=1)
    wrong_idx = np.where(y_true != y_pred)[0]
    wrong_sorted = wrong_idx[np.argsort(-conf[wrong_idx])]

    print("\nTop confident wrongs (VAL) [idx, true, pred, conf]:")
    for k in wrong_sorted[:20]:
        print(k, CLASS_NAMES[y_true[k]], CLASS_NAMES[y_pred[k]], round(float(conf[k]), 4))

    # Plot
    plot_cm(cm, CLASS_NAMES, normalize=False, title="VAL Confusion Matrix")
    plot_cm(cm, CLASS_NAMES, normalize=True, title="VAL Confusion Matrix")

    # -----------------------
    # TEST değerlendirme (istersen)
    # -----------------------
    y_true_t, y_pred_t, _ = collect_preds(model, test_loader)
    cm_t = confusion_matrix(y_true_t, y_pred_t)

    print("\n==== TEST Confusion Matrix (raw) ====\n", cm_t)
    print("\n==== TEST Classification Report ====\n")
    print(classification_report(y_true_t, y_pred_t, target_names=CLASS_NAMES, digits=4))

    dist_t = np.abs(y_true_t - y_pred_t)
    print("\nTEST Mean |error|:", round(dist_t.mean(), 4))
    print("TEST >=2 away errors ratio:", round(float((dist_t >= 2).mean()), 4))

    plot_cm(cm_t, CLASS_NAMES, normalize=False, title="TEST Confusion Matrix")
    plot_cm(cm_t, CLASS_NAMES, normalize=True, title="TEST Confusion Matrix")

if __name__ == "__main__":
    main()