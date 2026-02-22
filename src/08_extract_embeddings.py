import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_backbone():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(*list(m.children())[:-1])  # (B,512,1,1)
    return backbone.to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@torch.no_grad()
def extract(csv_path: Path, out_prefix: Path, project_root: Path):
    df = pd.read_csv(csv_path)

    X_list, y_list = [], []
    backbone = build_backbone()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_path.name):
        rel_path = row["path"]
        y = int(row["label"])

        img_path = project_root / rel_path  # artık relative olduğu için bu çalışacak
        img = Image.open(img_path).convert("RGB")

        x = tf(img).unsqueeze(0).to(DEVICE)
        feat = backbone(x).flatten(1).cpu().numpy()[0]  # (512,)

        X_list.append(feat)
        y_list.append(y)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    np.save(str(out_prefix) + "_X.npy", X)
    np.save(str(out_prefix) + "_y.npy", y)
    print("Saved:", str(out_prefix) + "_X.npy", str(out_prefix) + "_y.npy", X.shape, y.shape)

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    runs_dir = project_root / "runs"
    runs_dir.mkdir(exist_ok=True)

    extract(data_dir / "train.csv", runs_dir / "emb_train", project_root)
    extract(data_dir / "val.csv",   runs_dir / "emb_val", project_root)
    extract(data_dir / "test.csv",  runs_dir / "emb_test", project_root)

if __name__ == "__main__":
    main()