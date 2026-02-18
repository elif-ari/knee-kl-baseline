import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# CSV oku
train_df = pd.read_csv(BASE_DIR / "data" / "train.csv")

print("Train örnek sayısı:", len(train_df))

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

class KneeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = row["label"]
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 1 kanal → 3 kanal yap
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# Dataset oluştur
train_dataset = KneeDataset(train_df, transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Test batch al
images, labels = next(iter(train_loader))

print("Batch görüntü shape:", images.shape)
print("Batch label shape:", labels.shape)
print("Label örnek:", labels[:8])
