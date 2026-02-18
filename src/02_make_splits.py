from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# PROJE YOLU (sen değiştirme, otomatik alıyoruz)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "labeled"

print("Data directory:", DATA_DIR)

# Tüm görüntüleri topla
rows = []

for label in range(5):
    folder = DATA_DIR / str(label)
    images = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    
    for img_path in images:
        rows.append({
            "path": str(img_path),
            "label": label
        })

df = pd.DataFrame(rows)

print("Toplam görüntü:", len(df))
print(df["label"].value_counts())

# Stratified split (80% train, 10% val, 10% test)

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print("\nTrain:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# CSV olarak kaydet
train_df.to_csv(BASE_DIR / "data" / "train.csv", index=False)
val_df.to_csv(BASE_DIR / "data" / "val.csv", index=False)
test_df.to_csv(BASE_DIR / "data" / "test.csv", index=False)

print("\nCSV dosyaları oluşturuldu.")
