from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_recovery_days(row):
    """
    DEMO amaçlı sentetik kural:
    - KL arttıkça iyileşme süresi artsın
    - Yaş/BMI/komorbidite etkisi olsun
    - Biraz gürültü eklensin
    """
    kl = row["predicted_kl"]
    age = row["age"]
    bmi = row["bmi"]
    diabetes = row["diabetes"]
    smoker = row["smoker"]
    sex = row["sex"]  # 0=female, 1=male (örnek)

    base = 25
    days = (
        base
        + 8.0 * kl
        + 0.22 * age
        + 0.35 * max(bmi - 25, 0)
        + 6.0 * diabetes
        + 3.5 * smoker
        + 1.0 * sex
    )

    # KL olasılıklarından “belirsizlik” etkisi (opsiyonel ama hoş)
    # Belirsizlik yüksekse süre biraz artsın (daha temkinli)
    probs = np.array([row[f"p_kl{i}"] for i in range(5)])
    entropy = -np.sum(np.clip(probs, 1e-9, 1.0) * np.log(np.clip(probs, 1e-9, 1.0)))
    days += 2.0 * entropy  # 0..~3 arası küçük etki

    noise = rng.normal(0, 5.0)
    days = days + noise

    # mantıklı aralığa kırp
    days = float(np.clip(days, 10, 180))
    return days

def generate(n=1200, out_path=DATA_DIR / "recovery_synth.csv"):
    # Klinik feature’ları “web formu gibi” üretelim
    age = rng.integers(40, 86, size=n)
    sex = rng.integers(0, 2, size=n)  # 0/1
    bmi = np.clip(rng.normal(28, 4.5, size=n), 18, 45)

    diabetes = rng.binomial(1, p=0.22, size=n)
    smoker = rng.binomial(1, p=0.25, size=n)

    # predicted_KL üretimi: yaş/BMI/komorbidite ile ilişkili olsun (tam rastgele değil)
    z = (
        -2.2
        + 0.03 * (age - 55)
        + 0.08 * (bmi - 27)
        + 0.35 * diabetes
        + 0.20 * smoker
    )
    # KL0..KL4 olasılıklarını kaba şekilde üret
    p4 = sigmoid(z - 1.0)
    p3 = sigmoid(z - 0.2) - p4
    p2 = sigmoid(z + 0.6) - (p3 + p4)
    p1 = sigmoid(z + 1.3) - (p2 + p3 + p4)
    p0 = 1 - (p1 + p2 + p3 + p4)

    probs = np.vstack([p0, p1, p2, p3, p4]).T
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    predicted_kl = np.array([rng.choice(5, p=probs[i]) for i in range(n)])

    df = pd.DataFrame({
        "age": age.astype(int),
        "sex": sex.astype(int),
        "bmi": bmi.astype(float),
        "diabetes": diabetes.astype(int),
        "smoker": smoker.astype(int),
        "predicted_kl": predicted_kl.astype(int),
    })

    for i in range(5):
        df[f"p_kl{i}"] = probs[:, i].astype(float)

    df["recovery_days"] = df.apply(simulate_recovery_days, axis=1)

    df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(df.head(3))

if __name__ == "__main__":
    generate(n=1200)