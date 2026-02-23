from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "runs"

ARTIFACT = RUNS_DIR / "xgb_recovery_synth.joblib"

def predict_recovery_days(
    age: int,
    sex: int,
    bmi: float,
    diabetes: int,
    smoker: int,
    predicted_kl: int,
    kl_probs: list[float]  # [p0,p1,p2,p3,p4]
) -> float:
    pack = joblib.load(ARTIFACT)
    model = pack["model"]
    feature_cols = pack["feature_cols"]

    row = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "diabetes": diabetes,
        "smoker": smoker,
        "predicted_kl": predicted_kl,
        "p_kl0": float(kl_probs[0]),
        "p_kl1": float(kl_probs[1]),
        "p_kl2": float(kl_probs[2]),
        "p_kl3": float(kl_probs[3]),
        "p_kl4": float(kl_probs[4]),
    }
    X = pd.DataFrame([row])[feature_cols]
    pred = float(model.predict(X)[0])
    return pred

if __name__ == "__main__":
    # Demo örnek
    days = predict_recovery_days(
        age=67, sex=1, bmi=31.2, diabetes=1, smoker=0,
        predicted_kl=3,
        kl_probs=[0.02, 0.08, 0.19, 0.60, 0.11]
    )
    print("Predicted recovery_days:", round(days, 1))