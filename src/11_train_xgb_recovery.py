from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_DIR / "recovery_synth.csv")

    feature_cols = ["age", "sex", "bmi", "diabetes", "smoker", "predicted_kl",
                    "p_kl0", "p_kl1", "p_kl2", "p_kl3", "p_kl4"]
    target_col = "recovery_days"

    X = df[feature_cols].copy()
    y = df[target_col].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    print("=== Recovery XGBoost (Synthetic Labels) ===")
    print(f"MAE:  {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2:   {r2:.3f}")

    out_path = RUNS_DIR / "xgb_recovery_synth.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()