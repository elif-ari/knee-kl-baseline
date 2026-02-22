import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

def main():
    base = Path(__file__).resolve().parent.parent
    runs = base / "runs"

    # 1) Load embeddings
    X_train = np.load(runs / "emb_train_X.npy")
    y_train = np.load(runs / "emb_train_y.npy")

    X_val = np.load(runs / "emb_val_X.npy")
    y_val = np.load(runs / "emb_val_y.npy")

    X_test = np.load(runs / "emb_test_X.npy")
    y_test = np.load(runs / "emb_test_y.npy")

    num_classes = int(len(np.unique(y_train)))
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Val shapes:", X_val.shape, y_val.shape)
    print("Test shapes:", X_test.shape, y_test.shape)
    print("num_classes:", num_classes)

    # 2) Class imbalance weight
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # 3) Model
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss"
    )

    # 4) Train
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # 5) Eval (test)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n=== Classification Report (Test) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n=== Confusion Matrix (Test) ===")
    print(confusion_matrix(y_test, y_pred))

    # 6) Save
    out_path = runs / "xgb_from_cnn_embeddings.joblib"
    joblib.dump(model, out_path)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
    