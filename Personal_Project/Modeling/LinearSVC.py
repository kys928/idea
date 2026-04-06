#!/usr/bin/env python3
# LinearSVC on frozen TF-IDF (char2-3 + word3-4) with probability calibration.

from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, classification_report

# --- Paths (edit if needed) ---
ARTIFACTS_DIR = Path(r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\Artifacts")
RUNS_DIR = Path("runs/LinearSVC"); RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load frozen features/labels ---
X_train = load_npz(ARTIFACTS_DIR / "X_train.npz")
X_val   = load_npz(ARTIFACTS_DIR / "X_val.npz")
X_test  = load_npz(ARTIFACTS_DIR / "X_test.npz")
y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
y_val   = np.load(ARTIFACTS_DIR / "y_val.npy")
y_test  = np.load(ARTIFACTS_DIR / "y_test.npy")

# --- Optional: class names for nice reports ---
try:
    le = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    CLASS_NAMES = [str(c) for c in le.classes_]
except Exception:
    CLASS_NAMES = None

def eval_model(model, X, y, split="val"):
    y_hat = model.predict(X)
    macro = f1_score(y, y_hat, average="macro")
    acc   = accuracy_score(y, y_hat)
    print(f"[{split}] acc={acc:.4f} macroF1={macro:.4f}")
    print(classification_report(y, y_hat, target_names=CLASS_NAMES, digits=3) if CLASS_NAMES
          else classification_report(y, y_hat, digits=3))
    return macro, acc

# --- Train base LinearSVC (dual=True tends to be better when n_samples < n_features) ---
svc = LinearSVC(C=1.0, class_weight="balanced", dual=True, max_iter=20000)
svc.fit(X_train, y_train)


# --- Evaluate ---
eval_model(X_val,  y_val,  split="val")
eval_model(X_test, y_test, split="test")

# --- Save calibrated model ---
out_path = RUNS_DIR / "LinearSVC_best.joblib"
joblib.dump(out_path)
print(f"Saved: {out_path.resolve()}")
