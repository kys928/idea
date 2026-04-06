#!/usr/bin/env python3
# Naive Bayes sweep (ComplementNB vs MultinomialNB) on frozen TF-IDF.

from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import joblib
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, classification_report

# --- Paths (edit if needed) ---
ARTIFACTS_DIR = Path(r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\Artifacts")
RUNS_DIR = Path("runs/NaiveBayes"); RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --- Data ---
X_train = load_npz(ARTIFACTS_DIR / "X_train.npz")
X_val   = load_npz(ARTIFACTS_DIR / "X_val.npz")
X_test  = load_npz(ARTIFACTS_DIR / "X_test.npz")
y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
y_val   = np.load(ARTIFACTS_DIR / "y_val.npy")
y_test  = np.load(ARTIFACTS_DIR / "y_test.npy")

# --- Label names (optional) ---
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

# --- Try both NB flavours and keep the best on validation ---
candidates = [
    ("ComplementNB",  ComplementNB(alpha=0.5, norm=False)),
    ("MultinomialNB", MultinomialNB(alpha=0.5)),
]

best_name, best_model, best_score = None, None, -1.0
for name, model in candidates:
    model.fit(X_train, y_train)
    macro, _ = eval_model(model, X_val, y_val, split=f"val/{name}")
    if macro > best_score:
        best_name, best_model, best_score = name, model, macro

# --- Final test ---
eval_model(best_model, X_test, y_test, split=f"test/{best_name}")

# --- Save best ---
joblib.dump(best_model, RUNS_DIR / f"{best_name}_best.joblib")
joblib.dump(best_model, RUNS_DIR / "NaiveBayes_best.joblib")
print(f"Saved best NB: {(RUNS_DIR / f'{best_name}_best.joblib').resolve()}")
