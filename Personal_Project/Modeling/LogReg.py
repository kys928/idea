#!/usr/bin/env python3
# ============================================================
# LOGISTIC REGRESSION — Train/Evaluate on saved TF-IDF (no argparse)
# Multinomial + SAGA; class_weight='balanced'; saves model + metrics.
# ============================================================

import json, logging
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# ---- Paths (must match encoder script) ----
ARTIFACTS_DIR = r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\Artifacts"

# ---- Training config ----
MULTI_CLASS = "multinomial"
SOLVER      = "saga"        # efficient on huge sparse TF-IDF; supports multinomial + L1/L2
PENALTY     = "l2"          # switch to "l1" for feature selection (may need larger C)
C_VALUE     = 1.0
TOL         = 1e-4
MAX_ITER    = 1000
CLASS_WEIGHT = "balanced"
N_JOBS      = -1
RANDOM_STATE = 42

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
L = logging.getLogger("train_logreg_ed")

def load_features(art_dir: Path):
    Xtr = load_npz(art_dir / "X_train.npz")
    Xva = load_npz(art_dir / "X_val.npz")
    Xte = load_npz(art_dir / "X_test.npz")
    ytr = np.load(art_dir / "y_train.npy")
    yva = np.load(art_dir / "y_val.npy")
    yte = np.load(art_dir / "y_test.npy")
    le  = joblib.load(art_dir / "label_encoder.joblib")
    classes = list(le.classes_)
    return Xtr, Xva, Xte, ytr, yva, yte, classes

def make_model():
    return LogisticRegression(
        multi_class=MULTI_CLASS, solver=SOLVER, penalty=PENALTY,
        C=C_VALUE, tol=TOL, max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT, n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=0
    )

def evaluate(model, X, y, labels, split_name: str):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred)
    L.info(f"[{split_name}] Accuracy={acc:.4f} | MacroF1={f1m:.4f} | CM={cm.shape}")
    L.info("\n" + classification_report(y, pred, target_names=labels, digits=3))
    return {"accuracy": float(acc), "macro_f1": float(f1m), "cm": cm}

def main():
    art_dir = Path(ARTIFACTS_DIR)
    art_dir.mkdir(parents=True, exist_ok=True)

    Xtr, Xva, Xte, ytr, yva, yte, classes = load_features(art_dir)
    L.info("Loaded shapes — train=%s | val=%s | test=%s | dim=%d",
           Xtr.shape, Xva.shape, Xte.shape, Xtr.shape[1])

    base = make_model()

    model = base
    model.fit(Xtr, ytr)
    best_params = {"C": C_VALUE, "penalty": PENALTY}


    # Validation
    val_stats = evaluate(model, Xva, yva, classes, "valid")
    # Test
    test_stats = evaluate(model, Xte, yte, classes, "test")

    # Save artifacts
    joblib.dump(model, art_dir / "logreg_tfidf_model.joblib")
    np.save(art_dir / "confusion_test.npy", test_stats["cm"])
    metrics = {
        "best_params": best_params,
        "validation": {k:v for k,v in val_stats.items() if k != "cm"},
        "test": {"accuracy": test_stats["accuracy"], "macro_f1": test_stats["macro_f1"]},
        "notes": {
            "why_saga": "Handles large sparse TF-IDF; true multinomial softmax; supports L1/L2.",
            "penalty_effect": "L2 is stable; L1 can zero-out features (sparser) but may need larger C."
        }
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    L.info("Saved model and metrics to: %s", str(art_dir.resolve()))

if __name__ == "__main__":
    main()
