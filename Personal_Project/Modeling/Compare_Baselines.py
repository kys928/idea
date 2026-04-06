#!/usr/bin/env python3
# compare_baselines.py — evaluate saved models directly on frozen TF-IDF features (no metrics JSONs needed)

from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.sparse import load_npz
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")  # safe on headless
import matplotlib.pyplot as plt

# ---------- Hard-coded artifact paths ----------
ART = Path(r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\Artifacts")

# Use your exact LogReg path (first choice), plus a couple of fallbacks just in case
CANDIDATES = {
    "Logistic Regression": [
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\Artifacts\logreg_tfidf_model.joblib"),
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\logreg_tfidf_model.joblib"),
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\LogReg\LogReg_best.joblib"),
    ],
    "LinearSVC": [
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\LinearSVC\LinearSVC_best.joblib"),
    ],
    "KNN": [
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\KNN\KNN_best.joblib"),
    ],
    "Naive Bayes": [
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\NaiveBayes\NaiveBayes_best.joblib"),
        Path(r"C:\Users\adm\Desktop\IndividualChallenge\Modeling\runs\NaiveBayes\ComplementNB_best.joblib"),
    ],
}

OUTDIR = Path("runs/_comparison"); OUTDIR.mkdir(parents=True, exist_ok=True)

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

# ---------- Load frozen features/labels ----------
X_val  = load_npz(ART / "X_val.npz");   y_val  = np.load(ART / "y_val.npy")
X_test = load_npz(ART / "X_test.npz");  y_test = np.load(ART / "y_test.npy")

rows = []

for name, paths in CANDIDATES.items():
    model_path = first_existing(paths)
    if not model_path:
        print(f"[warn] No saved model found for {name}. Skipping.")
        continue

    clf = joblib.load(model_path)

    def evaluate(split_name, X, y):
        yhat = clf.predict(X)
        return dict(
            model=name,
            variant=model_path.name.replace(".joblib",""),
            source=str(model_path),
            **{
                f"macro_f1_{split_name}": f1_score(y, yhat, average="macro"),
                f"accuracy_{split_name}": accuracy_score(y, yhat),
            }
        )

    r_val  = evaluate("val",  X_val,  y_val)
    r_test = evaluate("test", X_test, y_test)

    rows.append({
        "model": name,
        "variant": r_val["variant"],
        "source": r_val["source"],
        "macro_f1_val":  r_val["macro_f1_val"],
        "accuracy_val":  r_val["accuracy_val"],
        "macro_f1_test": r_test["macro_f1_test"],
        "accuracy_test": r_test["accuracy_test"],
    })

df = pd.DataFrame(rows)
if df.empty:
    print("[error] No models evaluated. Check paths in CANDIDATES.")
    raise SystemExit(1)

df_sorted = df.sort_values("macro_f1_val", ascending=False)
df_sorted.to_csv(OUTDIR / "baseline_comparison.csv", index=False)

print("\n=== Baseline Leaderboard (by validation macro-F1) ===")
cols = ["model", "variant", "macro_f1_val", "accuracy_val", "macro_f1_test", "accuracy_test"]
pd.options.display.float_format = "{:.4f}".format
print(df_sorted[cols].to_string(index=False))

# ---------- Plots ----------
def plot_val(df_sorted, path):
    fig = plt.figure(figsize=(7, 4))
    names = (df_sorted["model"] + " (" + df_sorted["variant"] + ")").tolist()
    vals  = df_sorted["macro_f1_val"].tolist()
    plt.bar(names, vals)
    plt.title("Validation Macro-F1 — Baseline Models")
    plt.ylabel("Macro-F1"); plt.ylim(0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plot_val_test(df_sorted, path):
    fig = plt.figure(figsize=(8, 4.5))
    names = (df_sorted["model"] + " (" + df_sorted["variant"] + ")").tolist()
    idx = np.arange(len(names)); w = 0.4
    v = df_sorted["macro_f1_val"].tolist()
    t = df_sorted["macro_f1_test"].tolist()
    plt.bar(idx - w/2, v, width=w, label="val")
    plt.bar(idx + w/2, t, width=w, label="test")
    plt.title("Macro-F1: Validation vs Test — Baseline Models")
    plt.ylabel("Macro-F1"); plt.ylim(0, 1.0); plt.legend()
    plt.xticks(idx, names, rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

plot_val(df_sorted, OUTDIR / "baseline_macroF1_val.png")
plot_val_test(df_sorted, OUTDIR / "baseline_macroF1_val_test.png")

print(f"\n[done] Wrote: {OUTDIR/'baseline_comparison.csv'}")
print(f"[done] Plots: {OUTDIR/'baseline_macroF1_val.png'}, {OUTDIR/'baseline_macroF1_val_test.png'}")
