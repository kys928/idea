
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from datasets import load_from_disk

# ---------------- USER CONFIG (edit if paths differ) ----------------
ED_BASE = Path(r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\EmotionDataset")
CHARTS_DIR = Path(r"C:\Users\adm\Desktop\IndividualChallenge\Reports\Charts")

FORMAT = "png"   # or "pdf"
DPI = 150
TOP_LABELS = 20
ROLLING = 50     # rolling window for length trend
SHOW_FIGS = False  # True to display windows locally; False = headless save

# If you want to force a text column name (e.g., "utterance"), set it here. Otherwise leave None.
FORCE_TEXT_COLUMN: Optional[str] = None

# ---------------- Helpers ----------------
def choose_text_col(df: pd.DataFrame) -> str:
    """Strict preference for message text columns; blacklist common non-text fields."""
    if FORCE_TEXT_COLUMN and FORCE_TEXT_COLUMN in df.columns:
        return FORCE_TEXT_COLUMN

    # Strong preference order (common ED names first)
    for k in ["utterance", "text", "response", "sentence", "content"]:
        if k in df.columns:
            return k

    # Fallback: any object-like column that is not an obvious meta/label field
    blacklist = {
        "label", "labels", "primary_label", "emotion",
        "speaker", "id", "conv_id", "conversation_id", "dialogue_id",
        "utterance_idx", "turn_id", "index", "source_file", "split"
    }
    for c in df.columns:
        if df[c].dtype == object and c not in blacklist:
            return c
    # Last resort: first column
    return df.columns[0]

def choose_label_col(df: pd.DataFrame) -> Optional[str]:
    for k in ["primary_label", "label", "labels", "emotion"]:
        if k in df.columns:
            return k
    return None

def pdf_from_ds(ds) -> Dict[str, pd.DataFrame]:
    out = {}
    for name in ds.keys():
        if name in {"train", "validation", "valid", "dev", "test"}:
            out[name] = ds[name].to_pandas()
    return out

def save_and_optionally_show(fig, name: str):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fp = CHARTS_DIR / f"{name}.{FORMAT}"
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    print(f"[save] {fp}")
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)

# ---------------- Load dataset ----------------
def load_ed_dataset(ed_base: Path):
    for sub in [Path("CleanNoLeak") / "hf_disk", Path("Clean") / "hf_disk", Path("hf_disk")]:
        candidate = ed_base / sub
        if candidate.exists():
            print(f"[load] Using: {candidate}")
            return load_from_disk(str(candidate))
    raise SystemExit(f"No ED dataset found under {ed_base} "
                     f"(expected CleanNoLeak/hf_disk, Clean/hf_disk, or hf_disk).")

# ---------------- Build pooled rows ----------------
def build_rows(pdf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    chosen_text = {}
    chosen_label = {}

    for sp, df in pdf.items():
        if df is None or df.empty:
            continue
        tcol = choose_text_col(df)
        lcol = choose_label_col(df)

        chosen_text[sp] = tcol
        chosen_label[sp] = lcol

        # Word count (simple whitespace split is fine for quick EDA)
        lengths = df[tcol].astype(str).map(lambda s: len(str(s).split()))
        labels = (df[lcol] if lcol and lcol in df.columns else pd.Series([None] * len(df)))

        # If labels are lists, flatten per row by joining or taking primary; for EDA counts, str() is ok
        labels = labels.apply(lambda x: x[0] if isinstance(x, (list, tuple)) and x else x)

        for i, w in enumerate(lengths.tolist()):
            lab = labels.iloc[i] if i < len(labels) else None
            rows.append({"__length__": w, "__split__": sp, "__label__": lab})

    rows = pd.DataFrame.from_records(rows)
    # Diagnostics
    print("\n[diagnostics] Chosen columns per split:")
    for sp in sorted(set(chosen_text.keys())):
        print(f"  - {sp:12s} text={chosen_text[sp]!r}  label={chosen_label.get(sp)!r}")
    print(f"[diagnostics] Rows pooled: {len(rows)}")
    if not rows.empty:
        desc = rows["__length__"].describe()
        print("[diagnostics] Length describe():")
        print(desc.to_string())
        if (rows["__length__"].nunique() == 1) and (rows["__length__"].iloc[0] == 1):
            print("[warning] All utterance lengths are 1 word. "
                  "This usually means the wrong column was selected as text (e.g., a label). "
                  "Consider setting FORCE_TEXT_COLUMN = 'utterance' (or 'text') and re-run.")
    return rows

# ---------------- Plots ----------------
def plot_hist(rows: pd.DataFrame):
    if rows.empty:
        print("[viz] histogram: no data"); return
    fig = plt.figure(figsize=(7, 4))
    vals = rows["__length__"].astype(int).tolist()
    bins = min(60, max(10, int(len(vals) ** 0.5)))
    plt.hist(vals, bins=bins)
    plt.xlabel("Words per utterance"); plt.ylabel("Count"); plt.title("ED — Utterance length distribution")
    plt.tight_layout()
    save_and_optionally_show(fig, "ED_Utterance_length_distribution")

def plot_box(rows: pd.DataFrame):
    if rows.empty:
        print("[viz] boxplot: no data"); return
    fig = plt.figure(figsize=(5, 5))
    plt.boxplot(rows["__length__"].astype(int).tolist(), vert=True, showfliers=True)
    plt.ylabel("Words"); plt.title("ED — Utterance length spread")
    plt.tight_layout()
    save_and_optionally_show(fig, "ED_Utterance_length_spread")

def plot_labels_bar(rows: pd.DataFrame):
    exploded = rows.copy()
    exploded["__label__"] = exploded["__label__"].astype(str).str.strip().str.lower()
    counts = exploded["__label__"].value_counts().sort_values(ascending=False)
    topn = min(TOP_LABELS, len(counts))
    if topn <= 0:
        print("[viz] labels bar: no labels"); return
    fig = plt.figure(figsize=(10, 5))
    idx = np.arange(topn)
    plt.bar(idx, counts.values[:topn])
    plt.xticks(idx, counts.index[:topn].tolist(), rotation=60, ha="right")
    plt.ylabel("Count"); plt.title(f"ED — Label frequency (top {topn})")
    plt.tight_layout()
    save_and_optionally_show(fig, "ED_Label_frequency_topN")

def plot_scatter(rows: pd.DataFrame):
    if rows.empty:
        print("[viz] scatter: no data"); return
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(np.arange(len(rows)), rows["__length__"].astype(int).tolist(), s=8, alpha=0.6)
    plt.xlabel("Index (proxy for turn)"); plt.ylabel("Words"); plt.title("ED — Turn index vs utterance length")
    plt.tight_layout()
    save_and_optionally_show(fig, "ED_Turn_index_vs_length")

def plot_trend(rows: pd.DataFrame):
    if rows.empty:
        print("[viz] line: no data"); return
    y = pd.Series(rows["__length__"].astype(int).tolist())
    if ROLLING and ROLLING > 1:
        y = y.rolling(window=ROLLING, min_periods=1).mean()
    fig = plt.figure(figsize=(7, 4))
    plt.plot(y.values)
    plt.xlabel("Index"); plt.ylabel("Words"); plt.title("ED — Utterance length trend")
    plt.tight_layout()
    save_and_optionally_show(fig, "ED_Utterance_length_trend")

# ---------------- Driver ----------------
def main():
    ds = load_ed_dataset(ED_BASE)
    pdf = pdf_from_ds(ds)
    # quick split sizes
    print("[splits] sizes:", {k: (0 if v is None else len(v)) for k, v in pdf.items()})
    rows = build_rows(pdf)
    # plots
    plot_hist(rows)
    plot_box(rows)
    plot_labels_bar(rows)
    plot_scatter(rows)
    plot_trend(rows)
    print("[done] ED visualization finished. Charts saved to:", CHARTS_DIR)

if __name__ == "__main__":
    main()
