#!/usr/bin/env python3
# ============================================================
# TF-IDF ENCODING — EmpatheticDialogues (no argparse)
# Loads HF "hf_disk" (train/validation/test), normalizes text,
# builds TF-IDF (char 1–2 + word 3–4), encodes labels, saves X/y.
# Hypothesis: char(1–2)+word(3–4) improves Macro-F1 vs word-only
# by catching typos/subword cues in short utterances (CER benefit)
# while word 3–4 captures phrase structure (WER benefit).
# ============================================================

import json, logging, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.sparse import hstack, csr_matrix, save_npz
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import load_from_disk, Dataset

# ---------- Paths (edit these if you move things) ----------
DATASET_ROOT = r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\EmotionDataset\hf_disk"
ARTIFACTS_DIR = r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\Artifacts"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
L = logging.getLogger("tfidf_encode_ed")

# ---------- Config ----------
@dataclass
class ColumnsCfg:
    text_col: Optional[str] = None
    label_col: Optional[str] = None
    text_candidates: Tuple[str, ...] = ("text","utterance","response","sentence","content")
    label_candidates: Tuple[str, ...] = ("label","labels","primary_label","emotion")

@dataclass
class TfidfCfg:
    use_char: bool = True
    char_ngram: Tuple[int,int] = (1,2)        # CER-sensitive
    char_max_features: Optional[int] = 200_000
    use_word: bool = True
    word_ngram: Tuple[int,int] = (3,4)        # WER-sensitive
    word_max_features: Optional[int] = 200_000
    remove_stopwords: bool = False
    lowercase: bool = True
    min_df: int = 2
    max_df: float = 0.98

SEED = 42
LOG_EXAMPLES = 3

# ---------- Reproducibility ----------
random.seed(SEED)
np.random.seed(SEED)

# ---------- Helpers ----------
def normalize_text(s: str, lowercase: bool = True) -> str:
    if s is None:
        return ""
    s = s.strip()
    return s.lower() if lowercase else s

def preprocess_texts(texts: List[str], lowercase: bool) -> List[str]:
    return [normalize_text(t, lowercase=lowercase) for t in texts]

def detect_columns(ds: Dataset, cfg: ColumnsCfg) -> Tuple[str, str]:
    cols = ds.column_names
    text_col = cfg.text_col or next((c for c in cfg.text_candidates if c in cols), None)
    label_col = cfg.label_col or next((c for c in cfg.label_candidates if c in cols), None)
    if not text_col or not label_col:
        raise KeyError(f"Could not detect text/label columns. Found: {cols}")
    return text_col, label_col

def extract_texts_labels(ds: Dataset, text_col: str, label_col: str) -> Tuple[List[str], List[str]]:
    texts = ds[text_col]
    labels = ds[label_col]
    # If ClassLabel -> map ids to names
    if hasattr(ds.features.get(label_col, None), "names"):
        names = ds.features[label_col].names
        labels = [names[i] if isinstance(i, int) else i for i in labels]
    return texts, labels

def build_vectorizers(cfg: TfidfCfg):
    char_vec = None
    word_vec = None
    if cfg.use_char:
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=cfg.char_ngram,
            lowercase=False,
            min_df=cfg.min_df, max_df=cfg.max_df,
            max_features=cfg.char_max_features,
            sublinear_tf=True, dtype=np.float32
        )
    if cfg.use_word:
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=cfg.word_ngram,
            lowercase=False,
            min_df=cfg.min_df, max_df=cfg.max_df,
            max_features=cfg.word_max_features,
            stop_words=("english" if cfg.remove_stopwords else None),
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True, dtype=np.float32
        )
    return char_vec, word_vec

def fit_transform_features(
    tr_texts: List[str],
    va_texts: List[str],
    te_texts: List[str],
    cfg: TfidfCfg,
    out_dir: Path
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, Dict[str,int]]:
    char_vec, word_vec = build_vectorizers(cfg)

    parts_tr, parts_va, parts_te = [], [], []
    dims = {}

    if char_vec:
        L.info("Fitting char TF-IDF (ngram=%s, max_features=%s)", cfg.char_ngram, cfg.char_max_features)
        Xtr_c = char_vec.fit_transform(tr_texts)
        Xva_c = char_vec.transform(va_texts)
        Xte_c = char_vec.transform(te_texts)
        parts_tr.append(Xtr_c); parts_va.append(Xva_c); parts_te.append(Xte_c)
        dims["char_dim"] = Xtr_c.shape[1]
        joblib.dump(char_vec, out_dir / "tfidf_char_vectorizer.joblib")

    if word_vec:
        L.info("Fitting word TF-IDF (ngram=%s, max_features=%s, stopwords=%s)", cfg.word_ngram, cfg.word_max_features, cfg.remove_stopwords)
        Xtr_w = word_vec.fit_transform(tr_texts)
        Xva_w = word_vec.transform(va_texts)
        Xte_w = word_vec.transform(te_texts)
        parts_tr.append(Xtr_w); parts_va.append(Xva_w); parts_te.append(Xte_w)
        dims["word_dim"] = Xtr_w.shape[1]
        joblib.dump(word_vec, out_dir / "tfidf_word_vectorizer.joblib")

    X_train = parts_tr[0] if len(parts_tr)==1 else hstack(parts_tr).tocsr()
    X_val   = parts_va[0] if len(parts_va)==1 else hstack(parts_va).tocsr()
    X_test  = parts_te[0] if len(parts_te)==1 else hstack(parts_te).tocsr()
    dims["total_dim"] = X_train.shape[1]
    return X_train, X_val, X_test, dims

def main():
    print("HYPOTHESIS: char(1–2)+word(3–4) TF-IDF > word-only on Macro-F1 (ED).")

    dataset_root = Path(DATASET_ROOT)
    out_dir = Path(ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HF DatasetDict from disk (expects subdirs train/validation/test)
    L.info(f"Loading DatasetDict from disk: {dataset_root}")
    dset = load_from_disk(str(dataset_root))
    # Normalize split keys
    splits = {}
    for k in dset.keys():
        nk = "validation" if k.lower() in ("val","valid","validation") else k.lower()
        splits[nk] = dset[k]
    assert {"train","validation","test"}.issubset(splits.keys()), f"Found splits: {list(splits.keys())}"

    cols_cfg = ColumnsCfg()
    text_col, label_col = detect_columns(splits["train"], cols_cfg)
    L.info(f"Using text column='{text_col}', label column='{label_col}'")

    def get_xy(name: str):
        ds = splits[name]
        texts, labels = extract_texts_labels(ds, text_col, label_col)
        texts = preprocess_texts(texts, lowercase=True)
        return texts, labels

    tr_texts, tr_labels = get_xy("train")
    va_texts, va_labels = get_xy("validation")
    te_texts, te_labels = get_xy("test")

    uniq = sorted(set(tr_labels) | set(va_labels) | set(te_labels))
    L.info("Dataset sizes — train=%d | val=%d | test=%d", len(tr_texts), len(va_texts), len(te_texts))
    L.info("Num classes: %d", len(uniq))
    for i in range(min(LOG_EXAMPLES, len(tr_texts))):
        L.info("Sample[%d]: %r | label=%s", i, tr_texts[i][:120], tr_labels[i])

    tfidf_cfg = TfidfCfg()
    Xtr, Xva, Xte, dims = fit_transform_features(tr_texts, va_texts, te_texts, tfidf_cfg, out_dir)

    # Labels: fit on train, transform val/test
    le = LabelEncoder()
    ytr = le.fit_transform(tr_labels)
    yva = le.transform(va_labels)
    yte = le.transform(te_labels)
    joblib.dump(le, out_dir / "label_encoder.joblib")

    # Persist matrices + y
    save_npz(out_dir / "X_train.npz", Xtr)
    save_npz(out_dir / "X_val.npz",   Xva)
    save_npz(out_dir / "X_test.npz",  Xte)
    np.save(out_dir / "y_train.npy", ytr)
    np.save(out_dir / "y_val.npy",   yva)
    np.save(out_dir / "y_test.npy",  yte)

    meta = {
        "paths": {"dataset_root": str(dataset_root), "artifacts_dir": str(out_dir)},
        "columns": {"text": text_col, "label": label_col},
        "tfidf_cfg": asdict(tfidf_cfg),
        "num_classes": int(len(le.classes_)),
        "class_names": list(le.classes_),
        "dims": dims,
        "seed": SEED,
        "notes": {
            "char_ngrams": "Captures substrings; robust to typos (CER).",
            "word_ngrams": "Captures short phrases (WER)."
        }
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    L.info("Saved artifacts to: %s", str(out_dir.resolve()))

if __name__ == "__main__":
    main()
