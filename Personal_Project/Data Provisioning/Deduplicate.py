#!/usr/bin/env python3
# ED_Deduplicate.py — EmpatheticDialogues ONLY (no argparse)
# Loads HF dataset from your ED path, cleans & deduplicates within each split,
# optional token clipping/dropping, and saves to Clean/hf_disk.

from __future__ import annotations
import re, hashlib, datetime as dt
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# ---- USER CONFIG (no argparse) ----
ED_BASE = Path(r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\EmotionDataset")
TOKENIZER_JSON = None  # e.g., r"C:\...\tokenizer_v9.json" or leave as None
MAX_TOKENS = 0         # 0 = disable token clipping/dropping
DROP_LONG = False      # True=drop >MAX_TOKENS, False=clip in place
STRIP_HTML = False
MIN_ALPHA_RATIO = 0.35
MIN_LEN_CHARS = 3

# ---- Imports that might be optional ----
try:
    from datasets import load_from_disk, Dataset, DatasetDict
except Exception as e:
    raise SystemExit(f"'datasets' package is required. pip install datasets  [{e}]")

try:
    from tokenizers import Tokenizer
except Exception:
    Tokenizer = None

# ---- Text utils ----
URL_RE   = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
TAG_RE   = re.compile(r"(@\w+|#\w+)")
HTML_RE  = re.compile(r"<[^>]+>")
WS_RE    = re.compile(r"\s+")
ALNUM_RE = re.compile(r"[A-Za-z0-9]")

def normalize_text(s: str, strip_html: bool = False) -> str:
    if s is None: return ""
    t = str(s)
    t = URL_RE.sub("", t)
    t = TAG_RE.sub("", t)
    if strip_html: t = HTML_RE.sub(" ", t)
    t = t.lower()
    t = WS_RE.sub(" ", t).strip()
    return t

def min_info_ok(norm: str, *, min_alpha_ratio: float, min_len_chars: int) -> bool:
    if len(norm) < min_len_chars: return False
    if not ALNUM_RE.search(norm): return False
    alnum = sum(ch.isalnum() for ch in norm)
    ratio = alnum / max(1, len(norm))
    return ratio >= min_alpha_ratio

def h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def choose_text_col(df: pd.DataFrame) -> str:
    for k in ["text","utterance","Utterance","sentence","content","response"]:
        if k in df.columns: return k
    for c in df.columns:
        if df[c].dtype == object: return c
    return df.columns[0]

def load_tokenizer(path: Optional[str]):
    if not path or Tokenizer is None: return None
    try: return Tokenizer.from_file(path)
    except Exception: return None

def count_tokens(tok, text: str) -> int:
    if tok is None: return 0
    try: return len(tok.encode(str(text)).ids)
    except Exception: return 0

def clip_to_tokens(tok, text: str, max_tokens: int) -> str:
    if tok is None or max_tokens <= 0: return text
    enc = tok.encode(str(text)); ids = enc.ids
    if len(ids) <= max_tokens: return text
    return tok.decode(ids[:max_tokens])

# ---- HF helpers ----
def looks_like_hf_root(d: Path) -> bool:
    if not d.is_dir(): return False
    if (d/"dataset_dict.json").exists() or (d/"dataset_info.json").exists(): return True
    if any((d/sp).is_dir() for sp in ["train","validation","valid","dev","test"]) and list(d.rglob("data-*.arrow")):
        return True
    return False

def find_hf_saved_root(root: Path) -> Optional[Path]:
    if looks_like_hf_root(root): return root
    cand = root / "hf_disk"
    if looks_like_hf_root(cand): return cand
    for d in root.rglob("*"):
        if looks_like_hf_root(d): return d
    return None

def pdf_from_ds(ds) -> Dict[str, pd.DataFrame]:
    out = {}
    for name in ds.keys():
        if name in {"train","validation","valid","dev","test"}:
            out[name] = ds[name].to_pandas()
    return out

def ds_from_pdf(splits: Dict[str, pd.DataFrame]) -> DatasetDict:
    out = {}
    for k, df in splits.items():
        if df is None or df.empty: continue
        out[k] = Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)
    return DatasetDict(out)

# ---- main ----
def main():
    ed_hf = find_hf_saved_root(ED_BASE)
    if ed_hf is None:
        raise SystemExit(f"Could not find HF dataset under: {ED_BASE}")

    ds = load_from_disk(str(ed_hf))
    pdf = pdf_from_ds(ds)

    tok = load_tokenizer(TOKENIZER_JSON)

    # build pooled normalized view
    pooled_rows = []
    for sp, df in pdf.items():
        if df is None or df.empty: continue
        tcol = choose_text_col(df)
        tmp = pd.DataFrame({"split": sp, "text": df[tcol].astype(str)})
        tmp["norm"] = tmp["text"].map(lambda s: normalize_text(s, STRIP_HTML))
        tmp["hash"] = tmp["norm"].map(h)
        if tok and MAX_TOKENS > 0:
            tmp["n_tokens"] = tmp["text"].map(lambda s: count_tokens(tok, s))
            if DROP_LONG:
                tmp["keep_len"] = tmp["n_tokens"].le(MAX_TOKENS)
                tmp["text_clean"] = tmp["text"]
            else:
                tmp["keep_len"] = True
                tmp["text_clean"] = tmp["text"].map(lambda s: clip_to_tokens(tok, s, MAX_TOKENS))
        else:
            tmp["keep_len"] = True
            tmp["text_clean"] = tmp["text"]
        tmp["keep_min"] = tmp["norm"].map(lambda z: min_info_ok(z, min_alpha_ratio=MIN_ALPHA_RATIO, min_len_chars=MIN_LEN_CHARS))
        pooled_rows.append(tmp)

    if not pooled_rows:
        raise SystemExit("No ED data found.")

    pooled = pd.concat(pooled_rows, ignore_index=True)
    pooled["row_id"] = pooled.index
    pooled = pooled.sort_values(["split","row_id"])
    pooled["dup_within_split"] = pooled.duplicated(subset=["split","hash"], keep="first")
    pooled["keep"] = (~pooled["dup_within_split"]) & pooled["keep_len"] & pooled["keep_min"]
    kept = pooled[pooled["keep"]].copy()

    # reconstruct per-split DataFrames
    cleaned: Dict[str, pd.DataFrame] = {}
    map_clean_text = dict(zip(kept["hash"], kept["text_clean"]))

    for sp, df in pdf.items():
        if df is None or df.empty:
            cleaned[sp] = pd.DataFrame(); continue
        tcol = choose_text_col(df)
        df2 = df.copy()
        df2["_norm"] = df2[tcol].astype(str).map(lambda s: normalize_text(s, STRIP_HTML))
        df2["_hash"] = df2["_norm"].map(h)
        # keep only hashes that survived
        keep_hashes = set(kept[kept["split"]==sp]["hash"].tolist())
        df2 = df2[df2["_hash"].isin(keep_hashes)].copy()
        # dedup within split
        df2 = df2.loc[~df2.duplicated(subset=["_hash"], keep="first")].copy()
        # replace text if we clipped
        if tok and MAX_TOKENS > 0 and not DROP_LONG:
            df2[tcol] = df2["_hash"].map(lambda x: map_clean_text.get(x, None)).fillna(df2[tcol])
        df2.drop(columns=["_norm","_hash"], inplace=True)
        cleaned[sp] = df2.reset_index(drop=True)

    out_dir = ED_BASE / "Clean" / "hf_disk"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_out = ds_from_pdf(cleaned)
    ds_out.save_to_disk(str(out_dir))
    print(f"[save] Clean splits → {out_dir}")

if __name__ == "__main__":
    main()
