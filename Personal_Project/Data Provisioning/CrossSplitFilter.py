#!/usr/bin/env python3
# ED_CrossSplitFilter.py — EmpatheticDialogues ONLY (no argparse)
# Loads Clean/hf_disk, assigns each near-duplicate hash to exactly one split by
# precedence train > validation/dev/valid > test, then saves to CleanNoLeak/hf_disk.

from __future__ import annotations
import re, hashlib
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# ---- USER CONFIG ----
ED_BASE = Path(r"C:\Users\adm\Desktop\IndividualChallenge\Dataset\EmpatheticDialogues\EmpatheticDialogues\EmotionDataset")
SPLIT_PREFERRED = ["train","validation","dev","valid","test"]

try:
    from datasets import load_from_disk, Dataset, DatasetDict
except Exception as e:
    raise SystemExit(f"'datasets' package is required. pip install datasets  [{e}]")

URL_RE   = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
TAG_RE   = re.compile(r"(@\w+|#\w+)")
HTML_RE  = re.compile(r"<[^>]+>")
WS_RE    = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if s is None: return ""
    t = str(s)
    t = URL_RE.sub("", t)
    t = TAG_RE.sub("", t)
    t = HTML_RE.sub(" ", t)
    t = t.lower()
    t = WS_RE.sub(" ", t).strip()
    return t

def h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def choose_text_col(df: pd.DataFrame) -> str:
    for k in ["text","utterance","Utterance","sentence","content","response"]:
        if k in df.columns: return k
    for c in df.columns:
        if df[c].dtype == object: return c
    return df.columns[0]

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

def main():
    clean_dir = ED_BASE / "Clean" / "hf_disk"
    if not clean_dir.exists():
        raise SystemExit(f"Clean/hf_disk not found: {clean_dir}\nRun ED_Deduplicate.py first.")

    ds = load_from_disk(str(clean_dir))
    pdf = pdf_from_ds(ds)

    frames = []
    for sp, df in pdf.items():
        if df is None or df.empty: continue
        tcol = choose_text_col(df)
        tmp = df[[tcol]].copy()
        tmp["_split"] = sp
        tmp["_norm"] = tmp[tcol].astype(str).map(normalize_text)
        tmp["_hash"] = tmp["_norm"].map(h)
        frames.append(tmp[["_split","_hash"]])

    if not frames:
        raise SystemExit("No data to process.")
    pooled = pd.concat(frames, ignore_index=True)

    prio = {name: i for i, name in enumerate(SPLIT_PREFERRED)}
    pooled["_prio"] = pooled["_split"].map(lambda s: prio.get(s, 99))

    keep_split_for_hash = (
        pooled.sort_values(["_hash","_prio"])
              .drop_duplicates(subset=["_hash"], keep="first")
              .set_index("_hash")["_split"].to_dict()
    )

    result: Dict[str, pd.DataFrame] = {}
    for sp, df in pdf.items():
        if df is None or df.empty:
            result[sp] = df; continue
        tcol = choose_text_col(df)
        tmp = df.copy()
        tmp["_norm"] = tmp[tcol].astype(str).map(normalize_text)
        tmp["_hash"] = tmp["_norm"].map(h)
        tmp = tmp[tmp["_hash"].map(lambda hh: keep_split_for_hash.get(hh) == sp)].copy()
        tmp.drop(columns=["_norm","_hash"], inplace=True)
        result[sp] = tmp.reset_index(drop=True)
        print(f"[cross-split] {sp}: {len(df)} -> {len(tmp)}")

    out_dir = ED_BASE / "CleanNoLeak" / "hf_disk"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_out = ds_from_pdf(result)
    ds_out.save_to_disk(str(out_dir))
    print(f"[save] CleanNoLeak splits → {out_dir}")

if __name__ == "__main__":
    main()
