from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import sha1_text


def audit_schema_missingness(df: pd.DataFrame) -> dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    return {
        "n_rows": int(len(df)),
        "missing_text_count": int(df["text"].isna().sum()),
        "empty_text_count": int(texts.str.strip().eq("").sum()),
        "missing_label_count": int(df["label"].isna().sum()),
        "label_counts": df["label"].value_counts(dropna=False).to_dict(),
    }


def audit_distribution_length(df: pd.DataFrame) -> dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    lens_chars = texts.map(len).to_numpy()
    lens_words = texts.map(lambda x: len(x.split())).to_numpy()
    vc = df["label"].value_counts(dropna=False)
    return {
        "imbalance_ratio_max_over_min": float(vc.max() / max(vc.min(), 1)) if len(vc) else 0.0,
        "len_chars_min": int(lens_chars.min()) if len(lens_chars) else 0,
        "len_chars_median": int(np.median(lens_chars)) if len(lens_chars) else 0,
        "len_chars_p95": int(np.percentile(lens_chars, 95)) if len(lens_chars) else 0,
        "len_chars_max": int(lens_chars.max()) if len(lens_chars) else 0,
        "len_words_min": int(lens_words.min()) if len(lens_words) else 0,
        "len_words_median": int(np.median(lens_words)) if len(lens_words) else 0,
        "len_words_p95": int(np.percentile(lens_words, 95)) if len(lens_words) else 0,
        "len_words_max": int(lens_words.max()) if len(lens_words) else 0,
    }


def audit_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    texts = df["text"].fillna("").astype(str)
    hashes = texts.map(sha1_text)
    dup_mask = hashes.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    return {
        "exact_dup_count": dup_count,
        "exact_dup_ratio": float(dup_count / len(df)) if len(df) else 0.0,
    }


def render_audit_md(path: str | Path, title: str, sections: list[tuple[str, dict[str, Any]]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}\n\n"]
    for sec_title, obj in sections:
        lines.append(f"## {sec_title}\n")
        for k, v in obj.items():
            lines.append(f"- **{k}**: {v}\n")
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")
