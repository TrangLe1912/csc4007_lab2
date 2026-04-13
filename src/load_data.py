from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


LABEL_MAP_IMDB = {0: "negative", 1: "positive"}


def load_imdb(max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    ds = load_dataset("imdb")
    df_train = pd.DataFrame(ds["train"])
    df_test = pd.DataFrame(ds["test"])
    df_train["split_orig"] = "hf_train"
    df_test["split_orig"] = "hf_test"
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["id"] = range(len(df))
    df["label"] = df["label"].map(LABEL_MAP_IMDB)

    if max_rows is not None and max_rows < len(df):
        _, df = train_test_split(
            df,
            test_size=int(max_rows),
            random_state=seed,
            stratify=df["label"],
        )

    return df[["id", "text", "label", "split_orig"]].reset_index(drop=True).copy()


def load_local_csv(
    data_path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    df = pd.read_csv(data_path)
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}. Available: {list(df.columns)}")

    out = df.copy()
    if "id" not in out.columns:
        out.insert(0, "id", range(len(out)))

    out = out.rename(columns={text_col: "text", label_col: "label"})
    keep_cols = ["id", "text", "label"]
    if "split_orig" in out.columns:
        keep_cols.append("split_orig")
    return out[keep_cols].copy()


def load_dataset_any(
    name: str,
    max_rows: int | None = None,
    data_path: str | Path | None = None,
    text_col: str = "text",
    label_col: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    if name == "imdb":
        return load_imdb(max_rows=max_rows, seed=seed)
    if name == "local_csv":
        if data_path is None:
            raise ValueError("data_path is required when dataset=local_csv")
        df = load_local_csv(data_path=data_path, text_col=text_col, label_col=label_col)
        if max_rows is not None and max_rows < len(df):
            _, df = train_test_split(
                df,
                test_size=int(max_rows),
                random_state=seed,
                stratify=df["label"] if df["label"].nunique() > 1 else None,
            )
        return df.reset_index(drop=True)
    raise ValueError(f"Unsupported dataset: {name}")
