from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.audit_core import (
    audit_distribution_length,
    audit_duplicates,
    audit_schema_missingness,
    render_audit_md,
)
from src.error_analysis import build_error_analysis, save_error_analysis
from src.evaluate import compute_metrics, save_confusion_matrix, save_metrics
from src.load_data import load_dataset_any
from src.modeling import build_pipeline
from src.preprocess import basic_clean_text
from src.split import make_splits
from src.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imdb", choices=["imdb", "local_csv"])
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--language", default="en", choices=["en", "vi"])
    ap.add_argument("--vi_segment", default="none", choices=["none", "underthesea"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--vectorizer", default="tfidf", choices=["bow", "tfidf"])
    ap.add_argument("--model", default="logreg", choices=["logreg", "linearsvm"])
    ap.add_argument("--analyzer", default="word", choices=["word", "char", "char_wb"])
    ap.add_argument("--max_features", type=int, default=20000)
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--replace_number", action="store_true")
    ap.add_argument("--drop_punct", action="store_true")
    ap.add_argument("--output_dir", default="outputs")
    args = ap.parse_args()

    if args.vi_segment != "none" and args.language != "vi":
        raise ValueError("--vi_segment is only intended for --language vi")
    if args.ngram_min < 1 or args.ngram_max < args.ngram_min:
        raise ValueError("Require 1 <= --ngram_min <= --ngram_max")

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    for sub in [
        "logs",
        "splits",
        "metrics",
        "figures",
        "error_analysis",
        "pipeline",
        "predictions",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    df = load_dataset_any(
        name=args.dataset,
        max_rows=args.max_rows,
        data_path=args.data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        seed=args.seed,
    )
    df["text"] = df["text"].fillna("").astype(str)

    sec_before = [
        ("Schema / Missingness", audit_schema_missingness(df)),
        ("Distribution / Length", audit_distribution_length(df)),
        ("Duplicates", audit_duplicates(df)),
    ]
    render_audit_md(
        out_dir / "logs" / "audit_before.md",
        "Audit BEFORE preprocessing",
        sec_before,
    )

    df_clean = df.copy()
    df_clean["text"] = df_clean["text"].map(
        lambda x: basic_clean_text(
            x,
            lowercase=True,
            replace_url=True,
            replace_email=True,
            replace_number=args.replace_number,
            keep_punct=not args.drop_punct,
            normalize_unicode=True,
            vi_segment=args.vi_segment,
        )
    )

    sec_after = [
        ("Schema / Missingness", audit_schema_missingness(df_clean)),
        ("Distribution / Length", audit_distribution_length(df_clean)),
        ("Duplicates", audit_duplicates(df_clean)),
    ]
    render_audit_md(
        out_dir / "logs" / "audit_after.md",
        "Audit AFTER preprocessing",
        sec_after,
    )
    render_audit_md(
        out_dir / "logs" / "data_audit.md",
        "Audit Summary",
        sec_before + sec_after,
    )

    splits = make_splits(df_clean, seed=args.seed)
    for name, d in splits.items():
        d.to_csv(out_dir / "splits" / f"{name}.csv", index=False)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    pipe = build_pipeline(
        vectorizer_name=args.vectorizer,
        model_name=args.model,
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        analyzer=args.analyzer,
        seed=args.seed,
    )
    pipe.fit(train_df["text"], train_df["label"])

    y_pred_val = pipe.predict(val_df["text"])
    y_pred_test = pipe.predict(test_df["text"])

    try:
        y_proba_test = pipe.predict_proba(test_df["text"])
    except Exception:
        y_proba_test = None

    metrics = {
        "dataset": args.dataset,
        "dataset_path": args.data_path if args.dataset == "local_csv" else None,
        "language": args.language,
        "vi_segment": args.vi_segment,
        "seed": args.seed,
        "vectorizer": args.vectorizer,
        "analyzer": args.analyzer,
        "model": args.model,
        "max_features": args.max_features,
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "replace_number": args.replace_number,
        "drop_punct": args.drop_punct,
        "splits": {k: int(len(v)) for k, v in splits.items()},
        "val": compute_metrics(val_df["label"], y_pred_val),
        "test": compute_metrics(test_df["label"], y_pred_test),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Split trước, fit vectorizer trên train, evaluate trên val/test để tránh leakage. "
            "Vietnamese word segmentation is an experimental choice, not a mandatory rule."
        ),
    }
    save_metrics(metrics, out_dir / "metrics")
    cm_name = "IMDB" if args.dataset == "imdb" else "Local CSV / VietNewsSense"
    save_confusion_matrix(
        test_df["label"],
        y_pred_test,
        out_dir / "figures" / "confusion_matrix.png",
        title=f"Confusion Matrix ({cm_name})",
    )

    pred_df = test_df.copy()
    pred_df["pred_label"] = y_pred_test
    pred_df.to_csv(out_dir / "predictions" / "test_predictions.csv", index=False)

    errors = build_error_analysis(test_df, y_pred_test, y_proba=y_proba_test)
    min_expected = 5 if args.language == "vi" else 10
    save_error_analysis(
        errors,
        out_dir / "error_analysis",
        min_expected=min_expected,
        language=args.language,
    )

    joblib.dump(pipe, out_dir / "pipeline" / "model_pipeline.joblib")

    summary = {
        "dataset": args.dataset,
        "language": args.language,
        "vi_segment": args.vi_segment,
        "seed": args.seed,
        "vectorizer": args.vectorizer,
        "analyzer": args.analyzer,
        "model": args.model,
        "max_features": args.max_features,
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "replace_number": args.replace_number,
        "drop_punct": args.drop_punct,
        "metrics_test_macro_f1": metrics["test"]["macro_f1"],
        "metrics_test_accuracy": metrics["test"]["accuracy"],
    }
    (out_dir / "logs" / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"DONE. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
