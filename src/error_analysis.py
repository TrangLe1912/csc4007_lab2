from __future__ import annotations

from pathlib import Path

import pandas as pd


ERROR_HINTS = {
    "en": [
        "negation / contrast",
        "mixed sentiment",
        "sarcasm / irony",
        "very long review",
        "entity-heavy / domain-specific wording",
        "prediction confident but wrong",
    ],
    "vi": [
        "multi-topic article",
        "ambiguous / overlapping category",
        "named-entity dominated",
        "short / low-information text",
        "cross-category vocabulary",
        "prediction confident but wrong",
    ],
}


def build_error_analysis(df_test: pd.DataFrame, y_pred, y_proba=None) -> pd.DataFrame:
    out = df_test.copy().reset_index(drop=True)
    out["pred_label"] = list(y_pred)
    out["is_error"] = out["label"] != out["pred_label"]

    if y_proba is not None:
        out["pred_confidence"] = y_proba.max(axis=1)
    else:
        out["pred_confidence"] = None

    errors = out[out["is_error"]].copy()
    if errors["pred_confidence"].notna().any():
        errors = errors.sort_values(["pred_confidence"], ascending=[False])

    errors["suggested_error_group"] = ""
    errors["student_note"] = ""

    preferred = ["id", "text", "label", "pred_label", "pred_confidence"]
    metadata = [
        c
        for c in errors.columns
        if c not in preferred
        and c not in {"is_error", "suggested_error_group", "student_note"}
    ]
    return errors[
        preferred + metadata + ["suggested_error_group", "student_note"]
    ].reset_index(drop=True)


def save_error_analysis(
    errors: pd.DataFrame,
    out_dir: Path,
    min_expected: int = 10,
    language: str = "en",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out_dir / "error_analysis.csv", index=False)

    hints = ERROR_HINTS.get(language, ERROR_HINTS["en"])
    context = "VietNewsSense / Vietnamese news" if language == "vi" else "IMDB"

    lines = [
        "# Error Analysis Summary",
        f"- n_error_rows_exported: {len(errors)}",
        f"- minimum_errors_to_review_in_report: {min_expected}",
        "",
        f"## Suggested error groups for {context}",
    ]
    for item in hints:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Student task",
        f"1. Chọn ít nhất {min_expected} dòng trong error_analysis.csv.",
        "2. Gom các lỗi thành 2–4 nhóm.",
        "3. Ghi nguyên nhân khả dĩ và đề xuất cải thiện pipeline.",
    ]
    (out_dir / "error_analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
