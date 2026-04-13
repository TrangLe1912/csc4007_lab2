from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "labels": labels,
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def save_metrics(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Metrics Summary",
        f"- dataset: {metrics['dataset']}",
        f"- vectorizer: {metrics['vectorizer']}",
        f"- model: {metrics['model']}",
        f"- val_accuracy: {metrics['val']['accuracy']:.4f}",
        f"- val_macro_f1: {metrics['val']['macro_f1']:.4f}",
        f"- test_accuracy: {metrics['test']['accuracy']:.4f}",
        f"- test_macro_f1: {metrics['test']['macro_f1']:.4f}",
        "",
        "## How to read the result on IMDB",
        "- IMDB khá cân bằng nên accuracy và macro-F1 có thể gần nhau.",
        "- Dù vậy vẫn cần báo cáo macro-F1 để giữ thói quen đánh giá đúng khi sang dữ liệu lệch lớp.",
        "",
        "## Test per-class report",
    ]
    for label, stats in metrics["test"]["classification_report"].items():
        if not isinstance(stats, dict):
            continue
        lines.append(
            f"- class `{label}`: precision={stats['precision']:.4f}, recall={stats['recall']:.4f}, "
            f"f1={stats['f1-score']:.4f}, support={stats['support']}"
        )
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix (IMDB)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
