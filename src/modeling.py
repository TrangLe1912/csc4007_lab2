from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_vectorizer(
    name: str = "tfidf",
    max_features: int = 20000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    analyzer: str = "word",
):
    if ngram_min < 1 or ngram_max < ngram_min:
        raise ValueError("Require 1 <= ngram_min <= ngram_max")
    if analyzer not in {"word", "char", "char_wb"}:
        raise ValueError(f"Unsupported analyzer: {analyzer}")

    kwargs = {
        "max_features": max_features,
        "ngram_range": (ngram_min, ngram_max),
        "min_df": 1,
        "analyzer": analyzer,
    }
    if name == "bow":
        return CountVectorizer(**kwargs)
    if name == "tfidf":
        return TfidfVectorizer(**kwargs)
    raise ValueError(f"Unsupported vectorizer: {name}")


def build_estimator(name: str = "logreg", seed: int = 42):
    if name == "logreg":
        return LogisticRegression(max_iter=600, random_state=seed)
    if name == "linearsvm":
        base = LinearSVC(random_state=seed)
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(f"Unsupported model: {name}")


def build_pipeline(
    vectorizer_name: str = "tfidf",
    model_name: str = "logreg",
    max_features: int = 20000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    analyzer: str = "word",
    seed: int = 42,
) -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                build_vectorizer(
                    vectorizer_name,
                    max_features=max_features,
                    ngram_min=ngram_min,
                    ngram_max=ngram_max,
                    analyzer=analyzer,
                ),
            ),
            ("model", build_estimator(model_name, seed=seed)),
        ]
    )
