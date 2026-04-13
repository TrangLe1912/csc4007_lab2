from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def build_vectorizer(name: str = 'tfidf', max_features: int = 20000, ngram_max: int = 2):
    ngram_range = (1, ngram_max)
    if name == 'bow':
        return CountVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=1)
    if name == 'tfidf':
        return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=1)
    raise ValueError(f'Unsupported vectorizer: {name}')


def build_estimator(name: str = 'logreg', seed: int = 42):
    if name == 'logreg':
        return LogisticRegression(max_iter=400, random_state=seed)
    if name == 'linearsvm':
        base = LinearSVC(random_state=seed)
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(f'Unsupported model: {name}')


def build_pipeline(vectorizer_name: str = 'tfidf', model_name: str = 'logreg', max_features: int = 20000,
                   ngram_max: int = 2, seed: int = 42) -> Pipeline:
    return Pipeline([
        ('vectorizer', build_vectorizer(vectorizer_name, max_features=max_features, ngram_max=ngram_max)),
        ('model', build_estimator(model_name, seed=seed)),
    ])
