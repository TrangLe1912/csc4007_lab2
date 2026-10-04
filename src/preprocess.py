from __future__ import annotations

import re
import unicodedata

import regex as regex_u

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\b\d+(?:[.,/]\d+)?\b")
PUNCT_RE = re.compile(r"([!?.,;:()\[\]{}\"'\-_/])")


def segment_vietnamese(text: str, method: str = "none") -> str:
    """Optional Vietnamese word segmentation.

    'none' keeps the cleaned text unchanged.
    'underthesea' joins multi-syllable words with underscores.
    """
    if method == "none":
        return text
    if method != "underthesea":
        raise ValueError(f"Unsupported Vietnamese segmenter: {method}")

    try:
        from underthesea import word_tokenize
    except ImportError as exc:
        raise ImportError(
            "Vietnamese segmentation requires underthesea. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return word_tokenize(text, format="text")


def basic_clean_text(
    text: str,
    lowercase: bool = True,
    replace_url: bool = True,
    replace_email: bool = True,
    replace_number: bool = False,
    keep_punct: bool = True,
    normalize_unicode: bool = True,
    vi_segment: str = "none",
) -> str:
    """Moderate text cleaning for the Lab 2 comparison experiments.

    The function deliberately avoids aggressive cleaning. For Vietnamese text,
    Unicode NFC normalization is safe by default and word segmentation is an
    explicit experimental choice rather than a mandatory rule.
    """
    if text is None:
        return ""

    t = str(text).strip()
    if normalize_unicode:
        t = unicodedata.normalize("NFC", t)

    t = t.replace("\u00a0", " ")
    t = HTML_TAG_RE.sub(" ", t)
    t = regex_u.sub(r"\p{C}+", " ", t)

    if replace_url:
        t = URL_RE.sub(" <URL> ", t)
    if replace_email:
        t = EMAIL_RE.sub(" <EMAIL> ", t)
    if replace_number:
        t = DIGIT_RE.sub(" <NUM> ", t)
    if lowercase:
        t = t.lower()

    if keep_punct:
        t = PUNCT_RE.sub(r" \1 ", t)
    else:
        t = PUNCT_RE.sub(" ", t)

    t = MULTI_SPACE_RE.sub(" ", t).strip()

    if vi_segment != "none":
        t = segment_vietnamese(t, method=vi_segment)
        t = MULTI_SPACE_RE.sub(" ", t).strip()

    return t
