from __future__ import annotations

import re

import regex as regex_u

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\b\d+(?:[.,/]\d+)?\b")
PUNCT_RE = re.compile(r"([!?.,;:()\[\]{}\"'\-_/])")


def basic_clean_text(
    text: str,
    lowercase: bool = True,
    replace_url: bool = True,
    replace_email: bool = True,
    replace_number: bool = False,
    keep_punct: bool = True,
) -> str:
    """Tiền xử lý mức vừa phải cho IMDB.

    Chủ đích của Lab 2 là để sinh viên so sánh pipeline,
    nên hàm này chỉ làm sạch vừa phải thay vì “dọn quá tay”.
    """
    if text is None:
        return ""

    t = str(text).strip()
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
    return t
