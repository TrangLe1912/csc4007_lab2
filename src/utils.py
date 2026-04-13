from __future__ import annotations
import hashlib
import random
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha1_text(s: str) -> str:
    return hashlib.sha1(str(s).encode('utf-8', errors='ignore')).hexdigest()
