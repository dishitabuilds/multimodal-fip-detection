"""Shared preprocessing for the auxiliary SMS phishing experiment."""
from __future__ import annotations

import re

URL = re.compile(
    r"(?:https?[:：]?(?:\/\/|ःll)?|www\.)\S+|\b[a-z0-9][a-z0-9.-]+\.(?:com|net|org|co|in|uk|io|ly)(?:/\S*)?",
    re.I,
)


def normalize_message(text: str) -> str:
    """Remove exact URL identities so a classifier cannot memorize domains."""
    if not isinstance(text, str):
        raise TypeError("message must be text")
    return URL.sub(" URL ", text.lower()).strip()


def duplicate_key(text: str) -> str:
    # Numeric changes in the same lure must stay within one split.
    return re.sub(r"\W+", " ", re.sub(r"\d+", "0", normalize_message(text))).strip()
