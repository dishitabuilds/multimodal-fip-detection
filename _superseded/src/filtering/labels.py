"""Map each publisher's verdict wording onto our binary label.

Every fact-checker uses its own rating vocabulary — BOOM says "False", Factly
says "Misleading", Vishvas News says "झूठ", Newschecker says "Misplaced
Context". ClaimReview's `textualRating` preserves those strings verbatim, so
we need an explicit mapping table rather than a heuristic.

The design decision worth defending in the report: "misleading" and "partly
false" map to FAKE, not to a third class. A truncated-axis chart is factually
"partly true" and still a fraudulent promotion — for a fraud detector, the
operative question is whether the post deceives, not whether every pixel of
it is false. Items we genuinely cannot place map to UNKNOWN and are held out
of training rather than guessed at.
"""

from __future__ import annotations

import re

from ..collection.schema import LABEL_FAKE, LABEL_REAL, LABEL_UNKNOWN

# Order matters: the first pattern that matches wins, so put the specific
# "not false" style phrasings above the generic "false" pattern.
_RULES: list[tuple[str, str]] = [
    # --- explicitly true --------------------------------------------------
    (r"\b(true|correct|accurate|verified|genuine|authentic|confirmed)\b", LABEL_REAL),
    (r"\b(mostly true|largely true)\b", LABEL_REAL),
    (r"(सही|सत्य|सच)", LABEL_REAL),

    # --- explicitly false / deceptive ------------------------------------
    (r"\b(false|fake|fabricated|hoax|forged|doctored|morphed|manipulated)\b", LABEL_FAKE),
    (r"\b(misleading|miscaptioned|misattributed|misrepresented|misplaced context)\b", LABEL_FAKE),
    (r"\b(partly false|partially false|half true|mixture|distorted|exaggerated)\b", LABEL_FAKE),
    (r"\b(scam|fraud|deepfake|ai[- ]generated|altered|edited video)\b", LABEL_FAKE),
    (r"\b(no evidence|unsubstantiated|baseless|unfounded|debunked|not true)\b", LABEL_FAKE),
    (r"\b(satire|parody)\b", LABEL_FAKE),
    (r"(झूठ|फर्जी|फ़र्ज़ी|भ्रामक|गलत|ग़लत|असत्य|अफवाह)", LABEL_FAKE),

    # --- unresolved -------------------------------------------------------
    (r"\b(unverified|unclear|inconclusive|insufficient evidence|research ongoing)\b", LABEL_UNKNOWN),
]

_COMPILED = [(re.compile(p, re.I | re.U), lab) for p, lab in _RULES]

# Article-title conventions used when a site emits no ClaimReview rating.
_TITLE_FAKE = re.compile(
    r"^(fact check:?\s*)?(no,|scam alert|fake|viral (image|video|post|claim) (is|of))|"
    r"(is fake|is a deepfake|is false|is misleading|is doctored|is morphed|"
    r"falsely (shared|linked|viral|claimed)|does not show|did not|not real)",
    re.I,
)
_TITLE_REAL = re.compile(r"^(yes,|true:|confirmed:)|\b(is (true|real|genuine|authentic))\b", re.I)


def map_verdict(verdict_raw: str) -> str:
    """Map a publisher's rating string to fake / real / unknown."""
    v = (verdict_raw or "").strip()
    if not v:
        return LABEL_UNKNOWN
    for pattern, label in _COMPILED:
        if pattern.search(v):
            return label
    return LABEL_UNKNOWN


def label_from_title(title: str) -> str:
    """Fallback for records with no machine-readable verdict."""
    t = (title or "").strip()
    if not t:
        return LABEL_UNKNOWN
    if _TITLE_REAL.search(t):
        return LABEL_REAL
    if _TITLE_FAKE.search(t):
        return LABEL_FAKE
    return LABEL_UNKNOWN


def assign_label(record) -> str:
    """Verdict first, title heuristic second, unknown last."""
    label = map_verdict(record.verdict_raw)
    if label == LABEL_UNKNOWN:
        label = label_from_title(record.title)
    return label
