"""Decide whether a fact-check is about investment / financial fraud.

Indian fact-check archives are overwhelmingly political. A keyword sweep of
Alt News, for example, returns thousands of political checks and a handful of
financial ones. So this filter has to be reasonably precise or the dataset
fills up with communal-video debunks.

Approach: weighted lexicon over title + claim + body + image alt/OCR text,
covering English, Devanagari Hindi and romanised Hinglish (scam creatives are
routinely written as "paisa double", "guaranteed profit", "SEBI registered").
Weights are deliberately hand-set and inspectable rather than learned — at
collection time we have no labels yet, and a transparent rule is easier to
defend in the report than an opaque one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Lexicon. Weight ~= how strongly the term implies "investment promotion".
# ---------------------------------------------------------------------------

_STRONG = 3.0   # nearly conclusive on its own
_MEDIUM = 1.5   # financial but could be ordinary news
_WEAK = 0.7     # supporting evidence only

LEXICON: dict[str, float] = {}


def _add(terms: list[str], weight: float) -> None:
    for t in terms:
        LEXICON[t.lower()] = weight


# --- regulators and registration claims (the classic legitimacy prop) ------
_add([
    "sebi", "securities and exchange board", "rbi registered", "sebi registered",
    "sebi registration", "amfi", "irdai", "nse", "bse", "sebi approved",
    "rbi approved", "सेबी", "आरबीआई",
], _STRONG)

# --- investment-scam vocabulary ------------------------------------------
_add([
    "investment scam", "investment fraud", "ponzi", "chit fund", "mlm scheme",
    "multi level marketing", "money doubling", "double your money", "paisa double",
    "guaranteed return", "guaranteed profit", "assured return", "fixed profit",
    "high return", "risk free return", "profit screenshot", "trading tips",
    "stock tips", "intraday tips", "sure shot call", "jackpot call",
    "fake trading app", "fake trading platform", "unregistered advisor",
    "unregistered advisory", "investment advisory scam", "pump and dump",
    "निवेश घोटाला", "निवेश धोखाधड़ी", "पैसा डबल", "गारंटीड रिटर्न",
], _STRONG)

# --- crypto / forex --------------------------------------------------------
_add([
    "bitcoin", "cryptocurrency", "crypto trading", "forex trading", "binary options",
    "usdt", "ethereum", "crypto scam", "mining scheme", "क्रिप्टो", "बिटकॉइन",
], _MEDIUM)

# --- deepfake celebrity endorsement (the dominant 2024-26 pattern) --------
_add([
    "deepfake", "ai generated video", "morphed video", "fake endorsement",
    "endorsing investment", "promoting investment", "promoting trading",
    "investment platform", "trading platform", "डीपफेक",
], _MEDIUM)

# --- general finance -------------------------------------------------------
_add([
    "mutual fund", "share market", "stock market", "demat", "ipo", "nifty",
    "sensex", "portfolio", "dividend", "brokerage", "trading account",
    "loan app", "instant loan", "personal loan", "emi", "upi", "net banking",
    "bank account", "fixed deposit", "insurance policy", "pension scheme",
    "शेयर बाजार", "म्यूचुअल फंड", "निवेश", "बैंक खाता", "लोन",
], _MEDIUM)

# --- money words, weak on their own ---------------------------------------
_add([
    "invest", "investor", "investment", "profit", "return on investment", "roi",
    "lakh", "crore", "rupees", "money", "earning", "income", "wealth",
    "scheme", "scam", "fraud", "financial", "finance", "trader", "trading",
    "रुपये", "लाख", "करोड़", "मुनाफा", "कमाई", "घोटाला",
], _WEAK)

# Terms that strongly indicate a *political* check — used to damp false hits,
# since political stories mention "crore" and "scam" constantly.
NEGATIVE_LEXICON: dict[str, float] = {}
for _t in [
    "communal", "riot", "temple", "mosque", "church", "hindu", "muslim",
    "election", "vote", "voter", "ballot", "rally", "constituency", "mla", "mp seat",
    "cricket", "bollywood", "actor", "film", "movie", "vaccine", "covid",
    "pakistan army", "border", "terrorist", "encounter", "protest march",
]:
    NEGATIVE_LEXICON[_t] = 1.0

# Queries used to hit the Google Fact Check API. Ordered roughly by yield.
DISCOVERY_QUERIES = [
    "investment scam", "trading app fraud", "SEBI", "stock market fake",
    "money doubling scheme", "ponzi scheme", "deepfake investment",
    "cryptocurrency scam India", "fake trading platform", "guaranteed returns",
    "share market tips fake", "loan app fraud", "RBI fake", "chit fund scam",
    "mutual fund fake", "financial fraud", "निवेश घोटाला", "शेयर बाजार फर्जी",
    "पैसा डबल", "क्रिप्टो धोखाधड़ी",
]


# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"[^\w\sऀ-ॿ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class FinanceVerdict:
    is_finance: bool
    score: float
    hits: list[str]
    negative_hits: list[str]


def score_text(
    title: str = "",
    claim: str = "",
    body: str = "",
    image_text: str = "",
    threshold: float = 4.0,
) -> FinanceVerdict:
    """Weighted lexicon score over the parts of a record.

    Title and claim are weighted higher than body: fact-check articles quote
    the scam creative at length in the body, but a passing mention of "crore"
    in paragraph nine says much less than "investment" in the headline.
    """
    fields = [
        (normalise(title), 2.5),
        (normalise(claim), 2.0),
        (normalise(image_text), 1.5),
        (normalise(body)[:6000], 1.0),
    ]

    score = 0.0
    hits: list[str] = []
    seen: set[str] = set()

    for text, field_weight in fields:
        if not text:
            continue
        padded = f" {text} "
        for term, term_weight in LEXICON.items():
            if f" {term} " in padded or (len(term) > 6 and term in text):
                score += term_weight * field_weight
                if term not in seen:
                    hits.append(term)
                    seen.add(term)

    neg_score = 0.0
    neg_hits: list[str] = []
    head = f" {normalise(title)} {normalise(claim)} "
    for term, w in NEGATIVE_LEXICON.items():
        if f" {term} " in head:
            neg_score += w * 2.0
            neg_hits.append(term)

    final = score - neg_score
    return FinanceVerdict(
        is_finance=final >= threshold,
        score=round(final, 2),
        hits=hits[:25],
        negative_hits=neg_hits,
    )


def score_record(record, threshold: float = 4.0) -> FinanceVerdict:
    """Convenience wrapper for a FactCheckRecord."""
    image_text = " ".join(
        filter(None, [(a.alt_text or "") + " " + (a.caption or "") + " " + (a.ocr_text or "")
                      for a in record.images])
    )
    return score_text(
        title=record.title,
        claim=record.claim_text,
        body=record.body_text,
        image_text=image_text,
        threshold=threshold,
    )
