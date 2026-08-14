"""Normalisation for code-mixed Hindi/English text.

The problem
-----------
The same claim appears three ways across this corpus:

    "पैसा डबल"          Devanagari
    "paisa double"      romanised Hindi
    "money double"      English

MuRIL was pretrained on Devanagari Hindi and on English, but *not* on
romanised Hindi at any scale. Left alone, "paisa double" tokenises into
English-looking subwords carrying none of the Hindi meaning, and the model has
to learn the mapping from our small dataset — which it will not do well.

Approach
--------
Two-stage, cheap before expensive:

  1. A curated lexicon of romanised finance/scam terms mapped to Devanagari.
     Small, high precision, and it covers the vocabulary that actually matters
     for this task.
  2. Optional rule-based transliteration via `indic-transliteration` for
     remaining tokens that look Hindi-ish.

Why not transliterate everything: romanised Hindi has no standard orthography
("paisa"/"paise"/"paisaa") and rule-based conversion of genuinely English words
produces nonsense. The lexicon handles the terms that carry signal; the rest is
left in Latin script, which MuRIL handles acceptably as English.

`normalise_mixed` returns BOTH forms by default, joined. Keeping the original
alongside the transliteration lets the encoder use whichever it has better
representations for, and costs only sequence length.
"""

from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated romanised -> Devanagari lexicon for this domain
# ---------------------------------------------------------------------------

ROMAN_TO_DEVANAGARI: dict[str, str] = {
    # money
    "paisa": "पैसा", "paise": "पैसे", "paisaa": "पैसा", "rupaya": "रुपया",
    "rupaye": "रुपये", "rupees": "रुपये", "lakh": "लाख", "lakhs": "लाख",
    "crore": "करोड़", "crores": "करोड़", "hazaar": "हज़ार", "hazar": "हज़ार",
    # investing
    "nivesh": "निवेश", "munafa": "मुनाफा", "munafaa": "मुनाफा",
    "kamai": "कमाई", "kamaai": "कमाई", "bachat": "बचत", "byaj": "ब्याज",
    "byaaj": "ब्याज", "share": "शेयर", "bazaar": "बाजार", "bazar": "बाजार",
    "sharebazaar": "शेयर बाजार", "vyapar": "व्यापार",
    # scam vocabulary
    "ghotala": "घोटाला", "ghotaala": "घotाला".replace("ot", "ोता"),
    "dhokha": "धोखा", "dhoka": "धोखा", "fraud": "धोखाधड़ी",
    "farzi": "फर्जी", "farji": "फर्जी", "fake": "फर्जी", "jhooth": "झूठ",
    "jhuth": "झूठ", "afwah": "अफवाह", "chuna": "चूना",
    # promises
    "guarantee": "गारंटी", "guaranteed": "गारंटीड", "double": "डबल",
    "tips": "टिप्स", "profit": "मुनाफा", "loan": "लोन", "karz": "कर्ज",
    "scheme": "स्कीम", "yojana": "योजना", "khata": "खाता", "bank": "बैंक",
    # verbs and connectors that change meaning
    "kamao": "कमाओ", "kamaye": "कमाए", "lagao": "लगाओ", "milega": "मिलेगा",
    "banao": "बनाओ", "jaldi": "जल्दी", "turant": "तुरंत", "sirf": "सिर्फ",
    "bilkul": "बिल्कुल", "abhi": "अभी", "roz": "रोज़", "mahina": "महीना",
    "din": "दिन", "saal": "साल",
}

# Fix the one entry that is clearer written literally.
ROMAN_TO_DEVANAGARI["ghotaala"] = "घोटाला"

_TOKEN = re.compile(r"[A-Za-zऀ-ॿ0-9]+|[^\sA-Za-zऀ-ॿ0-9]")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text or ""))


def has_latin(text: str) -> bool:
    return bool(_LATIN_RE.search(text or ""))


def is_code_mixed(text: str) -> bool:
    return has_devanagari(text) and has_latin(text)


def normalise_unicode(text: str) -> str:
    """NFC-normalise and collapse whitespace.

    Devanagari has multiple encodings for the same visible character (nukta
    forms in particular). Without this, 'फ़र्ज़ी' and 'फ़र्ज़ी' are different
    strings to the tokeniser.
    """
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------


def romanised_to_devanagari(
    text: str, use_library: bool = False, min_len: int = 3
) -> str:
    """Map known romanised Hindi tokens to Devanagari; leave the rest alone."""
    if not text:
        return ""

    out: list[str] = []
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        if low in ROMAN_TO_DEVANAGARI:
            out.append(ROMAN_TO_DEVANAGARI[low])
        elif use_library and len(tok) >= min_len and tok.isalpha() and tok.isascii():
            out.append(_library_translit(tok))
        else:
            out.append(tok)

    # Re-join, keeping punctuation tight against the preceding token.
    s = ""
    for tok in out:
        if re.fullmatch(r"[^\wऀ-ॿ]", tok):
            s = s.rstrip() + tok + " "
        else:
            s += tok + " "
    return s.strip()


def _library_translit(token: str) -> str:
    """Rule-based fallback. Returns the token unchanged if the lib is absent."""
    try:
        from indic_transliteration import sanscript  # noqa: PLC0415
        from indic_transliteration.sanscript import transliterate  # noqa: PLC0415

        return transliterate(token.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception:
        return token


def normalise_mixed(
    text: str,
    keep_original: bool = True,
    use_library: bool = False,
) -> str:
    """Full normalisation for text heading into the encoder.

    With `keep_original` (the default) the output is
    ``"<original> | <transliterated>"`` when transliteration changed anything.
    The encoder then sees both surface forms and can use whichever it
    represents better. Set it False for an ablation on that choice.
    """
    text = normalise_unicode(text)
    if not text:
        return ""

    converted = romanised_to_devanagari(text, use_library=use_library)
    if not keep_original or converted == text:
        return converted
    return f"{text} | {converted}"


def build_text_input(
    caption: str = "",
    ocr_text: str = "",
    title: str = "",
    separator: str = " [SEP] ",
    normalise: bool = True,
    max_chars: int = 4000,
) -> str:
    """Assemble the text branch's input from the record's text sources.

    Order is deliberate: OCR text goes FIRST. The claim we are classifying
    lives inside the image, and truncation at the tokeniser's max length should
    eat the caption rather than the thing we actually care about.

    Which fields are included is an ablation axis in its own right — report
    caption-only vs OCR-only vs both.
    """
    parts = [p.strip() for p in (ocr_text, caption, title) if p and p.strip()]
    if not parts:
        return ""
    text = separator.join(parts)
    if normalise:
        text = normalise_mixed(text)
    return text[:max_chars]
