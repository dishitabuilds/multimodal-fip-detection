"""Inspect promotional wording without equating persuasion with factual falsity.

The rules are a limited deterministic English-language baseline, not a trained model or
an independent fact checker. Evidence is an attributed human review supplied by
the caller. No URL is fetched and no stock-price outcome enters the assessment.
Offsets always refer to the exact original text or caption supplied by the user.
"""
from __future__ import annotations

from datetime import date
import ipaddress
import re
from urllib.parse import urlsplit

AUDIT_VERSION = "promotion-framing-v1"
MAX_FIELD_CHARS = 50_000
MAX_EVIDENCE = 100
MAX_REVIEW_CHARS = 5_000
_FLAGS = re.IGNORECASE
_FINANCE = re.compile(
    r"\b(?:stocks?|shares?|invest(?:ment|ments|ing|or|ors)?|trad(?:e|ing)|"
    r"returns?|profits?|portfolio|capital|money|sebi|nse|bse|mutual fund)\b", _FLAGS)
_BOUNDARY = re.compile(r"[!?;\n]|(?<!\d)\.(?!\d)")
_EDUCATIONAL = re.compile(
    r"\b(?:beware|avoid|scam warning|warning|red flags?|fraud awareness|"
    r"misleading|unrealistic|false promises?|do not trust|don't trust|never trust|"
    r"cannot promise|can't promise)\b", _FLAGS)
_NEGATED_PREFIX = re.compile(
    r"(?:\b(?:no|not|never|without)\s+(?:(?:a|any|such|fully|ever|all)\s+){0,3}|"
    r"\b(?:cannot|can't|can not|do not|don't|does not|doesn't|will not|won't)\s+"
    r"(?:(?:promise|offer|guarantee|assure|provide|deliver|mean|imply)\s+)?"
    r"(?:(?:a|any|such|fully|ever)\s+){0,2})$", _FLAGS)
_NEGATED_SUFFIX = re.compile(
    r"^\s*(?:(?:is|are|was|were|can|should|will)(?:\s+be)?\s+not\s+"
    r"(?:possible|promised|guaranteed|assured|available|offered|real|true|provided|achievable)\b|"
    r"(?:is|are)\s+(?:a\s+)?(?:myth|impossible|unrealistic|misleading)\b)", _FLAGS)
_HEDGE_PREFIX = re.compile(
    r"\b(?:may|might|could|possibly|potentially|potential to|chance to)\s+"
    r"(?:(?:eventually|perhaps|potentially|possibly|even|also)\s+){0,2}$", _FLAGS)
_CONDITION_PREFIX = re.compile(
    r"^\s*(?:only\s+)?(?:if|assuming|depending on|subject to)\b", _FLAGS)
_CONDITION_SUFFIX = re.compile(
    r"^\s*,?\s*(?:only\s+)?(?:if|assuming|depending on|subject to)\b", _FLAGS)
_PURCHASE_CONDITION = re.compile(
    r"^\s*,?\s*(?:only\s+)?(?:if|assuming)\s+(?:(?:you|we|investors?|buyers?)\s+)?"
    r"(?:buy|invest|join|subscribe|purchase|deposit|pay)\b", _FLAGS)
_BOILERPLATE_CONDITION = re.compile(
    r"^\s*,?\s*subject to\s+(?:market risks?|terms(?: and conditions)?|conditions apply)\b",
    _FLAGS)
_DIRECT_CAUTION_SUFFIX = re.compile(
    r"^\s*(?:[,\-:\u2013\u2014]\s*(?:this|that)\s+)?"
    r"(?:is|are|was|were|can be|may be|would be)\s+"
    r"(?:(?:a|an|often|usually|potentially)\s+){0,2}"
    r"(?:misleading|unrealistic|false promises?|red flags?|warnings?)\b", _FLAGS)
_CONTRAST = re.compile(r"\b(?:but|however|yet|nevertheless)\b", _FLAGS)
_DISMISSAL = re.compile(
    r"\b(?:ignore|dismiss|disregard|forget)\s+(?:(?:the|these|any|all|such)\s+){0,2}$",
    _FLAGS)
_DO_NOT_DISMISS = re.compile(
    r"\b(?:do not|don't|never|cannot|can't)\s+(?:ignore|dismiss|disregard|forget)\s+"
    r"(?:(?:the|these|any|all|such)\s+){0,2}$", _FLAGS)

# (label, rule identifier, pattern, explanation). Overlapping matches for the
# same label are collapsed; the first (more specific) rule takes precedence.
_RULES = [
    ("return_promise", "return_guarantee",
     r"\b(?:guaranteed|assured|risk[-\s]?free)\s+(?:\d+(?:\.\d+)?\s*%\s+)?"
     r"(?:(?:double|triple)\s+)?(?:(?:daily|weekly|monthly|annual)\s+)?"
     r"(?:(?:stock|investment|trading)\s+)?"
     r"(?:profits?|returns?|income|gains?)\b(?:\s+(?:of\s+)?\d+(?:\.\d+)?\s*%)?",
     "The wording presents an investment return as guaranteed or free of risk. "
     "This is a presentation cue, not a finding that a historical fact is false."),
    ("return_promise", "return_guarantee",
     r"\b(?:returns?|profits?|gains?|income)\s+(?:(?:are|is|will be)\s+)?"
     r"(?:\d+(?:\.\d+)?\s*%\s+)?(?:guaranteed|assured)\b",
     "The wording assures an investment outcome. A past result does not establish a future return."),
    ("return_promise", "return_guarantee",
     r"\b(?:we|i)\s+guarantee\s+(?:(?:you|a|an)\s+){0,2}"
     r"(?:\d+(?:\.\d+)?\s*%\s+)?(?:profits?|returns?|income|gains?)\b",
     "The speaker explicitly guarantees an investment outcome; review the promise and its context."),
    ("return_promise", "certain_future_return",
     r"\b(?:(?:stock|share)\s+price|stock|shares?|price|investment|portfolio|capital|money)\s+"
     r"(?:will|is going to|is set to)\s+(?:(?:certainly|definitely)\s+)?"
     r"(?:double|triple|multiply|rise|soar|surge|increase|go up|skyrocket)\b"
     r"(?:\s+(?:by\s+|to\s+)?\d+(?:\.\d+)?\s*%)?",
     "The wording presents a future investment outcome as certain; the rules did not identify "
     "an uncertainty qualifier attached to this claim. "
     "The audit does not predict whether that outcome will occur."),
    ("return_promise", "certain_future_return",
     r"\b(?:double|triple)\s+(?:your\s+)?(?:money|investment|capital)\b",
     "The wording promotes multiplication of invested money. Check the full claim and assumptions."),
    ("return_promise", "certain_future_return",
     r"\b(?:earn|make|get|receive)\s+\d+(?:\.\d+)?\s*%\s+"
     r"(?:(?:every|each|per)\s+(?:day|week|month|year)|daily|weekly|monthly|annually)\b",
     "The wording promises a periodic percentage return; no future return is verified by this audit."),
    ("urgency", "scarcity_or_immediate_action",
     r"\b(?:last chance|limited[- ]time(?: offer)?|only\s+(?:\d+|a few)\s+"
     r"(?:spots|slots|seats)\s+left|before\s+it(?:'s| is)\s+too\s+late|"
     r"don't\s+miss\s+out|do\s+not\s+miss\s+out|act\s+now|buy\s+now|"
     r"invest\s+now|today\s+only)\b",
     "A time or scarcity cue encourages quick action. Such wording alone does not establish deception."),
    ("social_proof", "popularity_as_persuasion",
     r"\b(?:(?:thousands|millions|everyone|\d[\d,]*\+?)\s+(?:of\s+)?"
     r"(?:investors?|traders?|people|members|followers)\s+(?:(?:have|are|already)\s+){0,2}"
     r"(?:joined|invested|buying|earning|profiting)|everyone\s+is\s+(?:buying|investing)|"
     r"join\s+(?:thousands|millions|\d[\d,]*\+?)\s+(?:of\s+)?"
     r"(?:investors?|traders?|members))\b",
     "Popularity or other investors' participation is used as persuasion; it is not evidence of returns."),
    ("authority_cue", "credentials_or_endorsement",
     r"\b(?:SEBI[- ]registered|SEBI[- ]approved|NISM[- ]certified|"
     r"(?:celebrity|expert|famous investor)[- ](?:endorsed|approved|recommended)|"
     r"(?:endorsed|recommended)\s+by\s+(?:an?\s+)?(?:celebrity|expert|famous investor))\b",
     "Credentials or endorsement are invoked. The audit has not verified the identity, "
     "registration, endorsement or any performance assurance."),
]
_COMPILED_RULES = [(label, identifier, re.compile(pattern, _FLAGS), reason)
                   for label, identifier, pattern, reason in _RULES]
_RISK_DISCLOSURE = re.compile(
    r"\b(?:market risks?|capital at risk|(?:may|can|could) lose|loss(?:es)? possible|"
    r"possible loss(?:es)?|risk of loss|"
    r"no guarantee|not guaranteed|past performance.{0,70}(?:future|guarantee)|"
    r"returns?.{0,20}not guaranteed)\b", _FLAGS)
_SPONSORSHIP = re.compile(
    r"(?:\b(?:sponsored|paid promotion|paid partnership|affiliate link|affiliate commission|"
    r"not sponsored|not paid)\b|(?<!\w)#ad\b)", _FLAGS)


def _text(value: object, name: str, limit: int, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    if len(value) > limit:
        raise ValueError(f"{name} must contain at most {limit:,} characters.")
    if nonempty and not value.strip():
        raise ValueError(f"{name} must not be blank.")
    return value


def _date(value: object, name: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must use the ISO date format YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid calendar date in YYYY-MM-DD format.") from exc


def _public_url(value: object, name: str) -> str:
    url = _text(value, name, 2_048, nonempty=True)
    if any(character.isspace() or ord(character) < 32 for character in url):
        raise ValueError(f"{name} must be a public HTTP or HTTPS source URL without whitespace.")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        parsed.port  # Reject malformed port syntax even though no request is made.
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid public HTTP or HTTPS source URL.") from exc
    if (parsed.scheme not in {"https", "http"} or not host or parsed.username or parsed.password
            or host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost")):
        raise ValueError(f"{name} must be a public HTTP or HTTPS source URL without credentials.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Domain syntax check only: neither DNS resolution nor content verification.
        if "." not in host or not re.fullmatch(r"[A-Za-z0-9.-]+", host):
            raise ValueError(f"{name} must use a public domain or globally routable address.")
        if any(not part or part.startswith("-") or part.endswith("-")
               for part in host.split(".")):
            raise ValueError(f"{name} contains an invalid hostname.")
    else:
        if not address.is_global:
            raise ValueError(f"{name} must not use a private or loopback address.")
    return url


def _sentence(text: str, start: int, end: int) -> tuple[str, str, str]:
    left = 0
    for boundary in _BOUNDARY.finditer(text, 0, start):
        left = boundary.end()
    right_match = _BOUNDARY.search(text, end)
    right = right_match.start() if right_match else len(text)
    return text[left:right], text[left:start], text[end:right]


def _suppressed(text: str, start: int, end: int, rule: str) -> bool:
    sentence, prefix, suffix = _sentence(text, start, end)
    boundary = _BOUNDARY.search(text, end)
    if (boundary and boundary.group() == "?"
            and re.match(r"\s*(?:are|is|can|could|should|would|why|what|do|does|how)\b",
                         sentence, _FLAGS)):
        return True
    # A warning in an earlier contrast clause cannot cancel a later promise:
    # "Investments carry a warning, but this stock will double."
    local_start, local_end = len(prefix), len(prefix) + end - start
    left, right = 0, len(sentence)
    for boundary in _CONTRAST.finditer(sentence):
        if boundary.end() <= local_start:
            left = boundary.end()
        elif boundary.start() >= local_end:
            right = boundary.start()
            break
    prefix, suffix = sentence[left:local_start], sentence[local_end:right]
    sentence = sentence[left:right]
    # A caution must introduce the quoted claim or describe that claim directly.
    # Merely appending "read the risk warning" cannot erase an earlier promise.
    for caution in _EDUCATIONAL.finditer(prefix):
        before = prefix[:caution.start()]
        if (_DISMISSAL.search(before) and not _DO_NOT_DISMISS.search(before)):
            continue
        if (caution.group().lower() in {"misleading", "unrealistic"}
                and re.search(r"\b(?:not|isn't|aren't)\s+(?:really\s+)?$", before, _FLAGS)):
            continue
        return True
    if _DIRECT_CAUTION_SUFFIX.search(suffix):
        return True
    if _NEGATED_PREFIX.search(prefix) or _NEGATED_SUFFIX.search(suffix):
        return True
    if rule == "certain_future_return":
        if _HEDGE_PREFIX.search(prefix):
            return True
        # Only conditions attached to this claim qualify it. A separate phrase
        # such as "although you could lose money" does not cancel "will double".
        for fragment, pattern in ((prefix, _CONDITION_PREFIX), (suffix, _CONDITION_SUFFIX)):
            if (pattern.search(fragment) and not _PURCHASE_CONDITION.search(fragment)
                    and not _BOILERPLATE_CONDITION.search(fragment)):
                return True
    return False


def _claims(evidence: list[dict] | None, fields: dict[str, str],
            as_of: str | None) -> list[dict]:
    if evidence is None:
        return []
    if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE:
        raise ValueError(f"evidence must be a list of at most {MAX_EVIDENCE} review records.")
    if not evidence:
        return []
    if as_of is None:
        raise ValueError("Provide as_of when supplying evidence so later information is excluded.")
    cutoff = _date(as_of, "as_of")
    result = []
    for index, item in enumerate(evidence):
        name = f"evidence[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{name} must be an object containing an attributed claim review.")
        allowed = {"claim_text", "field", "start", "end", "status", "source_url",
                   "source_title", "published_at", "reviewed_by", "rationale", "context"}
        if set(item) - allowed:
            raise ValueError(f"{name} contains unsupported fields; use the documented evidence schema.")
        field = item.get("field", "text")
        if not isinstance(field, str) or field not in fields:
            raise ValueError(f"{name}.field must be 'text' or 'caption'.")
        claim = _text(item.get("claim_text"), f"{name}.claim_text", MAX_REVIEW_CHARS,
                      nonempty=True)
        start = fields[field].find(claim)
        if start < 0:
            raise ValueError(f"{name}.claim_text must be an exact quote present in {field}.")
        # Optional offsets resolve repeated identical claims. They must still match.
        if "start" in item or "end" in item:
            begin, end = item.get("start"), item.get("end")
            if (type(begin) is not int or type(end) is not int or begin < 0
                    or end <= begin or end > len(fields[field]) or fields[field][begin:end] != claim):
                raise ValueError(f"{name}.start and .end must identify the exact claim quote.")
            start = begin
        elif fields[field].find(claim, start + 1) >= 0:
            raise ValueError(f"{name}.claim_text occurs more than once; supply exact start and end.")
        status = item.get("status")
        if not isinstance(status, str) or status not in {
                "supported", "contradicted", "insufficient_evidence"}:
            raise ValueError(f"{name}.status must be supported, contradicted or insufficient_evidence.")
        published = _date(item.get("published_at"), f"{name}.published_at")
        if published > cutoff:
            raise ValueError(f"{name}.published_at is after as_of; exclude hindsight evidence.")
        row = {"claim_text": claim, "field": field, "start": start,
               "end": start + len(claim), "status": status,
               "source_url": _public_url(item.get("source_url"), f"{name}.source_url"),
               "published_at": published.isoformat()}
        for key in ("source_title", "reviewed_by", "rationale"):
            row[key] = _text(item.get(key), f"{name}.{key}", MAX_REVIEW_CHARS, nonempty=True)
        row["context"] = _text(item.get("context", ""), f"{name}.context", MAX_REVIEW_CHARS)
        row["evidence_origin"] = "reviewer_supplied_not_independently_verified"
        result.append(row)
    return result


def audit_promotion(text: str, caption: str = "", *, as_of: str | None = None,
                    evidence: list[dict] | None = None) -> dict:
    """Return located wording cues and separately attributed claim reviews.

    ``as_of`` is the promotion's assessment date, not today's date. Evidence
    published after it is rejected. The caller remains responsible for matching
    the company, reporting period, measure and source to the quoted claim.
    """
    fields = {"text": _text(text, "text", MAX_FIELD_CHARS),
              "caption": _text(caption, "caption", MAX_FIELD_CHARS)}
    if not any(value.strip() for value in fields.values()):
        raise ValueError("Provide promotion text or its original caption before running the audit.")
    if as_of is not None:
        _date(as_of, "as_of")
    claims = _claims(evidence, fields, as_of)
    findings = []
    combined = "\n".join(fields.values())
    financial = bool(_FINANCE.search(combined))
    for field, value in fields.items():
        for label, identifier, pattern, reason in _COMPILED_RULES:
            if not financial:
                continue
            for match in pattern.finditer(value):
                start, end = match.span()
                if _suppressed(value, start, end, identifier):
                    continue
                if any(row["field"] == field and row["label"] == label
                       and start < row["end"] and end > row["start"] for row in findings):
                    continue
                findings.append({"label": label, "rule_id": identifier, "field": field,
                                 "start": start, "end": end, "quote": value[start:end],
                                 "reason": reason})
    findings.sort(key=lambda row: (0 if row["field"] == "text" else 1,
                                   row["start"], row["end"], row["label"]))
    observations = []
    if claims:
        observations.append(
            f"{len(claims)} claim review(s) were supplied by the named reviewer(s). "
            "Source URLs and verdicts were not fetched or independently verified. "
            "All other factual claims remain unchecked.")
    else:
        observations.append("Factual claims have not been checked: no attributed source evidence "
                            "was supplied. Unchecked does not mean false or supported.")
    observations.append(
        "Risk-related wording was detected in the supplied text; its adequacy and visibility "
        "have not been assessed." if _RISK_DISCLOSURE.search(combined) else
        "No risk disclosure wording was detected in the supplied text. A disclosure may exist "
        "elsewhere, and this observation does not establish omission or deception.")
    observations.append(
        "Sponsorship or affiliation wording was detected in the supplied text; payment and "
        "conflicts have not been independently verified." if _SPONSORSHIP.search(combined) else
        "No sponsorship or affiliation disclosure wording was detected in the supplied text. "
        "This does not establish that payment or a conflict exists.")
    if not findings:
        observations.append("No configured English wording cues were detected. This is not a "
                            "finding that the promotion is balanced, authentic or suitable.")
    return {"version": AUDIT_VERSION, "method": "rules_and_reviewer_evidence", "as_of": as_of,
            "findings": findings, "claims": claims, "observations": observations,
            "limitations": [
                "English text rules are an unvalidated baseline; subtle persuasion, other "
                "languages, OCR errors, sarcasm and complex negation may be missed.",
                "Only supplied text and caption are inspected. Charts, cropped axes, visual "
                "prominence, audio, video, account identity and coordinated activity are not assessed.",
                "Selective presentation requires relevant contemporaneous context supplied and "
                "reviewed by a person; missing input is not proof of cherry-picking.",
                "Reviewers must match each claim to the correct company, reporting period, "
                "measure and source. Source matching and reviewer verdicts are not automated.",
                "Wording cues can occur in legitimate advertising. There is no fraud verdict, "
                "aggregate risk score, investment recommendation or future-price assessment.",
            ]}
