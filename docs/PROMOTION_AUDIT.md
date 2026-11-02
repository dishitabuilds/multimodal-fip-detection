# Promotion framing and claim review

The **Promotion Audit** page addresses promotions that can be misleading even
when some historical facts are correct. It works without the binary fake/real
model. Open it from Streamlit's page navigation after starting the application.

The page provides separate outputs:

- English wording rules quote selected return promises, urgency, social proof
  and authority cues, with exact field offsets and reasons.
- A reviewer evidence ledger records supported, contradicted or insufficiently
  evidenced claims and attributed material context.
- An optional, separately trained multi-label SVM predicts framing annotations.
  No framing research model is currently supplied. Follow [FRAMING_LABELS.md](FRAMING_LABELS.md)
  to prepare reviewed data and train one.

These outputs are independent of the existing binary classifier. Wording cues
alone do not establish deception. A source link is not fetched or independently
checked by this tool. Missing evidence means **facts not checked**. A later price
rise or fall does not establish the honesty of the earlier promotion, and stock
suitability is not assessed.

## Try the audit

Paste the exact transcript and original caption, or upload an image and choose
**Read image text**. Review OCR errors before auditing. The rule baseline is
English; OCR support for other scripts does not establish framing accuracy in
those languages. Examples in the page are explicitly hypothetical.

From PowerShell in the project root:

```powershell
.\.venv\Scripts\python.exe scripts/13_audit_promotion.py --text "Revenue grew 40%. Guaranteed double returns. Buy now."
```

The revenue figure remains unchecked, while the wording of the promise can be
flagged. Ordinary uncertainty such as “the stock could rise, but losses are
possible” should be treated differently from an assured return. Educational
warnings and negation are covered by regression cases, though the limited rules
can still misread ambiguous language.

The UI clears a result after changes to any audited input, date, evidence or
model selection. JSON download preserves the explanation and evidence ledger.
No count of cues is converted into a fraud probability or risk score.

## Record claim evidence

Supply the promotion date or another explicit information cutoff as `--as-of`
when adding evidence. The tool rejects sources published after this date.
Reviewers remain responsible for verifying publication availability and matching
the issuer, metric, units, reporting period, comparison base and accounting basis.

An evidence file must contain a JSON list. The following is a schema illustration
using hypothetical content and a reserved example URL. Replace it with an actual
review and source before using it for a real assessment:

```json
[
  {
    "claim_text": "Revenue grew 40%",
    "field": "text",
    "status": "supported",
    "source_url": "https://example.org/illustrative-filing",
    "source_title": "Hypothetical annual filing",
    "published_at": "2026-04-30",
    "reviewed_by": "Reviewer name",
    "rationale": "The hypothetical same-period revenue comparison matches the quoted figure.",
    "context": "Explain separately verified material context here, if applicable."
  }
]
```

`claim_text` must occur exactly in the selected `field`, either `text` or
`caption`. If the quote repeats, also supply its zero-based `start` and exclusive
`end` character offsets. Status is one of `supported`, `contradicted`, `insufficient_evidence`.
The source title, URL, date, reviewer and rationale are required. `context` is
optional. A historical claim assessed as supported does not support an unrelated
promise about future stock returns. Evidence validates only its quoted claim;
other claims remain unchecked.

```powershell
.\.venv\Scripts\python.exe scripts/13_audit_promotion.py --text-file promotion.txt --as-of 2026-10-30 --evidence evidence.json
```

The app displays these as **reviewer-supplied evidence**. Validation checks format,
span correspondence and the declared date; it cannot establish that the reviewer
is correct or that a URL supports the supplied rationale. No internet requests
are made during auditing. Do not use a later corrected filing as information that
was available before a promotion's publication.

## Use a trained framing model

Only use locally generated joblib artifacts; deserialization uses pickle.
Once a reviewed-data experiment exists, select its artifact on the audit page
or supply it to the CLI:

```powershell
.\.venv\Scripts\python.exe scripts/13_audit_promotion.py --text-file promotion.txt --framing-artifact data/processed/framing/research/YOUR_RUN/models/framing_linear_svm.joblib
```

Learned label margins stay separate from the rule quotations and claim evidence.
They are signed decision values, not calibrated probabilities. In particular, a
text prediction of `selective_context` is a review prompt; confirming an omitted
material fact requires external evidence.

## Scope and next data work

The tool cannot inspect chart axes or tiny disclosures visually, verify influencer
identity or registration, prove paid sponsorship, retrieve financial filings,
assess stock value, or establish coordinated market manipulation. Disclosure
observations concern only the supplied text. No detected disclosure does not prove
that a complete post or video lacked one, and an authority cue does not verify an
endorsement.

For the next evaluation, include balanced advertisements, optimistic opinions,
warnings, truthful but selective presentations and fabricated claims. Review
campaign grouping and semantic duplicates as well as exact matches. Measure false
alarms on legitimate content and per-label recall. Use an independent second
reviewer for context judgments and resolve disagreements before fixing the test
set.

The [SEBI Master Circular for Research Analysts](https://www.sebi.gov.in/sebi_data/attachdocs/feb-2026/1770375507051.pdf#page=35)
addresses misleading implications, omissions and return assurances in its
advertisement code. It motivates these review categories within its scope; this
app does not decide legal compliance. Research on
[pump-and-dump detection](https://arxiv.org/abs/2301.11403) also discusses accidental
coincidences between posts and market activity. Future coordination analysis needs
campaign/account timelines and market data, with separate evaluation.
