# Learning promotion framing from reviewed examples

This extension asks which presentation cues appear in a promotion. It is a
multi-label task: one promotion can contain a return promise, urgency and social
proof together. Its labels are separate from the existing fake/real classifier,
from claim correctness, from stock quality and from future price movements.
Truthful facts do not rule out misleading framing, and a bullish opinion does
not by itself establish a misleading promotion.

The runnable training path uses word and character TF-IDF followed by
`OneVsRestClassifier(LinearSVC(class_weight="balanced"))`. Vocabulary and IDF are
fitted on training text only. C is chosen by validation macro-F1, with the first
configured value winning a tie. Only the selected classifier is evaluated on
test; validation and test data never refit preprocessing or the classifier.

No reviewed framing dataset or trained framing artifact is supplied yet. The
existing binary pilot cannot provide these labels. Automated audit flags cannot
provide them either. Synthetic test fixtures check the implementation and do
not measure detection accuracy.

## Annotation meanings

- `return_promise`: an affirmative guarantee or unqualified promise of future
  investment profit or a specified return. A truthful historical return alone
  does not receive this label. Neither does an educational warning quoting a
  promise, or a conditional forecast that clearly acknowledges uncertainty.
- `urgency`: time pressure or fear of missing an investment opportunity. Record
  the pressure phrase, rather than treating every date or reporting deadline as
  urgency. A dated announcement can be an ordinary factual statement.
- `social_proof`: popularity, followers, investors or others' success used to
  persuade. It is a cue, not a finding that the popularity claim is false.
- `authority_cue`: a person, institution, credential or celebrity invoked to
  persuade. It can be accurate and legitimate. Annotate its use without inferring
  impersonation, a financial relationship or misconduct.
- `selective_context`: a reviewer finds that the presentation selectively omits
  or compares material information in a way that changes its interpretation.
  Require an exact promotion span, a cited source available when the promotion
  appeared, and a written explanation of the omitted context and its materiality.
  A model prediction for this label remains a review lead; text alone cannot
  establish which context was omitted.

Labels may overlap. An empty list means the reviewer checked the whole supplied
promotion for these five labels and found none; it does not certify the promotion
as authentic or safe to buy. Include balanced advertisements, ordinary bullish
opinions, educational warnings and accurate historical reporting among reviewed
negative examples. Include truthful but misleading promotions as well as false
claims, so factual status does not become a shortcut for the framing task.

Use at least two reviewers for a representative subset, record disagreements
and adjudicate them before approving records. Include different companies,
languages, channels, campaign styles and market periods. Group related
creatives, copied messages, promoter campaigns and templates together before
splitting. Use later campaigns or periods for an additional external evaluation
when enough data are available. Do not relabel a message because its stock later
rose or fell.

## Reviewed JSONL contract

Store one JSON object per line in `data/reviewed/framing.jsonl`. The following is
a format example only, not a reviewed training example:

```json
{"schema_version":1,"uid":"promotion-001","group_id":"campaign-001","split":"train","text":"Guaranteed profit. Buy today.","caption":"","as_of":"2026-10-20","labels":["return_promise","urgency"],"review":{"method":"human","status":"approved","reviewer_name":"Fill reviewer name","reviewed_at":"2026-10-21"},"evidence":[{"label":"return_promise","field":"text","start":0,"end":17,"quote":"Guaranteed profit","rationale":"An affirmative promise of investment profit."},{"label":"urgency","field":"text","start":19,"end":28,"quote":"Buy today","rationale":"The promotion asks for an immediate investment decision."}]}
```

Required fields and validation rules:

- `schema_version` is 1; `uid` is unique; `group_id` identifies the campaign or
  duplicate family; `split` is `train`, `val` or `test`.
- `text` contains the exact OCR transcript; `caption` contains the original
  promotional caption. Either can be empty, but their combination cannot be
  empty. Archive headlines, debunking text, reviewer comments, source filings and
  verdicts never enter model features. Each field has a 50,000-character limit.
- `as_of` is the promotion date used as the information cutoff, in `YYYY-MM-DD`
  format. `review.reviewed_at` records the annotation date and cannot precede it.
- Every row, including a row with `labels: []`, declares `review.method: "human"`,
  `review.status: "approved"` and a nonempty `reviewer_name`. These are attributed
  review declarations; software cannot authenticate a person's identity or confirm
  that they actually conducted a review.
- `labels` contains unique known labels. Every positive label has at least one
  evidence item with `label`, `field` (`text` or `caption`), zero-based `start`,
  exclusive `end`, exact `quote` and a nonempty reviewer `rationale`.
- Spans use Python/Unicode character offsets, not byte offsets. The stored quote
  must equal `row[field][start:end]`. Correct OCR carefully before annotating and
  update spans whenever wording changes.
- Every `selective_context` evidence item additionally requires `source_url`
  (HTTP/HTTPS), `source_title`, `published_at` and `material_context`. The source
  publication date must be on or before `as_of`. The reviewer must actually check
  the source and explain why the omitted material changes interpretation; the
  trainer only validates these fields and dates, and never fetches the URL.

For example, a positive revenue figure might be accurately quoted while a
contemporaneous filing shows that it includes a one-off item. Cite the actual
filing and the relevant reporting period, and explain the comparison. Do not
assume every revenue increase requires every other metric to be displayed, and
do not invent negative context because a stock later performed poorly.

The trainer rejects groups spanning splits and normalized exact promotion text
reused across splits. It also checks reused OCR text when captions differ.
Normalization uses Unicode NFKC, case folding and collapsed whitespace. These
checks do not detect paraphrases, semantically similar images or every shared
campaign; a person still needs to review the split assignments.

Every requested label needs both positive and negative examples in every split.
This is a structural minimum, not evidence that a small sample is sufficient.
Collect representative data before reporting model accuracy. `--labels` can
select a reviewed subset when other label categories are not ready.

## Train and inspect

From the main project folder, using the existing environment with the lab extras:

```powershell
.\.venv\Scripts\python.exe scripts/12_train_framing.py --records data/reviewed/framing.jsonl

# A subset can be trained only when its positive and negative reviews are ready.
.\.venv\Scripts\python.exe scripts/12_train_framing.py --records data/reviewed/framing.jsonl --labels return_promise urgency --C 0.1 1 10
```

Missing annotations stop training with a link to these instructions. The command
does not generate labels, approvals or a demonstration model. Research outputs
are saved under `data/processed/framing/research/<experiment-id>/`:

- `models/framing_linear_svm.joblib`: fitted preprocessing, independent SVM
  heads, label ordering, selected C, annotation contract, input/pipeline versions,
  scikit-learn version and input-file hash.
- `experiment.json`: split sizes, campaign counts, review attribution, C
  comparisons on validation, selected C, test metrics and artifact reload check.
- `evaluations/test.json`: per-label precision, recall, F1, positive/negative
  support and TP/FP/FN/TN, plus macro-F1 and exact match across all labels.
- `predictions.json`: held-out UIDs, true/predicted label lists and signed margins.

Inspect false positives on legitimate material and false negatives on selective
promotions separately. Report per-label results and sample sizes; one overall
accuracy number hides the relevant differences. A high margin is not a
probability and should not become a fraud-risk score.

Only load locally generated joblib files; loading an untrusted pickle can execute
code. Use the same scikit-learn version as training. From Python:

```python
from fipd.training.framing import load_framing_artifact, predict_framing

artifact = load_framing_artifact("path/to/models/framing_linear_svm.joblib")
result = predict_framing(artifact, text="Exact promotion wording", caption="")
# result contains labels, signed per-label margins, threshold and the scope notice.
```

The UI can present these learned flags beside an audit when a reviewed-data
artifact exists. A separate evidence review is needed to determine whether any
claim is supported or contradicted. Missing sponsorship disclosures, influencer
payment, market coordination, chart manipulation and company fundamentals need
additional evidence; this text model does not establish them.

Engineering checks can be run with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_framing_training.py -q
```
