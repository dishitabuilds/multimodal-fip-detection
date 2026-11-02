# Dataset status and review protocol

Data collected on **24 October 2026**, with implementation checks continued on
**25 October 2026**. This is a candidate corpus under
review, not a finished training/evaluation dataset.

## Populations and provenance

- Archive discovery: 1,208 raw records (Alt News 223, BOOM 8, Factly 400,
  Newschecker 177, Vishvas News 400).
- Existing image enrichment: 100 records; these replace matching raw UIDs,
  rather than adding 100 distinct examples.
- Historical curated candidates: 79 records, all `fake`, in 60 clusters.
- Historical split: train 61, validation 9, test 9; assignment hash
  `71ca28d778777ff1`. Every split is single-class.
- New AMFI collection: 38 downloaded image candidates (30 initial + 8 targeted),
  **all `unknown` pending human review**, with no download failures. The sitemap exposed 413 unique eligible
  image URLs. All 38 parent-page language values are `en`; image-language
  coverage has not been verified.
- Source selection now resolves 1,346 raw/enriched rows to 1,246 distinct UIDs
  (1,208 archive records plus 38 AMFI candidates). This source inventory is not
  the same population as the 79 curated candidates.

AMFI candidates retain their actual sitemap parent page and image URL, filename
provenance, and `suggested_label: real` in raw metadata. They have no invented
caption, article title or regulatory-approval claim in the model input. Official
source provenance alone does not establish task suitability or a training label.

Collected JSONL, images, OCR caches and checkpoints are local ignored artifacts.
Do not put them or credentials into Git. Keep a separate versioned data snapshot
and record its hashes when final experiments are produced.

## Problems established by the audit

An additional audit on **2 November 2026** inspected the complete selected source
inventory, independently of the historical 79-record curated corpus. All 1,246
selected candidates remain `unknown`; this is not a reviewed training dataset.
Fresh review files are under `data/processed/accuracy-review-20261102/`, including
`manual_review.csv` and `dataset_audit.json`. Reported blockers are unresolved
labels, missing both reviewed classes, and some missing usable images/OCR.
No reviewer decisions or source records were changed. Complete the attributed
review/import steps below before retraining. See [media support](docs/MEDIA.md)
for separate audio/video evaluation requirements.

AI-assisted visual inspection covered all 79 historical archive candidates:
46 were provisionally relevant investment promotions, 32 were off-task, and one
was uncertain. Most relevant records had only verdict-stamped hero composites.
This triage is stored separately under `data/processed/visual_review/`; original
source records and human decision fields were not changed.

Article extraction was choosing early sidebar `<article>` cards or empty React
stream placeholders instead of the main story. Corrected selectors recovered
61 image candidates across 13 of 19 attempted articles, into new ignored outputs.
Some images are verification/reference material rather than original promotions;
download success alone does not make an asset suitable. Unavailable files were
reported without bypassing source access errors.

A separate 14-image **internal-only AI-assisted pilot** now has OCR and a
7/4/3 split. Its proposed 8 fake / 6 real labels do not change the historical
research corpus. Normal research runs reject this pilot; details, checkpoints,
limitations and reproduction commands are in `docs/PILOT_RUN.md`.

The archive-only audit is saved at `data/processed/audit/dataset_audit.json`.
It found missing real examples and OCR, 52 titles with possible verdict markers,
30 candidates without clear investment terms in their claim/image text, and
nine possible nonpromotion finance/political stories. These are review flags,
not automatic relevance decisions.

Cluster overlap alone was insufficient: exact local image bytes appeared in
train, validation and test. Visual inspection identified the repeated Factly
asset as an unrelated blood-donation illustration. Fourteen records selected
it because its HTML dimensions were populated while their larger hero images
had no stored dimensions. Decoded dimensions and manual creative selection
are needed; image size alone does not establish task relevance.

The current split must be replaced after review. Its earlier `leakage_ok: true`
only described cluster assignments, not a complete clean-data guarantee. Stage
07 now rejects chosen-image byte overlap and single-class/empty evaluation splits.

## Reproducible preparation and audit

Use the existing virtual environment from the project root:

```powershell
# Collect a small additional batch of pending-review candidates.
.\.venv\Scripts\python.exe scripts/01b_collect_legitimate.py --limit 30

# Prefer complete enriched records over matching raw records, one UID per row.
.\.venv\Scripts\python.exe scripts/prepare_inputs.py --inspect-images --data-dir data

# Review the historical curated population and its split independently.
.\.venv\Scripts\python.exe scripts/08_audit_dataset.py --manifest data/processed/split_manifest.json --selection-report data/selected_input/selection_report.json

# Review the new AMFI population separately.
.\.venv\Scripts\python.exe scripts/08_audit_dataset.py --in-file data/raw/amfi.jsonl --out-dir data/processed/amfi_review
```

Selection refuses malformed inputs, conflicting known labels/provenance and
unequal records at the winning tier. On conflict it preserves the existing
selected corpus and writes the reasons. Audit exports keep every candidate and
leave reviewer decisions blank. CSV and lossless JSONL versions are provided.
An audit refuses to replace an existing queue containing reviewer entries in
either format; use a new output directory for a separate review round.
`--inspect-images` updates geometry only on selected copies and records the old
and decoded dimensions in the selection report; raw/enriched sources stay intact.

## Human review criteria

For each retained item, verify and record:

1. The item is an investment promotion within the approved problem statement.
   General finance, politics and investor education may require exclusion or a
   distinct evaluation stratum; they are not interchangeable by default.
2. The fake/real label has supporting evidence and a named reviewer. Do not
   derive the label from the article headline alone.
3. One original creative is explicitly selected. Exclude verdict overlays,
   stock art, unrelated recommendations and publisher UI. A large image may
   still be the fact-checker's annotated thumbnail.
4. OCR corresponds to that creative and is readable. Check English and Hindi
   separately where those languages remain in scope.
5. An original caption is recorded only if it is actually available and its
   source is known. An archive claim summary is not an original social caption.

The review CSV has explicit relevance/label/image-usability/reviewer fields,
including a chosen image path. Optional original-caption fields require a source
and confirmation. Accepted records should contain only the selected creative;
archive titles/claim summaries remain provenance rather than model input.

After completing the AMFI review CSV, import accepted decisions into a new folder:

```powershell
.\.venv\Scripts\python.exe scripts/apply_reviews.py --records data/raw/amfi.jsonl --reviews data/processed/amfi_review/manual_review.csv --data-dir data --out-dir data/reviewed/amfi_batch1
```

Pending, rejected and conflicting rows are reported and excluded. Approved
labels and relevance persist through curation, and the review CSV/source hashes
are retained. Chosen files are verified/decoded and checked against their stored
hashes; accepted geometry/hash and OCR-quality annotations are preserved. Do not
treat an empty review queue as accepted data.

Use independent overlapping review to measure agreement, and record disagreement
resolution. The earlier plan proposed 200 reviewed items and 50 overlapping
labels; confirm sample sizes against the final approved scope and corpus size.

## OCR and evaluation requirements

```powershell
# Listing installed backends never initializes engines or downloads weights.
.\.venv\Scripts\python.exe scripts/03_ocr.py --list-backends

# Small OCR pilot; model download is allowed unless --no-download is added.
.\.venv\Scripts\python.exe scripts/03_ocr.py --backend easyocr --in-dir data/enriched --model-dir data/models/easyocr --threads 4 --limit 3 --show
```

The pilot completed on three records / six images, with six text results and no
runtime errors, saved under `data/interim/ocr/easyocr/`. Some output is visibly
noisy. Its script-mix labels describe recognized characters, not verified language
coverage. An OCR run is not an OCR accuracy benchmark. Hand-transcribe a reviewed sample
and measure character error rates before making accuracy claims. Corpus-wide
OCR and manual quality review remain outstanding.

After labels/creatives/OCR are reviewed, rerun curation and dedup calibration,
freeze group-aware splits containing both classes, audit exact image exposure,
and record source/language/class distributions and the final assignment hash.
Stage 07 recomputes record coverage, current-cluster consistency, split counts
and assignment hash instead of trusting a previously saved leakage flag.
Keep all arms on the same held-out example IDs. Report multiple seeds, errors,
majority baseline, precision/recall, macro-F1 and PR-AUC.

A manually reviewed image-carried-claim subset uses a manifest such as
`{"name":"image_only_claims","reviewed":true,"uids":[...]}` only **after**
review. Its members must belong to the held-out test cohort and include both
classes. Automatic caption/finance flags do not establish the subset. Full and
subset evaluation now reuse the same trained model and predictions.

No trained model quality, completed Hindi coverage, balanced dataset or achieved
lightweight budget is claimed by this document.
