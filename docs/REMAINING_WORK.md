# Remaining project work

Data snapshot: **24 October 2026**. Implementation verification continued on
**26 October 2026**.

The classical lab track and promotion-framing extension were added on **30 October
2026**. See [ML_LAB.md](ML_LAB.md) for the seven-model engineering pilot and
[PROMOTION_AUDIT.md](PROMOTION_AUDIT.md) for the independent wording/evidence audit.
The reviewed multi-label training path is described in [FRAMING_LABELS.md](FRAMING_LABELS.md).
No reviewed framing dataset or validated framing artifact is supplied. Next steps
include attributed framing annotations, balanced/educational negative examples,
campaign-isolated evaluation and sourced review of selective context. Automated
retrieval of company filings, visual chart analysis and influencer/coordination
investigation remain unimplemented.

The collection, OCR, curation, splitting, model/training and Streamlit code is
implemented. The main outstanding work is preparing suitable data and producing
verified experimental results. Code being present does not establish model quality.

The user confirmed the title on 25 October 2026. The working core scope is
image/text detection plus measured lightweight performance. Detailed objectives
were not supplied; proposed extensions are listed separately.

## First implementation batch — 24 October 2026

- [x] Correct experiment seeds, common cohort checks, checkpoint provenance and
  separate experiment outputs; test actual tiny-model training and reload.
- [x] Require reviewed subset IDs and score them from the same full-test predictions.
- [x] Report actual FP32 execution separately from the INT8 deployment target;
  use OS peak-memory measurements where available and reject unsupported ONNX input.
- [x] Add deterministic raw/enriched input preparation and dataset audit/review exports.
- [x] Correct the AMFI collector's provenance and automatic-label assumptions;
  collect 30 image candidates with unknown labels pending review.
- [x] Make OCR backend listing offline, initialize once, retry failed cached OCR,
  and add explicit model-cache/download/thread controls.
- [x] Exclude annotation and unselected-image OCR from the model's text input.
- [x] Add explicit review import, preserving labels/relevance through curation
  and keeping original archive text outside model inputs.
- [x] Repair image geometry on selected copies and future downloaded assets.
- [x] Complete an OCR pilot: three records / six images, no runtime errors;
  manual OCR-quality measurement remains open.
- [x] Protect completed review queues from audit overwrite; verify reviewed image
  bytes and preserve OCR-quality annotations during import.
- [x] Revalidate current record/cluster coverage, split hashes and counts before
  training; reject stale manifests even if their earlier leakage flag passed.
- [x] Validate student/fused checkpoint architecture, tokenizer/text settings,
  image/token limits and input-pipeline version before frontend initialization.

See `DATASET.md` for actual populations, review requirements and current commands.
The historical cluster-only leakage check missed an unrelated image reused
across train/validation/test; review and regenerated splits remain necessary.

## Current evidence

Second batch on 25 October: visual triage, original-image recovery and an
internal end-to-end pilot are documented in `docs/PILOT_RUN.md`.

- [x] Visually inspect 79 archive records and 38 AMFI candidates with separate,
  explicitly AI-assisted findings; keep human approval fields untouched.
- [x] Recover 61 body-image candidates from 13 articles and exclude verifier
  screenshots when selecting the internal pilot.
- [x] Build/OCR a separate 14-image provisional two-class pilot; enforce an
  internal-only label policy for training.
- [x] Train all three tiny arms, save checkpoints, reload without retraining and
  measure the fused checkpoint's size/CPU latency.
- [x] Establish the original student's 159.7M parameter gap and add an optional
  39.9M architecture candidate; pretrained validation remains open.

- 1,208 raw archive records; 100 enriched records.
- 79 curated candidates, **79 fake / 0 real**, in 60 claim clusters.
- Existing train/validation/test split: **61 / 9 / 9**, all one class.
  The cluster-overlap check passes; assignment hash: `71ca28d778777ff1`.
- The historical curated candidates still lack OCR. Separate OCR pilots and
  38 unreviewed AMFI images are saved locally; the original-caption
  evaluation subset remains to be reviewed.
- No validated research checkpoint/results. Internal tiny checkpoints, smoke
  ablations and a partial budget report now exist, with no accuracy claim.
- Promotion Lens now provides the complete frontend flow: default OCR without a
  classifier checkpoint, persistent evidence/results, and a separate opt-in
  internal Pilot demo. Local browser checks exercised OCR and Pilot analysis.
  Research mode remains unavailable until a compatible trained student checkpoint
  is supplied at `data/checkpoints/fused_seed42.pt`.

Evidence: `data/interim/curation_report.json`,
`data/processed/split_manifest.json`, `data/interim/finance.jsonl`, and the local
data/output directories. Unshared collaborator artifacts may change this status.

## Checklist, in dependency order

### 1. Confirm the required deliverable

- [ ] Reconcile the original problem statement and supervisor requirements with
  the current implementation: required modalities, languages, baselines,
  benchmarks, compression and submission/demo expectations.
  The title is confirmed; Telegram/GNN remains optional under the working scope.
- [ ] Obtain any collaborator checkpoint and its matching config, dataset/split
  provenance and evaluation results; verify them before using them as evidence.

### 2. Finish the dataset — the immediate blocker

- [ ] Review investment-promotion relevance. Existing finance-keyword candidates
  include tax, political and crowdfunding claims that may fall outside the task.
- [ ] Collect and manually verify legitimate investment-promotion examples.
  Thirty-eight AMFI image candidates have been collected; their labels/relevance are
  still pending and they are excluded from training. Match
  creative styles and sources as far as possible to avoid label shortcuts.
- [ ] Audit labels, original creatives, archive verdict stamps and input text.
  Remove information that would reveal the answer but would be absent in a user upload.
- [x] Build one selected record per UID with explicit raw/enriched precedence.
  Current source inventory: 1,246 unique UIDs; the final reviewed corpus remains open.
- [ ] Run OCR, check quality on hand-transcribed English/Hindi examples if those
  languages remain in scope, and re-curate while retaining OCR text.
- [ ] Complete a documented annotation audit and reviewer agreement check.
- [ ] Recalibrate deduplication and regenerate frozen group-aware splits with
  both classes and adequate source/language coverage. Repeat the leakage check.
- [x] Write `DATASET.md` with current provenance, populations, review protocol and
  known problems.
- [ ] Add final annotation results, exclusions, duplicate rate and regenerated
  split hash to `DATASET.md` after review.

Done when: a reviewed two-class corpus is ready for training and independent
evaluation. The current one-class split is a pipeline rehearsal.

### 3. Make formal experiments reproducible

- [x] Set each run's seed **before model construction**, including newly
  initialized fusion and classifier weights.
- [x] Enforce identical evaluation example IDs across text, image and fusion
  arms, with readable chosen images and both classes in each split.
- [x] Separate checkpoint/report paths by architecture variant and experiment
  mode so teacher/student and full/subset runs cannot overwrite one another.
- [x] Run real-tensor training/checkpoint/reload tests with tiny local models.
- [ ] Smoke-test the configured pretrained encoders on the reviewed dataset
  before long runs.

Evidence: `scripts/07_ablation.py`, `src/fipd/training/loop.py`.

### 4. Train and evaluate

- [ ] Train text-only, image-only and fused arms on the final splits over the
  configured seeds (13, 42, 1337); retain configs, checkpoints and run reports.
- [ ] Add a caption-only baseline to distinguish OCR's benefit from image-feature
  fusion. The existing text-only arm already includes OCR.
- [ ] Report held-out precision, recall, macro-F1, PR-AUC, confusion matrices and
  mean/standard deviation across seeds, alongside the majority-class baseline.
- [ ] Create a manually reviewed benign-caption / image-carried-claim subset and
  evaluate the same models on it. Archive claim text is only a caption proxy.
- [ ] Inspect misclassifications and document recurring failure modes.

Done when: reproducible held-out results establish what fusion adds, including
the project's image-carried-claim premise.

### 5. Demonstrate that the model is lightweight

- [ ] Measure the trained artifact's parameters, serialized size, CPU p50/p95
  latency and peak RAM under the configured conditions.
- [x] Correct measurement limitations: end-of-run RSS is not peak RAM; an
  `int8` config label does not quantize weights; the current checkpoint loader
  does not implement the advertised ONNX input path.
- [ ] Compare results with the configured targets: <=50M parameters, <=100MB,
  <=100ms p50 / <=250ms p95 CPU latency, <=1,024MB RAM, and >=95% teacher
  macro-F1 retention when a teacher comparison is required.
- [ ] If needed to meet the agreed targets, compress/quantize/distill the model
  and repeat both accuracy and resource measurements on the final artifact.

Evidence: `configs/model.yaml`, `scripts/06_budget.py`,
`src/fipd/models/budget.py`. Frontend processing time also includes OCR and
initialization; it is a separate measurement from model-only latency.

### 6. Validate the complete demo and finish submission

- [ ] Load a verified **research** checkpoint into Streamlit and exercise upload ->
  OCR -> prediction with held-out real and fake examples; verify threshold and
  preprocessing. The completed OCR/Pilot frontend does not satisfy this research
  validation requirement.
- [ ] Record demonstration screenshots/video and package the model/config/setup
  instructions so another person can reproduce the demo.
- [ ] Finish the report: literature/related work, methodology, dataset, results,
  error analysis, limitations and reproducibility appendix.
- [ ] Prepare the presentation and update the historical checklist/documentation
  to reflect measured final results.

## Confirm against the problem statement before expanding scope

- [ ] Telegram collection, forwarding graphs and a GNN propagation branch.
- [ ] External benchmark imports/evaluation such as Fin-Fact or Fakeddit.
- [ ] Specific alternative encoders, a teacher/student study, distillation or
  ONNX/INT8 deliverables beyond what is needed for the agreed lightweight targets.
- [ ] A separate FastAPI service or hosted deployment.

These are proposed in earlier planning; their absence does not automatically
make every one a submission requirement.

## Verification note

In the project's `.venv`, **182 tests and 11 subtests** passed on
26 October 2026, including 12 UI tests and 22 frontend-analysis tests, with local
mock-server and temporary-file access. Regression
suites cover experiment orchestration, manifest validation, checkpoint
compatibility, OCR behavior, budgets, input selection, review preservation/import,
image metadata, original-body recovery, provisional-label policy and candidate
collection. Frontend coverage includes source/state transitions, OCR-only mode,
explicit Pilot warnings, missing Research checkpoints, local-only asset loading,
invalid/nonfinite inputs and result/error cleanup. Local browser verification
also completed OCR and Pilot analysis. The end-to-end tiny pilot saved and
reloaded all three checkpoints.
Its fused checkpoint measured 1.068M parameters, 4.111 MiB, and 32.55/36.86 ms
CPU p50/p95 under the short smoke conditions. Peak RAM and F1 retention remain
unmeasured. `git diff --check` passed.
Tiny-model training tests verify tensor/checkpoint mechanics; they do not verify
pretrained model quality or provide held-out project results.
