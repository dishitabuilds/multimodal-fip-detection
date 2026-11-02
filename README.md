# A Lightweight Multimodal Model for Detecting Fake Investment Promotions

Minor Project — supervisor Dr. Shivangi Surati.

**ML lab track:** [Setup, SVM training, model comparisons and demo instructions](docs/ML_LAB.md).
The lab implementation adds TF-IDF + Linear SVM, Logistic Regression and Naive Bayes,
plus frozen MobileNetV3 image features and feature fusion. Open **ML Lab** in the
Streamlit page navigation. A local seven-model engineering pilot is available;
validated evaluation still requires a representative human-reviewed dataset.

**Audio and video:** [Local speech transcription, sampled-frame OCR and setup](docs/MEDIA.md).
Open **Audio Video** to upload clips up to 50 MB / 2 minutes, inspect timestamped
speech and on-screen wording, optionally apply existing lab classifiers, and
download evidence JSON. Install `requirements-media.txt` and explicitly prepare
the speech model once. Media detection accuracy has not been established.

**Misleading promotion analysis:** [Promotion Audit and claim evidence](docs/PROMOTION_AUDIT.md)
adds independent English wording explanations and a dated reviewer evidence ledger.
Open **Promotion Audit** to inspect return promises and persuasion cues, including
promotions that contain correct facts. A [multi-label SVM training path](docs/FRAMING_LABELS.md)
supports separately reviewed framing annotations; no validated framing model is
currently supplied. Unchecked facts and future stock performance remain unassessed.

This project studies investment promotions whose claim is carried by the image:
doctored profit screenshots, misleading charts, forged regulator logos and
impersonated public figures. It combines image features with caption and OCR
text, then compares **text-only, image-only and fused** models.

The repository now includes a **Streamlit frontend**, an AMFI legitimate-creative
collector, a resumable collection/curation pipeline and model/training code.
**A validated research checkpoint and research evaluation results are not included.**
The frontend supports uploads, sample previews and default OCR independently of
a classifier checkpoint. An internal Pilot demo is available explicitly;
Research model analysis requires the compatible checkpoint described below.

See [PROJECT_DOCUMENTATION.html](PROJECT_DOCUMENTATION.html) for the design,
module reference, limitations and change log.

## Current implementation work (26 October 2026)

See [DATASET.md](DATASET.md) for actual data populations and review commands,
and [docs/REMAINING_WORK.md](docs/REMAINING_WORK.md) for the updated checklist.
An isolated [internal pilot](docs/PILOT_RUN.md) now completes OCR, all three
training arms, checkpoint save/reload and resource measurement on 14 provisional
AI-labelled images. Its tiny random models and three-example test split are
engineering evidence, not research accuracy or a deployable detector.

- Experiment runs now seed model construction, use a common verified cohort,
  isolate variant/experiment outputs, and record input/config/checkpoint provenance.
- Full and manually reviewed subset evaluation reuse the same model and test
  predictions; `--evaluate-only --experiment-id ID` reloads saved runs. Use the
  original effective epoch/batch overrides and input files when evaluating them.
- AMFI collection now produces **unknown candidates pending review** with actual
  source-page attribution and no fabricated model-input captions. The local pilot
  contains 38 images; none has been human-approved for the research corpus.
- Input preparation/audit/review-import tools preserve decisions and source
  provenance. Downloaded image dimensions come from decoded bytes, not HTML.
- OCR backend listing is offline, failed cached OCR can retry, and the CLI has
  model-directory, download and CPU-thread controls. A three-record/six-image
  English/Hindi OCR pilot completed; OCR accuracy still needs manual measurement.
- The PyTorch budget reports actual FP32 execution separately from an INT8 target.
  OS peak RAM is measured when available; unsupported ONNX inputs fail clearly.

The historical curated corpus remains 79 fake / 0 real. Its existing split is
unsuitable for training and contains a repeated unrelated image across splits.
No validated research accuracy or achieved deployment budget is claimed. Three
internal tiny-model checkpoints are saved separately from the frontend artifact.

Verified on 26 October 2026 in the project's `.venv`: **182 tests and 11 subtests
passed**, including 12 UI tests and 22 frontend-analysis tests. Coverage includes
real tiny-model training/checkpoint reload, strict split validation, review
preservation/import, input/result state and checkpoint compatibility. Local
browser checks also completed the OCR and opt-in Pilot flows; none of this
establishes pretrained classifier performance.

## Recent collaborator changes

Commit [`ac3d7b3`](https://github.com/dishitabuilds/multimodal-fip-detection/commit/ac3d7b36d09b86831efe2b2fbfe3b135223f3dd4)
by `veerbhadrasinh-8` (27 September 2026) added:

- `app.py`: image uploads, optional captions, six selectable sample presets,
  OCR details, model scores, processing time and keyword indicators.
- `scripts/01b_collect_legitimate.py`: AMFI campaign creatives labelled `real`.
  The implemented source is AMFI; SEBI collection is not implemented.
- `test_samples/`: 11 PNG/WebP images, four of which appear in the current frontend.
  These are demo assets; the repository does not establish their membership in
  a held-out test split.
- A module-level `ExampleDataset` for multiprocessing pickling, and configured
  `num_workers: 0` for single-process data loading. Dropout remains `0.1` for
  both configured architectures; `drop_last=False` is unchanged.
- EasyOCR model lookup: an explicit `model_storage_directory`, otherwise
  `data/models/easyocr` when present, otherwise EasyOCR's default location.

The commit message mentions training on real data. Checkpoints and per-run
metrics were not committed, so that training and its performance cannot be
verified from this checkout.

## Run Promotion Lens

For the free hosted demo, see [Streamlit Community Cloud deployment](docs/DEPLOYMENT.md).
Its separate cloud entrypoint prepares public OCR and speech assets; the local
entrypoint below continues to use existing local model files only.

Promotion Lens is the local Streamlit research workbench. Use **standard CPython
3.10 or newer** on Windows; an MSYS2/MinGW Python may not have compatible
PyTorch wheels. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-frontend.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), or the address printed by
Streamlit. Reuse an existing `.venv` instead of recreating it. The frontend
requirements install the editable package and its Streamlit, PyTorch,
Transformers and EasyOCR dependencies; the `serve` extra is not a substitute.

Upload a JPEG, PNG or WebP image up to **10 MB**, or choose a demonstration
image. Captions are optional and should contain the original accompanying
post text. Demonstration captions start blank; source descriptions stay
separate from model inputs. These examples are not a held-out test set.

The interface separates three modes:

- **OCR** is the default. It extracts image text and shows descriptive text
  cues without a classifier or research checkpoint. OCR loads only prepared
  local EasyOCR files from `data/models/easyocr/`, with downloads disabled.
  Missing files produce an unavailable message; image previews still work.
- **Pilot demo** is an explicit opt-in to the internal engineering model.
  It uses a **14-image provisionally AI-labelled dataset** (7 train, 4 validation,
  3 test), using tiny
  randomly initialized encoders. Its scores are unvalidated demonstration
  outputs, not research accuracy or evidence that a promotion is genuine.
  The local checkpoint is
  `data/checkpoints/student/internal-ai-pilot-20261025/fused_seed42.pt`,
  with `configs/tiny-smoke.yaml`. These ignored artifacts are absent from a
  fresh clone. See [the internal pilot record](docs/PILOT_RUN.md).
- **Research model** requires a compatible checkpoint at
  `data/checkpoints/fused_seed42.pt` and locally cached matching
  encoder/tokenizer assets. No validated research checkpoint is supplied.
  The artifact must come from the current student fused training runner,
  match `configs/model.yaml`, and include `state_dict`, a validation-selected
  `threshold`, and architecture/text/input-pipeline provenance.
  Incompatible, metadata-free legacy and internal-pilot artifacts are rejected.

Previewing images or changing modes does not initialize models. Analysis uses
local assets and does not automatically download model weights or tokenizer
files. A class score is a model output, not a calibrated authenticity guarantee
or regulator approval; text cues are independent word-pattern observations.
Processing time includes OCR and any first-use initialization, so it is
separate from the model-only CPU budget benchmark.

## Collection and training

For a collection-only environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe tests/run_all.py
```

The test runner executes four offline suites: collection, curation, model
utilities and pipeline integration. Optional installs include `.[ocr]`,
`.[model]`, `.[graph]`, `.[serve]` and `.[dev]`; model training needs `.[model]`.
The frontend extra supplies EasyOCR without installing every OCR backend.

For the offline frontend regressions, install the development extra alongside
the frontend and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests/test_frontend.py tests/test_frontend_analysis.py -q
```

Verified on 26 October 2026 in the project's `.venv`: **182 tests and 11 subtests
passed**. The 12 UI tests cover source switching, caption/reset behavior,
persistent results and invalidation, OCR-only operation, explicit Pilot warnings,
missing Research checkpoints and invalid images. The 22 analysis tests cover
local-only loading, input fingerprints, alpha/EXIF handling, saved thresholds,
checkpoint compatibility, nonfinite-output rejection and error cleanup. These
checks verify behavior; they do not validate classifier accuracy.

Stage 02 is needed for Google API discovery records. Before OCR, prepare
`data/selected_input/` with one complete record per UID: use enriched records
in place of matching discovery records, together with direct-archive and AMFI
records. The JSONL directory reader does not merge duplicate UIDs.

Run the stages from the repository root using the environment's Python:

```powershell
python scripts/01_collect.py                  # fact-check archives; API route needs FACTCHECK_API_KEY
python scripts/01b_collect_legitimate.py --limit 100  # AMFI real-class candidates
python scripts/02_enrich_images.py            # fetch missing creatives
python scripts/03_ocr.py --backend easyocr --in-dir data/selected_input
python scripts/04_curate.py --in-dir data/interim/ocr/easyocr --no-dedup
python scripts/calibrate_dedup.py             # requires the initial finance.jsonl
python scripts/04_curate.py --in-dir data/interim/ocr/easyocr  # cluster after reviewing threshold
python scripts/05_split.py                    # group-aware, leakage-checked manifest
python scripts/06_budget.py --targets-only    # print targets; does not measure a model
python scripts/07_ablation.py                 # all configured arms/seeds; requires both classes
```

OCR writes `data/interim/ocr/easyocr/records.jsonl`; pass that directory to
curation so OCR text survives. Calibration prints a suggested normalized image
threshold; it does not change configuration. Supply a reviewed value through
`04_curate.py --image-threshold VALUE` when rerunning curation.

Here `python` means the virtual environment's interpreter; activate it or use
`.\.venv\Scripts\python.exe` explicitly. Collection stages are resumable.
Stage 01 obeys `robots.txt` and retries empty sources using configured fallback
routes. Stage 01b currently uses its own requests session and a 0.3-second delay
after successful image downloads; it does not share that robots/rate-limit
machinery. Review source access requirements before a larger AMFI collection.

Training reads `data/interim/finance.jsonl` and
`data/processed/split_manifest.json`. A quick debugging run is:

```powershell
python -m pip install -e ".[model]"
python scripts/07_ablation.py --arms fused --seeds 42 --epochs 1
```

One epoch verifies a training path; it is not evidence of a usable classifier.
The full run saves checkpoints to `data/checkpoints/<variant>/<experiment-id>/`
and per-run/aggregate metrics to `data/processed/ablation/<variant>/<experiment-id>/`.
Explicitly select a verified student fused artifact for the frontend's deployment
path above. The configured seeds are 13, 42 and 1337. `--image-only-claims` now
requires a manually reviewed UID manifest via `--reviewed-subset`; automatic
caption flags are only review aids. See `DATASET.md` for the manifest contract.

## Current evidence and remaining work

- The collection pipeline, deduplication, group splitting, OCR backends,
  transliteration, ablation arms, metrics and training loop are implemented.
- The local reports inspected on 1 October 2026 contain **100 input records →
  79 finance-labelled records, all `fake`**, grouped into 60 clusters. The local
  manifest assigns 61/9/9 records to train/validation/test
  (assignment hash `71ca28d778777ff1`). These files are gitignored and are absent
  from a fresh clone; they are not a newly collected or balanced corpus.
- AMFI collection provides a route to `real` examples, but the current local
  reports do not include those examples. Collect, manually review, curate and
  regenerate splits before training. Source/branding differences between AMFI
  and debunk archives can become shortcuts for a classifier.
- No trained checkpoint, held-out metrics or measured lightweight results are
  tracked. Verify the collaborator's artifacts before reporting accuracy/F1.
- Training text currently includes archive titles and claim text; these may
  reveal verdicts and differ from user-supplied captions. Audit text and image
  annotations, and compare arms on a common evaluation set before claiming
  that fusion improves detection. The text-only arm includes OCR; add a
  caption-only baseline to isolate the value of in-image text.
- Telegram propagation/GNN, distillation, INT8/ONNX export and API serving
  remain future work.

## Lightweight targets

These are **targets for the deployed compressed student**, not achieved results
or guaranteed hosting limits. The configured DistilBERT student must still be
measured and may require further compression.

| Metric | Target |
|---|---|
| Parameters | ≤ 50 M |
| Serialized model | ≤ 100 MB |
| CPU latency p50 / p95 | ≤ 100 / 250 ms |
| Peak RAM | ≤ 1,024 MB |
| Macro-F1 retention | ≥ 0.95 × teacher |

`configs/model.yaml` defines measurement conditions: batch 1, four threads,
224×224 images, 128 text tokens, 10 warm-ups and 100 timed runs, with an Intel
Core 7 150U reference machine. `scripts/06_budget.py` measures architecture
inference; its default weights/serialization are FP32. The configuration's INT8
target does not itself quantize the model. Quote actual hardware, precision,
checkpoint and evaluation data with any reported result.

## Layout

```text
app.py                         Streamlit frontend
requirements-frontend.txt      focused frontend installation
PROJECT_DOCUMENTATION.html     detailed project record
configs/                       sources, architecture, training and budgets
src/fipd/
  schema/                      shared FactCheckRecord / ImageAsset contract
  collection/                  WordPress, sitemap/JSON-LD, Google discovery
  enrichment/                  OCR and transliteration
  curation/                    finance relevance, labels, dedup and splits
  datasets/                    examples and tensor data loading
  models/                      encoders, fusion, ablation arms and budget
  training/                    metrics and training loop
  utils/                       HTTP, IO and logging
scripts/                       numbered pipeline/training CLIs
test_samples/                  tracked frontend demo images
tests/                         offline unit/integration suites
docs/                          earlier checklist and Phase 0 planning guide
data/                          local data, caches, reports and checkpoints
```

The earlier guides in `docs/` retain historical planning and commands; use this
README for the current file names and frontend setup. Keep secrets, collected
data and model checkpoints out of Git.
