# Project Checklist

**A Lightweight Multimodal Model for Detecting Fake Investment Promotions**

Ordered by dependency — items near the top unblock items below them. Estimates
are effort-days for one person, not calendar days. Two people can run most of
Phase 1 and Phase 2 in parallel; suggested splits are marked **[A]** / **[B]**.

Four items are marked **GATE**. Stop and get an answer before continuing past
them — each one can invalidate work done downstream.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · **GATE** decision point

---

## Phase 0 — Unblock (do this week)

- [ ] **Get a Google Fact Check Tools API key** — *0.5 d*
  Cloud Console → enable "Fact Check Tools API" → create key → `setx FACTCHECK_API_KEY "..."` → restart terminal.
  *Done when:* `python scripts/collect_factchecks.py --source googlefactcheck --limit 5 --no-images` returns records.

- [ ] **Run a full collection** — *0.5 d* (mostly waiting)
  `python scripts/collect_factchecks.py` then `python scripts/enrich_images.py --finance-only`
  *Done when:* `data/raw/` has records from every enabled source, or you've logged why a source failed.

- [ ] **GATE 1 — Read the yield table** — *0.5 d*
  `python scripts/build_dataset.py`
  *Decide:* if finance-labelled items are **under ~500**, the fact-check archives are a seed and evaluation set only, and Telegram collection moves from Phase 3 to Phase 1. Write the number down; it goes in the report.

- [ ] **Verify Factly and Vishvas News work from your machine** — *0.25 d*
  Both returned 403 from the dev sandbox (Cloudflare). If still blocked, flip `kind: wordpress` → `article` in `configs/sources.yaml` and use the `fallback` sitemap block.

- [ ] **GATE 2 — Agree the negative-class strategy with Dr. Surati** — *1 d discussion*
  Currently 100% of collected records are `fake`. Present the options (SEBI-registered advisor posts, AMFI campaigns, registered broker marketing, RBI/SEBI investor education) and the constraint: **negatives must be the same genre of image** — promotional finance graphics — differing only in whether they deceive.
  *Done when:* you have a written decision on where negatives come from.

- [ ] **Set up Git + GitHub** — *0.5 d*
  `.gitignore` is already written and excludes `data/`, `.session` files and API keys. Push before you accumulate more work.

- [ ] **Define "lightweight" as a number** — *0.25 d*
  Pick targets now, e.g. **< 100 MB model size**, **< 100 ms CPU inference**, **< 50 M params**. Without a target the word in your title is unfalsifiable.
  *Done when:* the targets are written into the README.

---

## Phase 1 — Data foundation

### 1.1 Negative class

- [ ] **Build the negative-class collector** — *3–4 d* **[A]**
  New scraper(s) under `src/collection/` following the same `BaseScraper` interface. Sources per Gate 2.
- [ ] **Verify negatives are genre-matched, by eye** — *0.5 d*
  Put 20 positives and 20 negatives side by side. If you can tell them apart by image *style* rather than content, the model will too — and it will learn the wrong thing.
- [ ] **Target rough balance** — aim for at least 1:3 negative:positive; log the final ratio.

### 1.2 OCR — the core of the premise

- [ ] **Choose and install the OCR engine** — *1 d* **[B]**
  PaddleOCR first (better on Devanagari), EasyOCR as comparison, Tesseract as fallback.
- [ ] **Benchmark OCR on 30 hand-transcribed images** — *1 d*
  Report character error rate for English and Hindi separately. Needed for the report; also tells you whether OCR is good enough to build on.
- [ ] **Write `src/ocr/run_ocr.py`** — *1–2 d* **[B]**
  Batch over `data/images/`, write into `ImageAsset.ocr_text`. Must be resumable, same as the scrapers.
- [ ] **Add transliteration normalisation** — *1 d*
  `indic-transliteration` so `paisa double` and `पैसा डबल` don't look unrelated to MuRIL. Decide and document: normalise to Devanagari, or to Latin, or keep both.
- [ ] **Re-run the finance filter with OCR text present** — *0.25 d*
  The filter already reads `ocr_text`. Yield should rise — record by how much, it's a nice result.

### 1.3 Deduplication — skip this and your results are invalid

- [ ] **Add perceptual hashing** — *1 d* **[A]**
  `imagehash` (pHash + dHash). Current SHA-256 only catches byte-identical files; it will miss re-compressed and re-cropped copies of the same creative.
- [ ] **Cluster near-duplicates into claim groups** — *1 d*
  The same viral creative is debunked independently by BOOM, Factly and Newschecker. Those are one item, not three.
- [ ] **Report the duplicate rate** — goes in the paper.

### 1.4 Splits

- [ ] **Write `src/data/split.py` with group-aware splitting** — *1 d* **[A]**
  Split by **claim cluster**, never by article. If a creative appears in train and test, your accuracy measures memorisation.
- [ ] **Stratify by label and by source** — so no split is accidentally all-BOOM.
- [ ] **Assert zero cluster overlap between splits** — write it as a test, not a manual check.
- [ ] **Freeze the splits to disk and commit the manifest** — reproducibility.

### 1.5 Manual verification

- [ ] **Set up Label Studio** — *0.5 d* **[B]**
- [ ] **Hand-check a 200-item sample** — *1–2 d, split between both of you*
  Specifically hunting: verdict-stamp leakage, wrong labels, non-creative images, OCR garbage.
- [ ] **Report inter-annotator agreement** (Cohen's κ) — *0.25 d*
  Both of you label the same 50 items. Free credibility in the paper.
- [ ] **Fix whatever the sample exposes** — likely lexicon and blocklist tweaks.

### 1.6 Benchmarks

- [ ] **Obtain Fin-Fact and load it into `FactCheckRecord` format** — *1–2 d* **[B]**
- [ ] **Obtain Fakeddit (or justify dropping it)** — *1 d*
  It's Reddit-general, not finance — decide whether it earns its place or is just cited.
- [ ] **Record published baseline numbers on Fin-Fact** — what you must beat or match.

- [ ] **GATE 3 — Dataset is ready** — *0.5 d*
  *Done when:* balanced classes, OCR populated, duplicates clustered, splits frozen with no leakage, sample manually verified, counts written into a `DATASET.md`.

---

## Phase 2 — Model and ablation

Build the ablation arms **first**. They are the deliverable your guide asked for,
not an appendix, and they give you a working number early.

- [ ] **Data loader + preprocessing** — *2 d* **[A]**
  Tokenisation (MuRIL), image transforms, caption + OCR text concatenation strategy (document what you chose — separator token vs. plain concat matters).
- [ ] **Text-only baseline (MuRIL)** — *2 d* **[A]**
- [ ] **Text-only baseline (IndicBERT)** — *1 d* — pick the better one, report both.
- [ ] **Image-only baseline (ViT)** — *2 d* **[B]**
- [ ] **Image-only baseline (CLIP)** — *1 d* **[B]**
- [ ] **Training loop + W&B logging + checkpointing** — *2 d*
  Log seeds and configs from day one; you will need them for the paper.
- [ ] **Cross-attention fusion model** — *3–4 d* **[A]**
- [ ] **Hyperparameter sweep** — *2 d* (W&B sweeps; keep it modest, Colab has limits)

### Evaluation — get the metrics right

- [ ] **Report precision, recall, F1, PR-AUC — not accuracy** — *1 d*
  Accuracy is close to meaningless on imbalanced fraud data.
- [ ] **Precision–recall curve across thresholds** — a detector that flags everything has perfect recall and no value.
- [ ] **Confusion matrices per arm** — *0.5 d*
- [ ] **Run the ablation table: text-only vs image-only vs fused** — *1 d*
  This is what Dr. Surati specifically asked for. Report the *difference*, with the sign and magnitude, not just three numbers.
- [ ] **Statistical significance** — *0.5 d*
  3–5 seeds per arm, report mean ± std. A 1% gain on one seed is noise.

- [ ] **GATE 4 — The targeted experiment** — *1–2 d*
  Build an evaluation subset of posts where **the caption is benign and the claim is image-only**. If fusion doesn't beat text-only *there*, the central thesis is not demonstrated regardless of overall F1.
  *This is the single most convincing experiment in the project.* Budget real time for it.

- [ ] **Error analysis** — *1 d*
  Pull 30 failures. Categorise them. What kind of fraud does it miss, and why? Reviewers read this section closely.
- [ ] **Compare against published Fin-Fact results** — *1 d*

---

## Phase 3 — Propagation branch

Feasibility risk lives here. Set the decision point before you start building.

- [ ] **Telethon setup + API credentials** — *0.5 d* **[B]**
- [ ] **Identify seed public channels** — *1–2 d*
  Use SEBI orders against unregistered advisors to help label channels.
- [ ] **Collect posts + forwarding metadata** — *3–5 d, mostly waiting*
- [ ] **GATE — Are the forwarding chains dense enough?** — *0.5 d*
  *Decide:* if the graphs are too sparse to carry signal, **degrade to two modalities and report the graph attempt as an honest negative result.** That is a legitimate finding, not a failure — but only if you decide it deliberately rather than run out of time.
- [ ] **Build forwarding graphs with NetworkX** — *2 d*
- [ ] **Node/edge feature engineering** — *2 d*
- [ ] **GNN branch in PyTorch Geometric** — *3–4 d*
- [ ] **Three-way fusion + re-run the ablation** — *2–3 d*

---

## Phase 4 — Lightweight and demo

- [ ] **Measure the teacher model first** — *0.5 d*
  Params, size in MB, CPU and GPU latency. You cannot claim a reduction without a baseline.
- [ ] **Knowledge distillation into a student** — *3–4 d* **[A]**
- [ ] **ONNX export** — *1 d*
- [ ] **Quantisation (INT8) + ONNX Runtime benchmark** — *1 d*
- [ ] **Report the size/latency/accuracy tradeoff table** — *0.5 d*
  This is what turns "lightweight" from an adjective into a result. Check it against the targets you set in Phase 0.
- [ ] **FastAPI backend** — *2 d* **[B]**
- [ ] **Streamlit frontend** — *2 d* **[B]**
  Upload a screenshot → OCR → prediction → confidence → which modality drove it. The last part demos well and shows the fusion is doing something.
- [ ] **Record a demo video / screenshots** — *0.5 d*

---

## Phase 5 — Writing and submission

- [ ] **Literature review** — *3–4 d* — start early, run it in parallel with Phase 1
- [ ] **Related work section** — *2 d*
- [ ] **Methodology section** — much of it is already written in `docs/PHASE1_WALKTHROUGH.md` §3
- [ ] **Results + figures** — *3 d* — ablation table, PR curves, confusion matrices, size/latency table
- [ ] **Limitations section** — *1 d*
  Write it honestly: archive bias toward political content, single-country scope, class-balance construction, OCR error propagation. Examiners reward this.
- [ ] **Reproducibility appendix** — seeds, configs, split manifest, package versions
- [ ] **Clean the repo + write final README** — *1 d*
- [ ] **Report / paper draft** — *5–7 d*
- [ ] **Presentation deck** — *2 d*

---

## Running risk register

Review at every supervisor meeting.

| Risk | Trigger to watch | Fallback |
|---|---|---|
| Not enough finance items | Gate 1 yield under ~500 | Telegram moves to Phase 1 |
| No usable negative class | Gate 2 unresolved after 2 weeks | Reframe as anomaly detection / one-class |
| Duplicate leakage inflates results | Test F1 suspiciously high (> ~0.95) | Re-check cluster overlap before believing any number |
| Verdict-stamp leakage | Image-only arm does implausibly well | Manual sample; strip annotated images |
| Forwarding graphs too sparse | Phase 3 gate | Two-modality model + honest negative result |
| OCR too weak on Devanagari | CER above ~30% | Weight OCR text lower; lean on the image encoder |
| Colab/Kaggle GPU limits | Sweeps not finishing | Smaller backbones, shorter schedules, fewer seeds |

---

## Suggested split between the two of you

**[A]** — data engineering and modelling: negatives, dedup, splits, fusion model, distillation
**[B]** — OCR, benchmarks, annotation, graph branch, demo

Both: manual annotation sample, evaluation, writing.

---

*Phase 1 pipeline is complete: 23 files, 2,164 lines, 56 tests passing.
Everything above is what remains.*
