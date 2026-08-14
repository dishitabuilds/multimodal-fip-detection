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

- [~] **Run a full collection** — *0.5 d* (mostly waiting)
  `python scripts/01_collect.py` then `python scripts/02_enrich_images.py --finance-only`
  *Done when:* `data/raw/` has records from every enabled source, or you've logged why a source failed.
  **2026-08-15:** first real collection run, no images yet. Every key-free
  source is now either producing or has a logged reason. Still to do: re-run
  without `--limit`, and with images, once Gate 1 is answered.

- [~] **GATE 1 — Read the yield table** — *0.5 d*
  `python scripts/04_curate.py`
  *Decide:* if finance-labelled items are **under ~500**, the fact-check archives are a seed and evaluation set only, and Telegram collection moves from Phase 3 to Phase 1. Write the number down; it goes in the report.

  **First measurement, 2026-08-15** — 400 records/source cap, no images, no OCR:

  | source | raw | finance | labelled+finance |
  |---|---|---|---|
  | newschecker | 177 | 80 | 20 |
  | factly | 400 | 143 | 14 |
  | vishvasnews | 400 | 14 | 13 |
  | altnews | 223 | 32 | 5 |
  | boomlive | 8 | 0 | 0 |
  | **total** | **1208** | **269** | **52** |

  **52 usable items from 1208 collected — a 4.3% end-to-end yield.** All 52 are
  labelled `fake` (Gate 2 confirmed empirically, not just suspected).

  This is provisional and the true number is higher, for three reasons:
  1. **The verdict bug below cost 81% of the yield** and is now fixed; the
     re-run should land nearer **~113**.
  2. **No OCR yet.** The finance filter is scoring captions and body text only.
     Yield should rise once `ocr_text` is populated — measure by how much, it
     is a result worth reporting.
  3. **Capped at 400/source.** Not a full harvest.

  Even at ~113 this is far short of 500, so **plan for Gate 1 to fail**: the
  archives look like a seed and evaluation set, and Telegram collection should
  be expected to move into Phase 1. Do not treat that as decided until the
  re-run with OCR gives a real number.

- [x] **Fix the verdict loss in the WordPress route** — *found and fixed 2026-08-15*
  **217 of 269 finance-relevant records (81%) were being dropped as unlabelled,
  every one of them with an empty `verdict_raw`.** The cause: the WordPress
  REST API has no rating field at all, so `verdict_raw` was never populated for
  any WP-sourced record. The verdict was published all along, in ClaimReview
  markup on the article page the REST route never fetched.
  Now recovered by fetching the article when a post arrives without a verdict
  (one extra cached request per record). Measured recovery, on a 272-record
  sample:

  | source | recovered | how |
  |---|---|---|
  | newschecker | **75 / 100** | Next.js streamed payload |
  | altnews | **60 / 100** | plain `ld+json` tag |
  | factly | 0 / 72 | publishes no ClaimReview at all |

  Newschecker needed a second extraction route: it is a Next.js app-router site
  and injects its JSON-LD client-side from an escaped string inside
  `self.__next_f.push([...])`, so there is no `<script type="application/ld+json">`
  for a parser to find and the verdict looks absent when it is right there.
  Both routes are covered by tests.
  **Factly still yields no verdicts — worth 30 minutes to find out whether its
  verdict lives somewhere else in the page**, since it is the largest single
  source of finance-flagged records (143).

- [x] **Verify Factly and Vishvas News work from your machine** — *0.25 d*
  **Answered 2026-08-15.**
  * **Factly — works.** The WP REST API answers normally from this machine; the
    dev-sandbox 403 was the sandbox's IP, not the site. Left on `kind: wordpress`.
  * **Vishvas News — blocked, and not by Cloudflare.** Its WP REST API returns
    **HTTP 401** to anonymous callers on every search term, so no amount of
    retrying helps. Now collected through its sitemap instead (`sitemap_index.xml`,
    34 child sitemaps, whole archive), which works.
  * The `fallback:` block in `configs/sources.yaml` was documented but **never
    read by any code**. It is now wired up: a source that returns zero records
    is retried through its fallback automatically (`scripts/01_collect.py`,
    disable with `--no-fallback`).
  * **BOOM Live publishes only three daily sitemaps** — `/sitemap.xml` and
    `/sitemap-index.xml` are both 404. That route reaches the last few days,
    not the archive (8 records). BOOM's back catalogue needs the Google Fact
    Check API, which makes the API key below more important than it looks.

- [ ] **GATE 2 — Agree the negative-class strategy with Dr. Surati** — *1 d discussion*
  Currently 100% of collected records are `fake`. Present the options (SEBI-registered advisor posts, AMFI campaigns, registered broker marketing, RBI/SEBI investor education) and the constraint: **negatives must be the same genre of image** — promotional finance graphics — differing only in whether they deceive.
  *Done when:* you have a written decision on where negatives come from.

- [~] **Set up Git + GitHub** — *0.5 d*
  **2026-08-15:** repo initialised, initial commit made, work branched onto
  `phase0-unblock`. Two `.gitignore` fixes came out of it: the `.gitkeep`
  negation could never fire (git cannot re-include a file whose parent
  directory is excluded, so `data/raw/` became `data/raw/*`), and a
  `.gitattributes` now pins LF so two machines do not fight over line endings.
  **Still to do: create the GitHub remote and push.** That needs your account.

- [x] **Define "lightweight" as a number** — *0.25 d*
  **Done 2026-08-15.** Targets are in `configs/model.yaml` and the README:
  **≤ 50 M params · ≤ 100 MB on disk · ≤ 100 ms p50 / 250 ms p95 CPU ·
  ≤ 1 GB peak RAM · ≥ 0.95 × teacher macro-F1.**
  The last row is what stops the budget being gamed — without it, a model that
  answers "fake" in 3 ms passes every other target.
  Measurement conditions are pinned alongside them (batch 1, 4 threads, 224px,
  128 tokens, median of 100 runs after 10 warm-ups, named reference machine),
  because a latency number without its conditions is not comparable to anything.
  `python scripts/06_budget.py` is the test; `--targets-only` prints the budget
  without needing torch or a model.

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

> **2026-08-15 — the scaffolding for this phase is written and the non-torch
> parts are tested (78 new assertions).** What exists: `src/fipd/datasets/`
> (records → examples, torch dataset), `src/fipd/models/` (encoders, arms,
> cross-attention fusion, budget), `src/fipd/training/` (metrics, loop,
> aggregation), and `scripts/07_ablation.py`. What has NOT happened: nothing has
> been trained, because there is no negative class yet (Gate 2) and torch is not
> installed on this machine. The arms are unverified against real tensors —
> treat the shapes as unproven until a smoke run passes.

- [x] **Data loader + preprocessing** — *2 d* **[A]**
  `datasets/examples.py` (split-aware, drops `unknown` rather than treating it
  as a negative, picks the creative over the fact-checker's verdict stamp) and
  `datasets/torch_data.py`. Concatenation strategy is documented and configured
  in `configs/model.yaml`: explicit `[SEP]`, **OCR text first** so truncation
  eats the caption rather than the claim.
- [ ] **Text-only baseline (MuRIL)** — *2 d* **[A]** — arm builds; needs data + a run
- [ ] **Text-only baseline (IndicBERT)** — *1 d* — pick the better one, report both.
- [ ] **Image-only baseline (ViT)** — *2 d* **[B]** — arm builds; needs data + a run
- [ ] **Image-only baseline (CLIP)** — *1 d* **[B]**
- [~] **Training loop + W&B logging + checkpointing** — *2 d*
  `training/loop.py` — seeded, class-weighted, early-stops on val macro-F1,
  picks the decision threshold on **validation** and freezes it, records the
  split-manifest hash with every run. **W&B is not wired up yet.**
- [~] **Cross-attention fusion model** — *3–4 d* **[A]**
  `models/arms.py` — bidirectional cross-attention plus a concat-fusion
  baseline, because if attention does not beat concatenation it is not earning
  its parameters and that belongs in the report. Untrained and unverified.
- [ ] **Hyperparameter sweep** — *2 d* (W&B sweeps; keep it modest, Colab has limits)

### Evaluation — get the metrics right

- [x] **Report precision, recall, F1, PR-AUC — not accuracy** — *1 d*
  `training/metrics.py`. Accuracy is computed but never headline, and every
  result carries the majority-class baseline next to it — a number only means
  something beside what you would get for free. Cross-checked against sklearn
  in the test suite: two independent implementations agreeing is evidence.
- [x] **Precision–recall curve across thresholds** — `pr_curve` / `pr_auc`,
  step-wise average precision rather than the optimistically biased trapezoid.
- [x] **Confusion matrices per arm** — *0.5 d* — on every `Metrics` object.
- [x] **Run the ablation table: text-only vs image-only vs fused** — *1 d*
  `ablation_table()` prints the delta with its sign and magnitude against the
  text-only baseline, plus fused-minus-best-single-modality, and says
  "within seed noise" when the gap is smaller than the seed spread.
  **The table is built; it has no numbers in it until something is trained.**
- [x] **Statistical significance** — *0.5 d*
  `aggregate()` reports mean ± std over seeds; three seeds are configured by
  default. Nothing gets reported without it.

- [~] **GATE 4 — The targeted experiment** — *1–2 d*
  Build an evaluation subset of posts where **the caption is benign and the claim is image-only**. If fusion doesn't beat text-only *there*, the central thesis is not demonstrated regardless of overall F1.
  *This is the single most convincing experiment in the project.* Budget real time for it.
  **2026-08-15:** the subset filter exists (`is_image_only_claim`) and
  `scripts/07_ablation.py --image-only-claims` runs the experiment on it.
  **One caveat that has to go in the limitations section:** fact-check archives
  do not publish the original post's caption. The closest field is `claim_text`,
  the fact-checker's own rendering of the viral claim, and that is what the
  filter scores. The article *title* is deliberately excluded — it is the
  fact-checker's headline and almost always names the scam outright ("Viral post
  falsely claims SEBI approved this app"), which would make every caption look
  informative and leave this subset permanently empty. So the filter is a proxy
  for a benign caption, not a measurement of one. **Hand-check the members
  before any claim rests on them.**

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
  Tooling is ready: `python scripts/06_budget.py --arm fused --variant teacher
  --from-scratch` measures the architecture without downloading any weights.
- [ ] **Knowledge distillation into a student** — *3–4 d* **[A]**
- [ ] **ONNX export** — *1 d*
- [ ] **Quantisation (INT8) + ONNX Runtime benchmark** — *1 d*
- [ ] **Report the size/latency/accuracy tradeoff table** — *0.5 d*
  This is what turns "lightweight" from an adjective into a result. Check it
  against the targets set in Phase 0 — they are now real numbers in
  `configs/model.yaml`, and `scripts/06_budget.py` prints the pass/fail table.
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

*Phase 1 pipeline is complete. Phase 2 scaffolding (datasets, models, training,
budget) is written and its non-torch parts are tested: 250 assertions passing,
all offline. Everything still unticked above is what remains — and the two that
block everything else are Gate 1 (yield) and Gate 2 (negative class).*
