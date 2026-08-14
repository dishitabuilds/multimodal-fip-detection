# A Lightweight Multimodal Model for Detecting Fake Investment Promotions

Minor Project — supervisor Dr. Shivangi Surati.

Detecting fraudulent investment promotions where the deceptive claim sits
**inside the image** (doctored P&L screenshots, truncated-axis charts, pasted
SEBI/RBI logos, borrowed photos of finance personalities) rather than in the
caption, which stays vague and harmless. Text-only classifiers miss these.

| # | Modality | Encoder | Status |
|---|----------|---------|--------|
| 1 | Image | ViT / CLIP | Phase 2 |
| 2 | Text — caption + OCR of in-image text, code-mixed Hi/En | MuRIL / IndicBERT | Phase 2 |
| 3 | Propagation — Telegram forwarding graph | GNN (PyTorch Geometric) | Phase 3 |

Fused with cross-attention. An ablation (text-only vs image-only vs fused)
measures what each modality contributes rather than assuming it.

## What "lightweight" means here

The word in the title is a claim, and a claim without a number cannot be wrong.
These are the targets, and they apply to the **deployed** (student, INT8) model
— the teacher is measured too, but only as the baseline the reduction is quoted
against.

| Budget | Target | Why this number |
|---|---|---|
| Parameters | ≤ 50 M | the largest model that still quantises under 100 MB |
| Size on disk | ≤ 100 MB | fits a free-tier container image and one HTTP download |
| CPU latency p50 | ≤ 100 ms | below the threshold where an upload feels instant |
| CPU latency p95 | ≤ 250 ms | tail latency on a shared CPU is mostly scheduling noise |
| Peak RAM | ≤ 1 GB | the Streamlit Community Cloud limit, where the demo runs |
| macro-F1 retention | ≥ 0.95 × teacher | without this, "fast" is satisfied by a model that is useless |

Measured at batch size 1, 4 threads, 224×224 image, 128 text tokens, median of
100 timed runs after 10 warm-ups, on an Intel Core 7 150U. **A latency figure
without those conditions is not comparable to anything, including itself.**

The targets live in `configs/model.yaml` and `python scripts/06_budget.py` is
the test of them. Missing one is a result, not an embarrassment — it is the
size/latency/accuracy tradeoff the report has to contain either way.

> **Full documentation:** open `PROJECT_DOCUMENTATION.html` in the project root.
> It is the living record of what has been built, how, and why.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e .                  # installs the fipd package
python tests/run_all.py           # expect: 261 passed, 0 failed
```

Optional extras: `pip install -e ".[ocr]"`, `".[model]"`, `".[graph]"`, `".[serve]"`.

## Pipeline

Stages run in order and each is independently resumable.

```bash
python scripts/01_collect.py                 # harvest fact-check archives
python scripts/02_enrich_images.py           # fetch creatives for API-found URLs
python scripts/03_ocr.py --backend easyocr   # extract in-image text
python scripts/calibrate_dedup.py            # tune the duplicate threshold on real data
python scripts/04_curate.py                  # filter, label, deduplicate
python scripts/05_split.py                   # freeze leakage-free splits
python scripts/06_budget.py --targets-only   # the lightweight budget
python scripts/07_ablation.py                # text-only vs image-only vs fused
```

Stage 01 obeys `robots.txt` and, if a source returns nothing at all, retries it
through the `fallback` route in `configs/sources.yaml` automatically — several
archives block their REST API but leave their sitemaps open.

## Layout

```
├── PROJECT_DOCUMENTATION.html   <- full write-up, open in a browser
├── pyproject.toml               installable package, optional extras per phase
├── configs/                     all tunable parameters, no magic numbers in code
├── src/fipd/
│   ├── schema/                  FactCheckRecord, ImageAsset — the shared contract
│   ├── collection/              stage 1: WordPress, sitemap+ClaimReview, Google API
│   ├── enrichment/              stage 2/3: OCR backends, transliteration
│   ├── curation/                stage 4/5: finance filter, labels, dedup, splits
│   ├── datasets/                stage 6: records -> examples, torch dataset
│   ├── models/                  encoders, cross-attention fusion, arms, budget
│   ├── training/                metrics, training loop, ablation
│   └── utils/                   HTTP, IO, logging
├── scripts/                     numbered stage CLIs + calibration tool
├── tests/
│   ├── unit/                    library-level
│   ├── integration/             the stage CLIs, end to end
│   └── fixtures/                local mock server replicating the real endpoints
└── docs/                        checklist, Phase 0 guide
```

## Where things stand

- **Phase 1 pipeline: complete and tested.** 261 assertions, all offline.
- **First real collection done** (2026-08-15): 1208 records from 5 archives.
  After filtering and labelling, **52 usable items — a 4.3% end-to-end yield**,
  every one of them `fake`. Numbers and caveats in `docs/CHECKLIST.md`.
- **Model code: scaffolding written, nothing trained.** Encoders, cross-attention
  fusion, the three ablation arms, metrics and the budget all exist; the parts
  that do not need torch are tested. No training has happened, because there is
  no negative class yet.

Three open problems, in the order they block things:

1. **Class balance.** Every record collected is `fake` — fact-checkers only
   publish debunks. No classifier can be trained until this is answered
   (Gate 2). It is the single blocking issue.
2. **Yield.** The archives are overwhelmingly political. 4.3% end to end, and
   the largest source (Factly) publishes no machine-readable verdict at all.
3. **Nothing is verified against real tensors.** torch is not installed on the
   collection machine, so the arms build in principle only.

## Collection ethics

Only publicly published fact-check articles, at 1.5–4 s between requests,
cached so nothing is fetched twice. `robots.txt` is fetched once per host and
obeyed, and a `Crawl-delay` longer than ours replaces ours (never shortens it).
These are small non-profit newsrooms — do not lower the delays. Phase 3 Telegram
collection uses Telethon against the official API on public channels only.
