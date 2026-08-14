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

> **Full documentation:** open `PROJECT_DOCUMENTATION.html` in the project root.
> It is the living record of what has been built, how, and why.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e .                  # installs the fipd package
python tests/run_all.py           # expect: 172 passed, 0 failed
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
```

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
│   ├── datasets/                stage 6: torch views, benchmark loaders  (empty)
│   └── utils/                   HTTP, IO, logging
├── scripts/                     numbered stage CLIs + calibration tool
├── tests/
│   ├── unit/                    library-level
│   ├── integration/             the stage CLIs, end to end
│   └── fixtures/                local mock server replicating the real endpoints
└── docs/                        checklist, Phase 0 guide
```

## Where things stand

- **Phase 1 pipeline: complete and tested.** 172 assertions, all offline.
- **No data collected yet** — that is Phase 0, in `docs/PHASE0_GUIDE.md`.
- **No model code yet**, deliberately. Architecture decisions wait on knowing
  the dataset is viable.

Two open problems are documented in `docs/CHECKLIST.md` and the HTML write-up:
yield (the archives are overwhelmingly political) and class balance (every
record collected is labelled `fake`, because fact-checkers only publish
debunks). Both need answers before Phase 2.

## Collection ethics

Only publicly published fact-check articles, at 1.5–4 s between requests,
honouring `robots.txt`, cached so nothing is fetched twice. These are small
non-profit newsrooms — do not lower the delays. Phase 3 Telegram collection
uses Telethon against the official API on public channels only.
