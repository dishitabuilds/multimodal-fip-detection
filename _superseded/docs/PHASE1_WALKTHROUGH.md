# Phase 1 Walkthrough — What Was Built, How, and Why

**Project:** A Lightweight Multimodal Model for Detecting Fake Investment Promotions
**Phase:** 1 — Data collection and labelling pipeline
**Date:** 13 August 2026
**Status:** Complete and tested. No model code yet, by design.

This document is a full account of the work: what was investigated, what was
decided, how each piece was built, how it was verified, and what the results
imply for the rest of the project. It is written so that a teammate can pick
the repo up cold, and so that the reasoning can be reused directly in the
project report.

---

## Table of contents

1. [Starting point and the decision made](#1-starting-point-and-the-decision-made)
2. [Investigation: what the data sources actually look like](#2-investigation-what-the-data-sources-actually-look-like)
3. [Architecture decisions and their justification](#3-architecture-decisions-and-their-justification)
4. [File-by-file walkthrough](#4-file-by-file-walkthrough)
5. [How it was tested without a network](#5-how-it-was-tested-without-a-network)
6. [Results: what the investigation proved](#6-results-what-the-investigation-proved)
7. [Two unsolved problems this exposed](#7-two-unsolved-problems-this-exposed)
8. [What to do next](#8-what-to-do-next)
9. [Appendix: quick reference](#9-appendix-quick-reference)

---

## 1. Starting point and the decision made

The project folder was empty. Nothing existed — no code, no data, no repo.

Three plausible starting points:

| Option | Argument for | Argument against |
|---|---|---|
| Data collection | Slowest component to mature; everything else depends on it | Least visible progress |
| Repo scaffolding + roadmap | Organises the work | Produces no actual capability |
| Model architecture | Most interesting; can train on Fin-Fact as a placeholder | Risks building a model for a dataset that turns out not to exist |

**Data collection was chosen**, for one reason: a multimodal model is only as
good as the labelled image–text pairs behind it, and until we know how many of
those actually exist, every architecture decision downstream is speculation.
Building the encoder stack first would mean discovering a data shortfall late,
when there is no time left to fix it.

That judgement turned out to be correct — Section 6 shows the investigation
surfaced two problems that would have invalidated a model built first.

---

## 2. Investigation: what the data sources actually look like

Before writing any scraper, each proposed source was probed directly to find
out what it actually serves. This mattered more than expected — several
assumptions in the original proposal did not survive contact with the sites.

### 2.1 Method

Each candidate site was queried for:

1. `robots.txt` — what crawling is permitted
2. A WordPress REST endpoint (`/wp-json/wp/v2/posts`) — is there a structured API?
3. A sitemap — is there a clean URL discovery path?
4. A live article page — what structured data is embedded?
5. A keyword search for finance terms — what is the actual yield?

### 2.2 What each source returned

**Alt News** — WordPress REST API is open and works. Full post objects with
`title`, `content`, `excerpt`, `categories`, `tags`, and importantly
`jetpack_featured_media_url`, which carries the hero image directly.

Querying the category list gave the single most important number in this
investigation:

| Category | Post count |
|---|---|
| Politics | 4,157 |
| News | 934 |
| Religion | 645 |
| Society | 280 |
| Media analysis | 256 |
| Science | 94 |
| Compilation | 32 |
| Technology | 26 |
| History | 23 |
| Sports | 11 |
| **Economics (slug: `finance`)** | **11** |
| Education | 5 |

Eleven finance posts against 4,157 political ones. A follow-up search for
`investment` across the whole archive returned **three** articles, none of them
about investment fraud. Alt News is essentially a political fact-checker.

**Newschecker** — WordPress REST API open and works. A search for `investment`
returned 10 posts, of which 5 were directly on-target:

- "Scam Alert! Video Of Rahul Gandhi Promoting Investment Scheme Is Fake"
- "Scam Alert: Nirmala Sitharaman Did Not Promote This AI-Powered Investment Scheme, Viral Video Is A Deepfake"
- "Viral Video Of S Jaishankar Promoting Dubious Investment Opportunity Is Deepfake"
- "Fake Indian Express Report On Ajit Pawar Entrusting AI-Based Investment Platform To Family Goes Viral"
- "Did Amit Shah Announce This Financial Scheme During A Rally? No, Viral Video Is Doctored"

Two useful signals here. First, Newschecker uses a consistent **"Scam Alert!"**
title prefix for financial fraud, which is a free high-precision filter.
Second, the dominant 2024–26 fraud pattern is clearly **deepfake celebrity
endorsement** — a public figure appears to promote an investment platform. That
is squarely a multimodal problem: the caption is innocuous, the deception is in
the video frame. It is good evidence for the project's core premise and worth
citing in the report.

**BOOM Live** — not WordPress. The Quintype API path (`/api/v1/stories`)
returned 404, so the platform has moved on. But `robots.txt` disclosed three
sitemaps, and `sitemap-daily.xml` parsed cleanly with `<lastmod>` timestamps on
every entry. Article URLs follow `/<section>/<slug>-<numeric-id>`.

Crucially, inspecting BOOM's navigation revealed **a dedicated `ScamCheck`
vertical** plus a `fact-check/business` sub-section covering investment fraud,
fake celebrity endorsements, crypto scams and fraudulent trading apps. BOOM is
by a wide margin the strongest single source for this project — and this was
not in the original proposal, which treated all four fact-checkers as roughly
equivalent.

**Factly** — the WordPress REST API returned **HTTP 403**. This is a Cloudflare
edge rule, not a site-level block; it commonly rejects datacentre IP ranges
while allowing ordinary residential browsers through. It will likely work from
a normal home connection. The config therefore carries a `fallback` block that
switches Factly to sitemap-based scraping if the API stays blocked.

**Vishvas News** — rejected at the proxy layer during testing, same category of
problem. Included in the config; verify from your own machine.

### 2.3 The find that changed the design

While researching how BOOM embeds structured data, a far better approach
surfaced: the **Google Fact Check Tools API**.

Every serious fact-checker publishes [schema.org `ClaimReview`](https://schema.org/ClaimReview)
markup — the structured data Google reads to build its fact-check panels in
search results. Google indexes all of it and exposes it through a free public
API.

That single endpoint returns, for every publisher at once:

- the claim text as stated
- the publisher's own verdict string (`"False"`, `"Misleading"`, `"झूठ"`)
- the article URL
- the language code
- the review date and the publisher's identity

This inverts the original data plan. Instead of five brittle HTML parsers that
break whenever a site is redesigned, there is **one stable query surface** for
discovery, with the direct scrapers demoted to a fallback role and to the job
the API cannot do — fetching the images.

---

## 3. Architecture decisions and their justification

Each of these is a defensible design choice worth stating explicitly in the
report, because each had a plausible alternative that was rejected for a reason.

### 3.1 One record schema, filled by many scrapers

Every scraper normalises into a single `FactCheckRecord` dataclass. Downstream
code — the finance filter, the label mapper, OCR, and eventually the model —
never needs to know which site an item came from.

**Why it matters:** adding a sixth source later means writing one scraper class,
not touching the filtering or training code at all.

### 3.2 Google Fact Check API as the primary discovery layer

**Alternative rejected:** crawl each fact-checker's HTML independently.

**Why:** five HTML parsers is five things that silently break on redesign. The
ClaimReview index is a standardised format that publishers maintain themselves
because Google search visibility depends on it — they have a strong incentive
to keep it accurate, which is exactly the property you want in a label source.

**Cost:** the API does not return images. That is why `enrich_images.py` exists
as a second stage.

### 3.3 "Misleading" maps to FAKE, not to a third class

**Alternative rejected:** a three-class problem — true / misleading / false.

**Why:** a chart with a truncated axis is factually "partly true" and is still a
fraudulent promotion. For a fraud detector, the operative question is *does this
post deceive*, not *is every pixel of it false*. Collapsing to binary also keeps
the class counts usable at the sample sizes this dataset is likely to reach.

Verdicts that genuinely cannot be placed (`"Unverified"`, `"Research Ongoing"`,
empty) map to `unknown` and are **held out of training rather than guessed at**.
Guessing at them would inject label noise that no amount of model capacity can
recover from.

### 3.4 The finance filter is a hand-weighted lexicon, not a classifier

**Alternative rejected:** train a small text classifier to identify finance
content.

**Why two reasons:** at collection time there are no labels to train it on —
that is circular. And a transparent rule can be inspected, tuned and defended
in a viva, where a black box cannot. Every weight in `finance_filter.py` is
visible and editable.

**How it works:** terms carry weights (3.0 strong, 1.5 medium, 0.7 weak) and
fields carry multipliers (title ×2.5, claim ×2.0, image text ×1.5, body ×1.0).
A term in the headline counts far more than a passing mention in paragraph nine.

**The negative lexicon is the important part.** Political stories mention
"crore", "scam" and "fraud" constantly and would flood the dataset. Terms like
`communal`, `election`, `temple`, `mosque` subtract from the score when they
appear in the title or claim. In testing this correctly rejected a
communal-riot article that mentioned "crores of views" — it scored **−4.0**,
well below the 4.0 keep threshold, while a genuine SEBI trading-app scam scored
**70.15**. That separation is wide enough to be robust.

The lexicon covers three writing systems deliberately: English, Devanagari
Hindi (`निवेश घोटाला`, `पैसा डबल`), and **romanised Hinglish** (`paisa double`,
`guaranteed profit`, `SEBI registered`) — because scam creatives are routinely
written in Latin-script Hindi, and a Devanagari-only lexicon would miss them
entirely. This directly supports the project's stated need for a multilingual
encoder.

### 3.5 Image hygiene — the most likely source of a fake result

This is the subtlest decision and the one most worth explaining to your guide.

**The trap:** fact-checkers stamp their own verdict graphics onto the images
they debunk — a red "FALSE" banner, a watermark, a logo overlay. If those images
go into training, **the model learns to detect the watermark, not the fraud**.
It will report excellent accuracy and be completely worthless on real posts.

Three defences were built:

1. **Blocklist** — logos, avatars, share icons, tracking pixels, spacers are
   dropped by URL pattern.
2. **Annotation tagging** — `ImageAsset.role` is set to `annotated` when the
   URL, alt text or caption matches verdict-stamp patterns, so those images can
   be excluded or inspected separately rather than silently entering the set.
3. **Article-body scoping** — image extraction is confined to the article
   element, so "related stories" sidebar thumbnails never enter the record.

**None of this is sufficient.** A manual spot-check of a sample before training
is mandatory, and this is flagged in the README as well. Automated hygiene
reduces the problem; it does not eliminate it.

### 3.6 Resumability and caching as first-class features

Every run is resumable: existing `uid` values are read from the output JSONL and
skipped, and HTTP responses are cached to disk so re-parsing never re-fetches.

**Why it matters practically:** collection runs take hours at polite rates. A
dropped connection at hour three must not cost you hours one and two. It also
means you can iterate on the *parsing* logic against cached responses without
touching the network at all.

### 3.7 Collection ethics

Requests are spaced 1.5–4 seconds, `robots.txt` is honoured, and responses are
cached so nothing is fetched twice. These are small non-profit newsrooms
running on donations — the delays are deliberate and should not be lowered.
Only publicly published fact-check articles are collected.

---

## 4. File-by-file walkthrough

23 files, 2,164 lines of Python.

```
Minor Project/
├── README.md                     project overview, setup, roadmap
├── requirements.txt              phased — only Phase 1 deps uncommented
├── .gitignore                    excludes data/, secrets, .session files
├── configs/
│   └── sources.yaml              the single place you edit to change sources
├── src/
│   ├── collection/
│   │   ├── schema.py             FactCheckRecord, ImageAsset, label constants
│   │   ├── base.py               BaseScraper — shared machinery
│   │   ├── wordpress.py          WordPressScraper — WP REST API
│   │   ├── article.py            ArticleScraper — sitemap + ClaimReview
│   │   ├── googlefactcheck.py    GoogleFactCheckScraper — the API discovery layer
│   │   └── registry.py           YAML config -> scraper instances
│   ├── filtering/
│   │   ├── finance_filter.py     weighted lexicon relevance scoring
│   │   └── labels.py             verdict string -> fake/real/unknown
│   └── utils/
│       └── http.py               PoliteSession — rate limit, retry, cache
├── scripts/
│   ├── collect_factchecks.py     stage 1
│   ├── enrich_images.py          stage 1b
│   └── build_dataset.py          stage 2
├── tests/
│   ├── mock_server.py            offline replica of the real endpoints
│   └── test_pipeline.py          56 assertions
└── docs/
    └── PHASE1_WALKTHROUGH.md     this file
```

### `src/collection/schema.py`

Defines the two dataclasses everything else speaks in.

`FactCheckRecord` carries provenance (`source`, `source_id`, `url`), content
(`title`, `body_text`, `claim_text`, `verdict_raw`), derived fields (`label`,
`is_finance`, `finance_score`, `finance_hits`), a list of `ImageAsset`, and a
`raw` dict holding the untouched source payload so nothing is lost.

The `uid` property is a SHA-1 of `source:source_id`, truncated to 16 chars. This
is what makes runs resumable — it is stable across re-runs and unique across
sources.

`ImageAsset` carries `url`, `local_path`, `role`, dimensions, `sha256`,
`alt_text`, `caption`, and an `ocr_text` slot that the OCR stage will fill.

### `src/utils/http.py`

`PoliteSession` wraps `requests` with four behaviours:

- **Rate limiting** — randomised 1.5–3 s gap, measured from the last request so
  slow responses do not stack extra delay on top.
- **Retry with backoff** — `urllib3.Retry` on 429/500/502/503/504, honouring
  `Retry-After`.
- **Disk caching** — responses keyed by SHA-256 of the URL. `get_json` also
  caches headers separately, because WordPress returns the pagination total in
  `X-WP-TotalPages` and losing that on a cache hit would break pagination.
- **Streaming downloads** — images written in 64 KB chunks, skipping files that
  already exist.

### `src/collection/base.py`

`BaseScraper` holds everything the three scrapers share.

`clean_html()` strips scripts, styles, iframes and forms, then collapses
whitespace.

`extract_images()` is the most intricate method, because real-world fact-check
pages are messy:

- reads `data-src`, `data-lazy-src`, `data-original` and `src` in order, so
  lazy-loaded images are not missed
- parses `srcset` and picks the **widest** declared variant, because OCR needs
  legible text and the default `src` is often a 300 px thumbnail
- resolves relative and protocol-relative URLs against the page URL
- drops `data:` URIs and blocklisted patterns
- captures `alt` text and `<figcaption>` — both are useful weak signals for the
  finance filter and for manual review
- tags suspected verdict stamps as `role="annotated"`

`download_images()` writes each image as `<uid>_<url-hash>.<ext>`, drops files
under 4 KB (spacers and error pages), and records a SHA-256 so exact duplicates
can be detected later — fact-checkers reuse the same stock art across dozens of
articles.

`run()` is the resumable driver: reads existing `uid`s, skips them, appends new
records to JSONL, flushing after each one so an interrupt loses at most one
record.

### `src/collection/wordpress.py`

Handles any WordPress-backed fact-checker. Two modes: `search` (keyword query,
best for extracting finance content from a mostly-political archive) and
`browse` (walk a category in date order).

It resolves category slugs to IDs by fetching `/wp-json/wp/v2/categories` once
and caching it, unescapes HTML entities in titles (WordPress returns
`&#8216;` for curly quotes), and inserts `jetpack_featured_media_url` at
position 0 of the image list — on fact-check sites the hero image is very often
the viral creative itself, so ordering by prominence is meaningful.

Uses `_fields` to request only needed fields, which materially reduces payload
size across thousands of posts.

### `src/collection/article.py`

For sites without a usable API. Two stages.

**Discovery** parses XML sitemaps, recursing one level into sitemap indexes,
and filters URLs by path fragment — this is what lets BOOM's config keep only
`/fact-check/business` and `/scamcheck` while discarding entertainment and
sports.

**Parsing** prefers `ClaimReview` JSON-LD, which yields claim text, the verdict
string, the rating scale and the publication date in machine-readable form. It
walks `@graph` arrays too, since many sites nest their structured data there.
It falls back to `NewsArticle` JSON-LD, then to Open Graph tags, then to the
`<title>` element.

Before extracting images it narrows to the article body — via a configured CSS
selector, or `<article>`, or a class matching `story-body`/`article-content`/
`post-detail` — so sidebar thumbnails stay out.

### `src/collection/googlefactcheck.py`

Queries `factchecktools.googleapis.com/v1alpha1/claims:search` across a list of
finance queries in English and Hindi, pages through results, and filters to a
whitelist of Indian publisher domains (the index is global and mostly US
politics).

One API "claim" can carry several publisher reviews — if BOOM and Factly both
debunked the same viral post, both appear. Each is emitted as its own record,
which is correct: they are two independent labelled items with different
articles and different images.

It overrides `run()` to force `with_images=False`, because this stage produces
URLs and labels only.

**Requires a free API key.** Google Cloud Console → enable "Fact Check Tools
API" → create an API key → `setx FACTCHECK_API_KEY "your-key"`. The class warns
and yields nothing if the key is absent, rather than failing obscurely.

### `src/filtering/finance_filter.py`

The lexicon, organised into groups: regulator/registration claims (SEBI, RBI,
AMFI — the classic legitimacy prop), investment-scam vocabulary (ponzi, money
doubling, guaranteed return, `पैसा डबल`), crypto and forex, deepfake celebrity
endorsement, general finance, and weak money words. Plus the negative lexicon
described in §3.4.

`normalise()` applies NFKC Unicode normalisation and strips punctuation while
preserving the Devanagari range `ऀ-ॿ`.

`DISCOVERY_QUERIES` — the 20 queries fed to the Google API — lives here too, so
the vocabulary of the project sits in one file.

### `src/filtering/labels.py`

An ordered rule list mapping publisher verdict strings to `fake` / `real` /
`unknown`. Order matters: `"Mostly True"` must be checked before the generic
false patterns. Covers English and Devanagari.

`label_from_title()` is the fallback for records with no machine-readable
verdict, exploiting fact-check title conventions — `"No, ..."`, `"Scam Alert"`,
`"...is fake"`, `"falsely shared"` → fake; `"Yes, ..."`, `"...is authentic"` →
real.

`assign_label()` tries the verdict first, the title second, `unknown` last.

### `scripts/collect_factchecks.py`

Stage 1 CLI. Iterates enabled sources from the config, builds each scraper via
the registry, and runs it. A failing source is logged and skipped rather than
killing the run — one Cloudflare block should not cost you the other four
sources. Prints a summary table at the end.

### `scripts/enrich_images.py`

Stage 1b. Reads the Google API discovery records, visits each article URL, and
extracts the images. Maintains one `ArticleScraper` per host so rate limiting
and image folders stay per-site.

Discovery metadata wins on merge — the API's verdict string is cleaner than
anything scraped off the page. `--finance-only` skips fetching images for
records the filter rejects, which is a large bandwidth saving.

### `scripts/build_dataset.py`

Stage 2. Reads all raw JSONL, applies the finance filter and label mapper, and
writes `dataset.jsonl` (everything, annotated), `finance.jsonl` (the training
candidate set) and `stats.json`.

Prints the yield table — raw / finance / labelled / with-image / kept per source
— **and the majority-class baseline**, which is the number that tells you
immediately whether the dataset is trainable. `--require-image` enforces that
every kept record has a downloaded image, which matters for a multimodal model.

### `configs/sources.yaml`

The single file you edit to change what gets collected: which sources are
enabled, per-source search terms, page limits, rate limits, path filters, and
the finance threshold. Adding a source requires no code change if it is
WordPress or sitemap-based.

---

## 5. How it was tested without a network

The development sandbox could not reach the live sites directly, and testing
scrapers against production sites during development is bad practice anyway —
it hammers servers that have done nothing to deserve it, and the tests become
non-deterministic when a site changes.

**Solution: a local mock server that replicates the real response shapes.**

`tests/mock_server.py` runs on `127.0.0.1:8765` and serves:

- `/wp-json/wp/v2/posts` with real pagination behaviour, `search` and
  `categories` filtering, and the genuine `X-WP-Total` / `X-WP-TotalPages`
  headers
- `/wp-json/wp/v2/categories`
- `/sitemap-daily.xml` with `<lastmod>`, including one URL that must be filtered
  out
- Article pages carrying real `ClaimReview` and `NewsArticle` JSON-LD, a header
  logo, a body creative with a caption, and a sidebar "related" thumbnail
- Valid JPEG bytes at `/img/*`

The field names and header names mirror what `altnews.in` and `newschecker.in`
actually return, verified during the investigation phase. A scraper that passes
here exercises the same code paths it will use in production.

The fixture posts are deliberately adversarial:

- one SEBI trading-app scam with a lazy-loaded image, a `srcset` where the
  correct answer is *not* the default `src`, a `<figcaption>`, a site logo and a
  verdict stamp
- one deepfake celebrity endorsement
- one communal-riot article containing the word "crores" — the false-positive
  trap the negative lexicon exists to catch

`tests/test_pipeline.py` runs **56 assertions** across six groups: WordPress
scraping, image download and dedup, sitemap + ClaimReview parsing, the finance
filter (including Devanagari and romanised Hinglish), verdict→label mapping
across 14 verdict strings, and persistence/resume/round-trip.

```
  56 passed, 0 failed
```

Both CLI scripts were additionally run end-to-end against the mock: 5 records
collected, 12 images downloaded, 4 records correctly kept as labelled finance
items, the communal article correctly rejected, and a second run confirmed as
idempotent.

Because the mock is local, the whole suite runs in seconds with no network and
no load on anyone's servers.

---

## 6. Results: what the investigation proved

Three findings that change the project plan:

**BOOM Live is the primary source, not one of four equals.** Its `ScamCheck`
vertical and `fact-check/business` section are the only place in the Indian
fact-check landscape with concentrated investment-fraud coverage. The config
prioritises it accordingly.

**Alt News is nearly useless for this topic.** 11 finance posts against 4,157
political ones; three results for "investment". It stays in the config for
coverage and completeness, but expecting volume from it would be a mistake.

**The dominant fraud pattern is deepfake celebrity endorsement.** Every
on-target Newschecker result was a public figure appearing to promote an
investment platform. This is strong empirical support for the multimodal
premise — the caption in these posts is innocuous and the deception lives
entirely in the video frame. Worth citing in the report's motivation section.

---

## 7. Two unsolved problems this exposed

These are the honest limitations. Both are better raised with your guide now
than discovered in the final month.

### Problem 1 — yield

The archives may not contain enough finance items to train on. Run
`build_dataset.py` and read the yield table before committing to the current
phasing.

**If the finance count lands in the low hundreds**, the fact-check archives are
a **seed and evaluation set**, not a training set — and Telegram collection has
to move from Phase 3 up to Phase 1, because Telethon on public channels is then
the only route to training-scale data.

This is a scheduling risk, not a fatal one, but it needs deciding early.

### Problem 2 — class balance (the serious one)

**Every record this pipeline produces is labelled `fake`.** Fact-checking
archives exist to document falsehoods; they do not publish articles confirming
that ordinary legitimate advertisements are legitimate.

A dataset with one class cannot train a binary classifier. The test run made
this concrete: `Label balance: {'fake': 4}`, `Majority-class baseline: 100.0%`.

Legitimate investment promotions must be sourced separately. Candidates:

- **SEBI-registered investment advisors' public posts** — the registry of
  registered IAs and RAs is public, so the label is externally defensible
- **AMFI "Mutual Funds Sahi Hai"** campaign creatives
- **Registered broker marketing** — Zerodha, Groww, Upstox and similar
- **RBI and SEBI investor-education material**
- **Legitimate financial news graphics** — as hard negatives that resemble the
  fake charts but are not fraudulent

**The critical constraint:** the negatives must be *the same genre of image* —
promotional finance graphics — differing only in whether they deceive. A model
trained on "scam creatives vs. random stock photos" will report 97% accuracy
and have learned nothing about fraud. It will have learned to recognise a
screenshot.

This is the single highest-risk decision in the project. It matters more than
the choice of image encoder, the fusion mechanism, or anything else in Phase 2.

---

## 8. What to do next

In priority order:

1. **Get a Google Fact Check Tools API key** and run a real collection. Read the
   yield table. This is the go/no-go for the current phasing. *(~1 hour)*
2. **Decide the negative-class strategy** with your guide. Problem 2 has to be
   settled before any training. *(discussion + ~1 week collection)*
3. **Build the OCR stage** — PaddleOCR or EasyOCR over `data/images/`, writing
   into `ImageAsset.ocr_text`, with `indic-transliteration` for romanised Hindi.
   The `ocr_text` field is already in the schema and already read by the finance
   filter. *(~2–3 days)*
4. **Manual verification pass** in Label Studio over a sample — specifically to
   check for verdict-stamp leakage (§3.5). *(~1 day)*
5. **Then** Phase 2: MuRIL + ViT/CLIP with cross-attention fusion, and the
   ablation.

Steps 1 and 2 should happen before anything else. Both can invalidate work done
downstream.

---

## 9. Appendix: quick reference

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tests/test_pipeline.py        # expect: 56 passed, 0 failed
setx FACTCHECK_API_KEY "your-key"    # optional but recommended
```

### Commands

```bash
python scripts/collect_factchecks.py --limit 10 --no-images   # smoke test
python scripts/collect_factchecks.py --source boomlive        # one source
python scripts/collect_factchecks.py                          # full run
python scripts/enrich_images.py --finance-only                # fetch creatives
python scripts/build_dataset.py                               # filter + label
python scripts/build_dataset.py --threshold 3.0               # loosen filter
python scripts/build_dataset.py --require-image               # multimodal-ready only
```

### Outputs

| Path | Contents |
|---|---|
| `data/raw/<source>.jsonl` | one record per line, as collected |
| `data/images/<source>/` | downloaded creatives, `<uid>_<hash>.jpg` |
| `data/interim/dataset.jsonl` | all records with labels and scores filled in |
| `data/interim/finance.jsonl` | finance-only, labelled — the training candidate |
| `data/interim/stats.json` | counts for the report |
| `data/.httpcache/` | cached responses; delete to force a refetch |

### Tuning knobs

| What | Where |
|---|---|
| Which sources run | `configs/sources.yaml` → `enabled` |
| Search terms per source | `configs/sources.yaml` → `search` |
| Finance sensitivity | `--threshold` (lower = higher recall) |
| Lexicon terms and weights | `src/filtering/finance_filter.py` |
| Verdict mappings | `src/filtering/labels.py` → `_RULES` |
| Rate limits | `configs/sources.yaml` → `min_delay` / `max_delay` |

### Common issues

| Symptom | Cause and fix |
|---|---|
| `googlefactcheck` yields 0 records | `FACTCHECK_API_KEY` not set; restart the terminal after `setx` |
| Factly returns HTTP 403 | Cloudflare. Switch `kind: wordpress` → `article` and use the `fallback` sitemap block |
| Collection seems to re-fetch everything | `data/.httpcache/` was deleted |
| Yield table shows lots of `unknown` labels | The source emits no ClaimReview rating; check `verdict_raw` in the raw JSONL |
| Re-running adds duplicate records | Should not happen — `uid` dedup is tested. Check the JSONL is not corrupted mid-line |

---

*Phase 1 complete. 23 files, 2,164 lines, 56 tests passing.*
