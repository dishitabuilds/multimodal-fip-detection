# Internal pilot — 25 October 2026

**This is an engineering smoke run, not a validated detection result.** It uses
random tiny encoders and 14 images with provisional AI-assisted labels. It does
not establish research accuracy, fusion benefit, Hindi performance or teacher
retention. The production frontend checkpoint was not replaced.

Validation: all four offline pipeline suites and 152 regression cases passed;
the checkpoint evaluation-only run completed for all three arms.

## What now works end to end

Original local image files -> selected-image OCR -> two-class split -> text,
image and fusion training -> checkpoint save -> evaluation-only checkpoint
reload -> CPU budget measurement.

- Selected inputs: 14 images, proposed 8 fake / 6 real.
- Proposed source basis: investment offers described by the archived fact-checks
  and first-party AMFI campaign creatives. These are AI-assisted judgments,
  not human approvals or a gold-standard dataset.
- Selected files have no fact-check verdict overlay. Article titles, archive
  claim summaries and verdict fields are absent from model text.
- OCR: English EasyOCR; 14/14 images yielded text without runtime errors.
  Recognition quality has not been measured against hand transcripts.
- Split: 7 train (4 fake / 3 real), 4 validation (2 / 2), 3 test (2 / 1).
- Manifest hash: `d046d4364d012316`.
- One epoch, seed 42, batch 4, CPU with four PyTorch threads.
- All three arms trained and saved separate checkpoints. Evaluation-only reload
  completed for all arms without retraining.

The split passes the existing ID/declared-cluster and exact chosen-image checks.
Its campaign groups are still provisional; semantic near-duplicates, class/source
confounding and tiny-sample uncertainty prevent research conclusions. In
particular, AMFI artwork and fake social screenshots can differ by presentation
style. Three test examples cannot establish generalization.

## Local artifacts

These files are ignored local outputs, not bundled with a fresh clone:

- Selection manifest: `data/processed/visual_review/assisted_pilot_manifest.json`.
- Source-preserving pilot: `data/interim/assisted_pilot_v1/records.jsonl`.
- OCR records/cache/report: `data/interim/assisted_pilot_ocr_v1/`.
- Split manifest: `data/interim/assisted_pilot_v1/processed/split_manifest.json`.
- Checkpoints: `data/checkpoints/student/internal-ai-pilot-20261025/`.
- Run reports/predictions: `data/processed/ablation/student/internal-ai-pilot-20261025/`.
- Review queue: `data/processed/assisted_pilot_audit/manual_review.csv`.
- Budget report: `data/processed/internal_smoke_budget/budget_student_fused.json`.

The audit deliberately reports `provisional_ai_labels_not_research_ready`.
Normal research configurations refuse these labels. Only a configuration with
`purpose: internal_smoke_only` can use them; no human-review flag is generated.

## Model and measurements

The opt-in `configs/tiny-smoke.yaml` uses local random DistilBERT/ViT snapshots
created by `scripts/09_tiny_smoke.py`, with a UTF-8 byte tokenizer. English and
Hindi text can be represented; the encoders have no pretrained language or
vision knowledge. The default research configuration is unchanged.

- Fused parameters: **1,068,034**.
- Saved fused checkpoint: **4.111 MiB** (including checkpoint metadata).
- CPU model-only latency: **32.55 ms p50 / 36.86 ms p95** in this pilot run.
- Execution: FP32, batch 1, image 224x224, 256 byte tokens, four CPU threads,
  two warm-ups and only ten timed runs, PyTorch 2.14.1+cpu on local Windows 11.
- Peak RAM: unmeasured because `psutil` is absent.
- Teacher-F1 retention: unmeasured; no teacher comparison was performed.

This short timing run excludes OCR and initialization. It is not the configured
100-run research benchmark, and the measured tiny model is not a validated
compressed detector. Passing its four measured resource rows alone proves
nothing about useful classification quality.

## Reproduce an internal run

From the project root with this local data snapshot available:

```powershell
# Fresh run: generates a new experiment ID and preserves earlier checkpoints.
.\.venv\Scripts\python.exe scripts/09_tiny_smoke.py --threads 4 --run-ablation -- --records data/interim/assisted_pilot_ocr_v1/records.jsonl --manifest data/interim/assisted_pilot_v1/processed/split_manifest.json

# Reload the saved experiment; no training and no model-weight downloads.
.\.venv\Scripts\python.exe scripts/09_tiny_smoke.py --threads 4 --run-ablation -- --records data/interim/assisted_pilot_ocr_v1/records.jsonl --manifest data/interim/assisted_pilot_v1/processed/split_manifest.json --experiment-id internal-ai-pilot-20261025 --evaluate-only

# Measure that saved fused checkpoint under the smoke conditions.
.\.venv\Scripts\python.exe scripts/06_budget.py --config configs/tiny-smoke.yaml --checkpoint data/checkpoints/student/internal-ai-pilot-20261025/fused_seed42.pt --out-dir data/processed/internal_smoke_budget
```

## Research model still to validate

Config-only construction established that the original fused student has
**159,654,530 parameters**, including a 134,734,080-parameter multilingual
DistilBERT backbone. INT8 alone cannot satisfy the 50M parameter target.

The optional `configs/indicbert-deit-tiny.yaml` instead combines shared ALBERT
IndicBERT v1 with DeiT-Tiny. Tests construct its architecture on the meta device
and count **39,858,114 fused parameters**. Official metadata sources are linked
in that configuration and `tests/test_lightweight_candidate.py`.

This is a candidate, not an evaluated replacement. IndicBERT's current Hugging
Face repository requires access; its real tokenizer/weights have not been loaded
here. FP32 weight storage is roughly 152 MiB, above the 100 MiB target. Compression,
accuracy, latency, memory and teacher retention still need real measurements.
