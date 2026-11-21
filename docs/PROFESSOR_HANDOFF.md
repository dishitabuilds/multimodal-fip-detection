# Promotion Lens demonstration handoff

## What runs

Start from the repository root with `.venv/bin/python -m streamlit run app.py`.
Open **ML Lab**, enable **Show internal pilot models**, and use the three-case
panel. It shows the sample image, OCR, text SVM and fused SVM margins, and a
Needs review state for an ambiguous sample or model disagreement. For a custom
image, use the upload and classification controls below the panel.

The local run requires `data/models/easyocr/`, the MobileNetV3 weights in
`data/models/mobilenet_v3_small/`, and the locally trained joblib files in
`data/processed/lab/internal_smoke_only/visual-source-holdout-20261113/models/`.
These data and model files are ignored by Git. A fresh clone of this repository
cannot run the trained model until those assets are supplied. The cloud
entrypoint prepares OCR and speech, but does not install classifier artifacts.

## Evidence and results

The original-media experiment contains 47 rows. The training split has 31
Newschecker and AMFI examples. Validation has six Newschecker and SBI examples.
The test has five Scamwatch scam screenshots and five iShares issuer factsheet
pages. Source URLs, chosen images, labels, and their rationale are in
`data/interim/fused-holdout-20261113/records.jsonl`. The exact split is in
`split_manifest.json`. The model uses selected-image OCR and the frozen
MobileNetV3 encoder. Article verdicts and archive titles are excluded.

On 10 test images, the fused SVM achieved macro-F1 0.792 and PR-AUC 0.800.
It caught three of five scams with no false alarms. The image-only SVM tied its
macro-F1. The text SVM trained on the same split scored 0.333 macro-F1.
An earlier text SVM scored 0.899 on this set, but five iShares pages had already
appeared in a diagnostic of that earlier model. Individual predictions and
campaign-grouped scores are in `comparison_with_prior_svm.json`.

The test has distinct source publishers but different visual formats by class.
The genuine examples are issuer factsheet pages and the scams are social posts
and fake news pages. The 47 labels were assigned from source evidence by the
assistant without independent human agreement. Hindi and mixed-language
performance have not been measured. This is a functional research prototype,
not a validated fraud screening system.

## Reproduce

With the same prepared dataset, weights and environment:

```zsh
.venv/bin/python scripts/10_train_lab.py --internal-pilot --models linear_svm \
  --records data/interim/fused-holdout-20261113/records.jsonl \
  --manifest data/interim/fused-holdout-20261113/split_manifest.json \
  --experiment-id visual-source-holdout-reproduction
.venv/bin/python data/interim/fused-holdout-20261113/compare.py
```

The second command reads the original experiment ID, so it reproduces its
comparison after that run is available. The exact images, OCR cache, manifests,
weights and artifacts are local ignored files. Share the complete local project
folder when reproducing outside this machine.

## CPU measurement

The three-run local warm-model check on `sample_fake_1.png` measured 5,654 ms
median, 6,371 ms p95 and 3,072 MB peak process memory. The fused classifier
artifact is about 0.078 MB plus 9.83 MB of MobileNet weights. OCR weights add
roughly 285 MB on disk. The check includes OCR and classifier inference and
excludes cold model loading. See `end_to_end_cpu.json` for the hardware and raw
samples. The earlier 100 ms latency and 1,024 MB RAM targets are not met by
this measured workflow.

## Submission scope

The app, sample demonstration, training code, dataset manifest, and per-example
results are ready for a local professor demonstration. Publishing a working
classifier on a fresh hosted instance still requires packaging the ignored
models and compatible Python dependencies. Research claims about generalization
need a larger matched-format dataset and independent review of the labels.
