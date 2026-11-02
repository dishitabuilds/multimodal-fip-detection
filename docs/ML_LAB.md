# Machine learning lab implementation

This track implements the lab recommendation alongside the existing transformer
research code. It trains a Linear SVM using TF-IDF text features, MobileNetV3-Small
image features, or both. Logistic Regression is compared on all three feature
sets; Multinomial Naive Bayes is compared on text only. This produces seven
comparisons using one common held-out cohort.

For correct facts presented in a potentially misleading way, use the separate
[Promotion Audit](PROMOTION_AUDIT.md). It explains selected wording and records
reviewer-supplied claim evidence without changing the binary model's classes.
[Framing annotations and multi-label SVM training](FRAMING_LABELS.md) are a separate
learning task that requires its own human-reviewed labels.

## Install and open

Run from the project root in PowerShell, reusing the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[lab,frontend,dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Choose **ML Lab** in Streamlit's page navigation. Research models appear only
after a reviewed-data run. To try the saved local engineering models, enable
**Show internal pilot models**. Choose a trained model, upload a promotion image,
click **Read image text** (or paste an exact transcript), add its original caption
if available, and click **Classify promotion**. Text models can also run from
transcribed text without an image. Image/fused models require an image.

SVM output is a signed decision margin: zero or greater selects `fake`. It is
not a percentage confidence. Logistic Regression and Naive Bayes produce class
probabilities that have not been independently calibrated.

## Train with reviewed data

The historical curated corpus has no reviewed real-class examples. Complete the
review/import/OCR/curation/group-splitting steps in [DATASET.md](../DATASET.md)
first. Training requires attributed manual review for every input record, current
split metadata, and both classes in train, validation and test. No script assigns
human approvals or treats an unknown example as legitimate.

```powershell
# The text baseline needs no image-model download.
.\.venv\Scripts\python.exe scripts/10_train_lab.py --arms text_only

# Explicit one-time preparation: official, hash-checked pretrained weights (~9.8 MB).
.\.venv\Scripts\python.exe scripts/10_train_lab.py --prepare-image-model

# Train all seven comparisons; image extraction and inference then work offline.
.\.venv\Scripts\python.exe scripts/10_train_lab.py
```

Use `--records PATH --manifest PATH --data-dir data` to select a different prepared
corpus. Image paths are resolved against `--data-dir`. Configuration lives in
[configs/lab.yaml](../configs/lab.yaml). For SVM only, pass `--models linear_svm`.
`--arms text_only image_only fused` selects the three feature experiments.
Research runs and internal pilots are stored in separate purpose directories.

## Reproduce the local engineering pilot

On 30 October 2026 (India time), all seven comparisons trained and reloaded using
the existing 14-image provisional pilot: 7 train, 4 validation, 3 test. MobileNet
uses pretrained ImageNet features; the classical classifiers train on the pilot.
These are engineering checks, not validated fraud-detection results. The pilot's
source/class confounding, provisional labels and tiny split prevent research
accuracy claims. Raw data and saved model files remain local ignored artifacts.

```powershell
.\.venv\Scripts\python.exe scripts/10_train_lab.py --internal-pilot --records data/interim/assisted_pilot_ocr_v1/records.jsonl --manifest data/interim/assisted_pilot_v1/processed/split_manifest.json
```

The completed initial run is under
`data/processed/lab/internal_smoke_only/lab-pilot-20261030/`. Repeat commands
generate new experiment IDs. An existing experiment is never overwritten.

## Prediction from the terminal

Only load joblib files generated locally by this project; joblib deserialization
uses pickle. Use the same environment/scikit-learn version used for training.

```powershell
# A text-only pilot example. Remove --internal-pilot for a reviewed research model.
.\.venv\Scripts\python.exe scripts/11_predict_lab.py --artifact data/processed/lab/internal_smoke_only/lab-pilot-20261030/models/text_only_linear_svm.joblib --ocr-text "Guaranteed double returns. Join our trading group." --internal-pilot

# Add --image PATH for image/fused models; --ocr runs prepared local EasyOCR.
.\.venv\Scripts\python.exe scripts/11_predict_lab.py --artifact data/processed/lab/internal_smoke_only/lab-pilot-20261030/models/fused_linear_svm.joblib --image test_samples/sample_fake_1.png --ocr --internal-pilot
```

The CLI and UI never download OCR/image weights during prediction. Multimodal
artifacts check that the image encoder hash matches the training run. Missing
inputs, incompatible artifacts and provisional artifacts in research mode fail
with an explanation.

## What the implementation learns

1. OCR text and the original caption are assembled using the same configuration
   at training and inference. Archive titles are excluded. Selected-image OCR
   excludes fact-check overlays and other unselected images.
2. Word unigrams/bigrams and character n-grams become TF-IDF features. Character
   features retain Hindi combining marks and can tolerate some OCR variation;
   this does not establish Hindi accuracy. Vocabulary and IDF are fitted on train
   only. The two text branches are normalized together.
3. Frozen pretrained MobileNetV3-Small produces 576 image features. Its official
   resize/crop/normalization is reused at inference. Image standardization is
   fitted on train only, followed by per-image L2 normalization.
4. Fusion concatenates normalized text and image features. The combined vector
   is L2-normalized. Only the classical classifier is trained; MobileNet stays
   frozen. This is feature fusion, rather than cross-attention or distillation.
5. SVM/LR regularization `C` and NB smoothing `alpha` are selected by validation
   macro-F1. Each candidate is fitted on train only. Native decision thresholds
   remain zero for SVM and 0.5 for LR/NB. Test metrics do not choose the model.

All selected arms use the same examples. The runner rejects stale split manifests,
claim clusters spanning splits, reused exact image bytes across splits, missing
classes, and malformed records. Semantic duplicates and source shortcuts still
require manual dataset review. Text-only runs can include examples without images.

## Files for your evaluation

Each completed experiment contains:

- `experiment.json`: purpose, configuration, versions, split/cohort hashes, counts,
  excluded UIDs and the recommended artifact selected using validation results.
- `comparison.csv`: precision, recall, F1, macro-F1, PR-AUC, ROC-AUC, accuracy,
  feature counts, training timings, artifact size and CPU latency for each model.
- `evaluations/*.json`: validation candidates, test predictions, confusion counts
  (TN/FP/FN/TP), saved-artifact hashes and verified reload results.
- `models/*.joblib`: classifier plus fitted TF-IDF/scaler and text configuration.

Latency measures a decoded single image/text input through fresh feature
extraction and classifier inference after warm-up. It excludes OCR, disk decoding
and model loading. The image weights are shared but must be counted alongside the
classifier artifact when reporting deployment size. No total RAM or end-to-end
OCR latency claim is made by this runner.

For the viva, explain preprocessing, TF-IDF, transfer learning, SVM margins,
regularization, feature fusion, class weighting, data leakage and why macro-F1
is useful when classes are imbalanced. Present measured generalization results
only after a sufficiently representative human-reviewed dataset is ready.

Run the relevant regression checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_lab_models.py tests/test_lab_page.py tests/test_experiments.py tests/test_experiment_policy.py -q
```
