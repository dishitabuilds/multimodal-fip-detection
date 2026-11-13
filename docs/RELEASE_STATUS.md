# Release status

## Local demonstration

The Streamlit ML Lab runs three bundled examples, displays OCR and text/fused
SVM decisions, and marks ambiguous or disputed cases Needs review. The
`demo_bundle` has model files, MobileNet weights, evaluations, config and a
SHA-256 manifest. Run `scripts/verify_demo_bundle.py` to check and restore it.

## Evaluation

The latest 47-image corpus has 31 train, 6 validation and 10 test examples.
The test uses two publishers absent from the training and validation splits.
On those 10 images, fused macro-F1 is 0.792 and PR-AUC is 0.800. It misses two
of five scam screenshots. It ties image-only classification at the native
threshold. Its visual formats are associated with the classes, so this result
cannot establish generalization to matched-format promotions. Five iShares
pages were already used in a diagnostic of an earlier text model.

## Performance

On the current Mac, a warm-model OCR plus fused prediction took 5.7 seconds
median over three runs, with 3.1 GB process peak RAM. The model artifact plus
MobileNet weights total about 9.91 MB; OCR model files are additional. The
100 ms and 1 GB targets are not achieved by the complete local workflow.

## External release

An independently reviewed representative corpus, matched visual formats across
classes, and measured Hindi and mixed-language performance are needed before
claiming reliable fraud detection. The local bundle is intended for the professor
demonstration. The hosted deployment would need the ignored bundle supplied by
an approved release channel and verified on its own hardware.
