"""fipd — Fake Investment Promotion Detection.

A lightweight multimodal model for detecting fraudulent investment promotions
where the deceptive claim is carried by the image rather than the caption.

Pipeline stages, in order:

    collection  -> raw FactCheckRecords from fact-check archives
    enrichment  -> OCR text and script normalisation added to those records
    curation    -> relevance filtering, labelling, dedup, splitting
    datasets    -> torch-facing views over the curated data
    models      -> encoders, fusion, distillation
    evaluation  -> metrics, ablation, error analysis
    serving     -> ONNX export, API, demo
"""

__version__ = "0.2.0"
