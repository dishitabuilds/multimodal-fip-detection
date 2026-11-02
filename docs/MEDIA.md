# Audio and video promotions

Open **Audio Video** in Promotion Lens. Upload a short recording, choose speech
transcription and/or sampled-frame OCR, and click **Analyze audio / video**.
The page works without a trained authenticity classifier. It displays speech
segments, English wording cues and on-screen text with playback timestamps, and
exports a JSON evidence report.

## Installation

Reuse the project's standard CPython virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-media.txt
.\.venv\Scripts\python.exe scripts/14_analyze_media.py --prepare-model
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The preparation command explicitly downloads `Systran/faster-whisper-base` from
Hugging Face into `data/models/faster-whisper-base/`. It resolves and records the
upstream commit and SHA-256 file hashes in `preparation.json`, then checks local
model loading. The model is multilingual and runs with CPU INT8 compute. Model
files total approximately 141 MiB; these are additional to OCR and classifier
assets and do not meet the original 100 MB whole-deployment target by themselves.
No end-to-end RAM or latency budget has been established for media processing.

Inference uses local assets only. No file is sent to a speech API, and prediction
never initiates a model download. Video decoding uses PyAV's bundled FFmpeg
libraries; a separate `ffmpeg.exe` installation is unnecessary. Existing local
EasyOCR English/Hindi files are reused for video text recognition.

The dependency extra is optional: existing image/text pages work without it.
This workspace has the speech dependencies and base-model weights prepared.
Weights, uploads, verification files and collected data remain ignored outputs.

## Supported files and bounds

- Audio: MP3, WAV, M4A, OGG and FLAC.
- Video: MP4, MOV, WebM and MKV. Container support does not guarantee every codec;
  a corrupt/unsupported file produces an explanation.
- Up to 50 MB and 120 seconds per file; overlong input is rejected rather than
  silently analyzing a prefix. Image pages still enforce their own 10 MB limit.
- Frames are sampled every 5 seconds by default; 2 or 10 seconds can be selected.
  Sampling can miss brief text and changes between sampled frames.
- Maximum 60 sampled frames, 8.3 million pixels per frame, 20,000 decoded frames and
  45 seconds of decoding work. The work timer is checked between decoded frames,
  not an operating-system kill timeout for a blocked native codec call.
- The decoder uses in-memory bytes and a format/protocol allowlist. Audio is
  resampled to 16 kHz mono; timestamp gaps remain silence so speech and frames
  share the container playback origin. The first audio stream and first video
  stream without an attached cover picture are used.

Choose **Automatic**, **English** or **Hindi** for speech. Hindi transcription is
available, but this project has no measured Hindi speech accuracy. The framing
rules remain English only; a Hindi transcript with no wording flags is not an
assurance that the content is balanced or legitimate.

## Existing classifiers

The optional classifier selector uses the ML Lab's saved SVM, Logistic Regression
and Naive Bayes artifacts. Research artifacts appear by default; provisional
14-image classifiers require explicitly enabling **Show internal pilot classifiers**.

Text-only classifiers can score speech segments and sampled-frame OCR. Image and
fused classifiers apply to sampled frames with the matching MobileNet weights.
Speech and visual evidence are displayed separately; no learned temporal or
speech/vision fusion model is trained here. Long text is split into overlapping
windows within the artifact's input limit, rather than silently truncated.

SVM outputs retain their signed margins. Other class scores are uncalibrated.
Each output belongs to a segment/window or sampled frame, and has no validated
audio/video generalization. There is no aggregate file-level fraud probability
or authenticity verdict. No detection of synthetic voices, face swaps, speaker
identity or account authenticity is implemented.

Transcription and frame OCR fail independently where possible: a missing speech
model does not hide successfully extracted frame text. Errors and skipped/empty
channels are retained in the report. Failed classification does not discard
successfully transcribed speech. Changing the input, caption, settings or model
invalidates the previous report.

## Command line

```powershell
.\.venv\Scripts\python.exe scripts/14_analyze_media.py promotion.mp3 --output data/processed/audio-evidence.json
.\.venv\Scripts\python.exe scripts/14_analyze_media.py promotion.mp4 --frame-interval 2 --output data/processed/video-evidence.json
.\.venv\Scripts\python.exe scripts/14_analyze_media.py promotion.mp4 --no-speech
```

`--language en`/`hi` selects speech language. `--no-frames` disables frame OCR.
Use `--artifact PATH` for a locally generated lab model, plus `--internal-pilot`
only for a provisional artifact. Existing output reports are never overwritten.
Exit 0 means all requested applicable channels completed; exit 2 means an input
error, partial failure or no evidence. A partial report is still written.

## Accuracy work and evaluation

A fresh audit of the selected source inventory on 2 November 2026 created
`data/processed/accuracy-review-20261102/manual_review.csv` and
`dataset_audit.json`. The 1,246 selected candidates all have unresolved labels;
they are not a reviewed research corpus. Missing usable creatives and OCR are
also reported. Human decisions were not fabricated or imported by this change.

Review task relevance, labels, original creatives/captions and OCR using the
existing [dataset protocol](../DATASET.md). Import attributed decisions using
`scripts/apply_reviews.py`, complete curation, regenerate group-aware splits and
train the baseline comparisons as described in [ML Lab](ML_LAB.md).

For media evaluation, keep every frame and audio chunk from the same original
clip/campaign in one split. Collect reviewed legitimate and misleading clips,
including warning/educational clips and factual promotions, with speaker/language
and source variety. Measure speech word error rate, OCR errors, claim localization,
segment and clip detection recall, false alarms and macro-F1 separately. File-level
aggregation must be selected on validation data and tested on held-out clips
before it is exposed as a verdict. Added input support is not a measured accuracy gain.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_media.py tests/test_media_page.py -q
```

Tests use real WAV/Matroska codec round trips and deterministic recognizer/OCR
doubles for failure and boundary cases. A separate real speech check used the
upstream faster-whisper JFK recording (11 seconds), and produced the expected
English transcript through the CLI and Streamlit page. This verifies speech
integration; it does not establish investment-promotion detection performance.

The initial related regression run passed 167 tests; the final media-only run
passed 16 tests after adding classifier/OCR failure-recovery coverage. A local
codec fixture made from an existing sample creative also completed real frame
OCR and fused-pilot inference at three playback timestamps. The fixture and
public speech clip are engineering checks, not held-out promotion examples.

Primary dependency references: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[PyAV](https://pyav.org/docs/stable/).
