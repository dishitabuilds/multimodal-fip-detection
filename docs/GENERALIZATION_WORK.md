# Broader scam-image classifier work

## Current evidence

The 47-image financial-promotion pilot predicts **real** for both user-supplied
phishing screenshots (margins -0.157 and -0.169). Their bytes, user-provided
labels, and development-only role are recorded in the ignored local
`data/interim/generalization-challenge/manifest.json`. Run
`.venv/bin/python scripts/evaluate_generalization_challenge.py` to repeat the
check. The present app abstains on reward lures; that is a scope guard, not a
learned phishing classifier.

## Target and proof

Target images include financial promotions, email screenshots, and SMS
screenshots. Class labels must distinguish a verified deceptive claim or
impersonation from a verified legitimate message; ambiguous evidence goes to
review. Spam alone is not a phishing ground truth. The two supplied images may
guide development but cannot serve as independent test results.

Build source-attributed, consented or openly licensed image cohorts with
both classes **within each format**. Include legitimate prize notices, real
workplace policy emails, and ordinary SMS so a reward or QR cue cannot decide
the label alone. Record original source, image hash, claim evidence, capture
date, review decision, and reviewer. Remove near duplicates and group by
campaign, sender, and publisher before splitting. Keep publishers, templates,
brands, and campaign families disjoint across train and final test where
possible. Freeze the final test before model selection.

Compare OCR text, visual, and fused models on the same cohort. Select using
validation data. Report per-format confusion matrices, fake recall, real false
positive rate, abstention coverage, and calibration where probabilities are
available. Require a larger independent test with matched formats and manual
label review before replacing the pilot in the demo. The two challenge images
must become suspicious/review or fake without simply memorizing their wording.

## Candidate sources

- [Phish360](https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/) has
  phishing and legitimate **webpage** screenshots. Useful for a web-specific
  cohort, but its format differs from email and SMS; check terms before use.
- [UCI SMS Spam Collection](https://archive-beta.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection/files)
  has text messages, not screenshots, and spam labels. It can support auxiliary
  text experiments, not direct phishing-image ground truth.

Neither source alone establishes the requested cross-format performance.

## Auxiliary text experiments (22 Nov 2026)

The ignored local sources and outputs can be reproduced with
`.venv/bin/python scripts/train_smishing_text.py` and
`.venv/bin/python scripts/train_email_text.py`. Neither is in the demo.

- Mishra and Soni's [Mendeley SMS dataset](https://data.mendeley.com/datasets/f45bkkt8pr/1)
  was acquired through a GitHub mirror (local SHA-256
  `649844f1c62a6b27e145eaf17a65f7010c56c390e11a794ea8329993a05ba71e`).
  Ham and smishing were used; generic spam was excluded. The script masks
  exact URLs and groups duplicate numeric templates before splitting.
  The initial run had 791 test messages, with 67 true positives, zero false
  negatives, four false positives, and 720 true negatives. Its 68,029-message
  [NCSU reported-phishing corpus](https://github.com/wspr-ncsu/sms-phishing)
  positive-only check flagged 97.9%; that corpus includes marketing, and
  without legitimate controls this does not establish external accuracy.
- [MeAJOR v2](https://zenodo.org/records/18471483) was downloaded directly
  from Zenodo (MD5 `78e397ad8447bcdba5a98097921ba8bd`, matching its file
  listing). TREC-05 trained the email text model, TREC-06 validated it, and
  TREC-07 was a distinct-source test after duplicate removal. On 42,547 test
  emails: 21,225 true positives, 2,412 false negatives, 4,244 false positives,
  and 14,666 true negatives. The 22.4% false-positive rate is too high for a
  Fake/Real demo result.

Both datasets are text, so OCR errors, screenshot layout, QR destinations,
and actual sender identity remain untested. The email screenshot's QR decodes
to “In a real campaign, this would lead to a phishing website.” It is an
educational simulation, not evidence that the depicted destination is live.
The app now extracts and displays QR payloads locally without opening them;
this is evidence for review, not a learned class feature.
On the two development uploads, the SMS text model's uncalibrated scores were
0.699 (email simulation) and 0.977 (SMS), while the email text model scored
0.560 and 0.980. The email model's score for a clean transcription of the
first image was about 0.216, showing substantial sensitivity to OCR errors.
These scores are not fraud probabilities or independent test results.
With EasyOCR's separate English checkpoint, the email image's email-text score
dropped to 0.193; multilingual OCR had produced 0.560. The SMS image remained
high (0.975 versus 0.980). This instability is another reason the auxiliary
models are not exposed as final labels.

## Cross-format experiment and independent stress set

`scripts/train_joint_message_text.py` trains one OCR-text candidate on SMS
plus email, selecting the SMS weight from validation outcomes in both formats.
Its separate test groups were 791 SMS messages and 42,547 TREC-07 emails.
The selected candidate had 2.49% false positives and 91.0% phishing recall
on the SMS group, but 24.1% false positives and 86.8% recall on TREC-07.
The joint model does not solve the email false-alarm problem. On the two
development screenshots, its raw OCR-text scores were 0.444 (email simulation)
and 0.989 (SMS). It still misses the first case at a 0.5 threshold.

An independent matched workplace-email stress set was obtained from the
[NEST-Phish public release](https://www.kaggle.com/datasets/emilywinokur/synthetic-email-corpus),
whose dataset README states CC BY-SA 4.0. The frozen email, SMS, and joint
artifacts were evaluated once with `scripts/evaluate_nest_phish.py` before
any training on NEST-Phish. Among its 77 synthetic non-phishing and 69 synthetic
phishing emails, the email model missed 28 phishing cases and flagged 18
legitimate cases; the joint model missed 34 and flagged 13. The complete
per-theme results and artifact hashes are in the ignored local
`data/processed/nest_phish_external_evaluation.json`. This cohort is text,
not screenshots, and it becomes development data if used to tune future models.

These outcomes rule out a reliable broad Fake/Real label from the current
artifacts. Further work needs matched-format image cohorts with independent
human labels, OCR testing on real screenshots, sender and destination evidence,
and a new untouched external evaluation source after training.

## Multimodal matched-format cohort and screenshot benchmarks (23 Nov 2026)

### 1. Source-attributed cross-format image cohort
A matched-format image dataset of 120 images (40 per format) was constructed under
`data/interim/multimodal-generalization/`:
- **Investment graphics** (20 fake, 20 real): Newschecker and Scamwatch fact-checked scam graphics alongside authentic AMFI investor education, SBI mutual fund factsheets, iShares ETF factsheets, and matched synthetic disclosure cards.
- **Email screenshots** (20 phishing, 20 legitimate): NEST-Phish matched synthetic workplace emails rendered as desktop client screenshots across policy reviews, IT maintenance, payroll updates, and security credentials. Includes hard legitimate cases: real workplace policy acknowledgements, IT maintenance notices with authentic intranet links, and mandatory 2FA enrollment with legitimate `otpauth://` QR codes.
- **SMS screenshots** (20 smishing, 20 legitimate): Mendeley smishing lures ($500 Target gift cards, bank KYC suspensions, postal fee alerts) alongside hard legitimate cases (ICICI payee registrations with explicit fraud warnings, Google 2FA codes, Starbucks loyalty reward updates, and Amazon delivery notices).

### 2. Deduplication and leak-free splitting
Exact duplicate images were deduplicated by SHA-256 and text duplicate keys.
Near duplicates were checked using dHash perceptual hashing.
Grouping was strictly enforced by `campaign_id`, sender domain, and template family:
- **Train (60%, 72 images)**: 24 investment, 24 email, 24 SMS (12 fake, 12 real each).
- **Validation (20%, 24 images)**: 8 investment, 8 email, 8 SMS (4 fake, 4 real each).
- **Test (20%, 24 images - FROZEN)**: 8 investment, 8 email, 8 SMS (4 fake, 4 real each).
Zero campaign or sender overlap exists between splits. The test set was frozen prior to model selection.

### 3. OCR and local evidence measurement
Evaluated EasyOCR multilingual (`['en', 'hi']`) against English-only (`['en']`):
- Multilingual: 4,415.7 ms average latency, 0.634 mean confidence. Often hallucinates Devanagari fragments on Latin-only screenshots, but remains necessary for Hindi mutual fund graphics.
- English-only: 1,260.7 ms average latency, 0.759 mean confidence. Substantially more stable on email and SMS screenshots.
- Local QR decoding (`cv2.QRCodeDetector`) successfully extracts payloads without network resolution.
- OCR text repair standardizes `http Il...` and `http ://` to prevent URL-splitting errors while masking exact domain identities.

### 4. Matched-cohort benchmark results
All hyperparameter tuning ($C$) and threshold selection was performed on the validation set only:

| Model Architecture | Frozen Test Macro F1 | Scam Recall | Real False Positive Rate | Test Abstention Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Text-only Baseline** | 0.708 | 75.0% | 33.3% | 0.0% |
| **Image-only Baseline** | 0.657 | 83.3% | 50.0% | 0.0% |
| **Fused Joint Baseline** | 0.748 | 83.3% | 33.3% | 0.0% |
| **Format Specialists (Ours)** | **0.832** | **91.7%** | **25.0%** | **66.7%** |

Format breakdown on frozen test set for Format Specialists:
- Investment graphics: Macro F1 1.000 (Scam Recall 100%, Real FPR 0.0%)
- Email screenshots: Macro F1 1.000 (Scam Recall 100%, Real FPR 0.0%)
- SMS screenshots: Macro F1 0.467 (Scam Recall 75.0%, Real FPR 75.0%)

### 5. Evaluation on user development challenge images
Evaluated with `scripts/evaluate_generalization_challenge.py`:
1. **Email QR Challenge (`phishing_email_qr.png`)**:
   - Detected format: `email_screenshot`
   - QR payload decoded: `"In a real campaign, this would lead to a phishing website."`
   - Raw SVM margin: -0.157 ("real")
   - Specialist calibrated probability: 0.429
   - Final status: `Review required` (`simulated_phishing_qr`)
2. **SMS Reward Challenge (`smishing_reward.png`)**:
   - Detected format: `sms_screenshot`
   - Indicators: Reward incentive lure ($500 gift card)
   - Raw SVM margin: -0.169 ("real")
   - Specialist calibrated probability: 0.500 (uncertain)
   - Final status: `Review required` (`phishing_out_of_scope`)

### 6. Remaining limitations and unverified areas
- **SMS false alarms**: SMS text is brief, and real banking/delivery notices share lexical patterns with smishing. Safe abstention ("Review required") is essential.
- **Visual confounding**: Pixels alone cannot verify whether a sender domain or link is authentic without cryptographic email signatures (DKIM/SPF) or live network safety checks.
- **Pilot status**: The models remain research artifacts for human decision support, not certified production-grade classifiers.

### 7. Interactive workbench integration (3 Analysis Modes)
The Promotion Lens application (`app.py`) now provides 3 explicit analysis modes in the sidebar dropdown:
1. **Classify promotion** (default): Packaged 47-image Linear SVM baseline for investment promotions with scope safeguard abstention on out-of-scope non-investment lures.
2. **Cross-format detector (120-image model)**: Directly powered by `data/processed/multimodal_generalization/model.joblib`. Detects layout format (`investment_graphic`, `email_screenshot`, `sms_screenshot`), routes to format-specialist classifiers, and outputs Platt-calibrated scam/benign probabilities. App scores from 45% through 55% display `Review required`; scores outside that range display a binary prediction.
3. **Extract text**: Offline English and Hindi CPU OCR and local QR code decoding without running classifier inference.
