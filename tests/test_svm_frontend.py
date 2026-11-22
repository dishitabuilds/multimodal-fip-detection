"""Regressions for the presentation's saved SVM in the main analysis flow."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fipd.frontend import analysis
from fipd.enrichment.ocr import OCRResult


class OCR:
    def __init__(self, text='Guaranteed monthly returns'):
        self.text = text

    def read(self, path):
        return OCRResult(text=self.text, lines=[self.text], backend='test')


@pytest.mark.parametrize('text', [
    'Congratulations! You have won a $500 gift card. Click here to claim your reward: https://example.com',
    'Review our remote working policy and win an iPhone. Use the attached QR code to review the policy.',
])
def test_phishing_promotion_should_not_be_presented_as_real(monkeypatch, text):
    from fipd.models import classical

    def predict(*args, **kwargs):
        return dict(label='real', score=-0.16, score_kind='decision_margin', threshold=0.0,
                    model='linear_svm', arm='fused', purpose='internal_smoke_only',
                    experiment_id='visual-source-holdout-20261113', unvalidated=True)

    monkeypatch.setattr(classical, 'predict_lab', predict)
    components = SimpleNamespace(mode='svm', purpose='internal_smoke_only', threshold=0.0,
                                 artifact={'saved': True}, encoder='matching-encoder',
                                 artifact_sha256='verified-model-hash')
    result = analysis.analyze_image(Image.new('RGB', (32, 24)), mode='svm',
                                    ocr_loader=lambda: OCR(text), model_loader=lambda: components)
    assert result['status'] == 'complete'
    assert result['label'] is None
    assert result['review_reason'] == 'phishing_out_of_scope'
    assert result['model_label'] == 'real'


def test_phishing_guard_should_tolerate_observed_ocr_noise_without_flagging_investments():
    assert analysis.phishing_out_of_scope('Win an iPhone 1५! Use the attached OR codc to review the policy')
    assert analysis.phishing_out_of_scope('Congratulations! Youive won a $500 card. Click here to claim your reward')
    assert not analysis.phishing_out_of_scope('Invest in a mutual fund and win a prize. Click to learn more')
    assert not analysis.phishing_out_of_scope('Our gift card policy explains how rewards work')


def test_qr_payload_should_be_extracted_without_visiting_destination():
    cv2 = pytest.importorskip('cv2')
    import numpy as np
    payload = 'https://example.invalid/verify'
    qr = cv2.QRCodeEncoder_create().encode(payload)
    image = Image.fromarray(np.pad(cv2.resize(qr, None, fx=4, fy=4,
                                             interpolation=cv2.INTER_NEAREST),
                                   40, constant_values=255))
    assert analysis.decode_qr(image) == [payload]


@pytest.mark.parametrize('label,score', [('fake', 0.202), ('real', -0.003)])
def test_main_analysis_should_use_saved_svm_with_original_caption(monkeypatch, label, score):
    from fipd.models import classical
    calls = []

    def predict(artifact, text, caption, image, encoder):
        calls.append((text, caption, image.mode, encoder))
        return dict(label=label, score=score, score_kind='decision_margin', threshold=0.0,
                    model='linear_svm', arm='fused', purpose='internal_smoke_only',
                    experiment_id='visual-source-holdout-20261113', unvalidated=True)

    monkeypatch.setattr(classical, 'predict_lab', predict)
    components = SimpleNamespace(mode='svm', purpose='internal_smoke_only', threshold=0.0,
                                 artifact={'saved': True}, encoder='matching-encoder',
                                 artifact_sha256='verified-model-hash')
    result = analysis.analyze_image(Image.new('RGB', (32, 24)), 'Original post caption',
                                   mode='svm', ocr_loader=lambda: OCR(),
                                   model_loader=lambda: components)
    assert result['status'] == 'complete', result['error']
    assert result['label'] == label
    assert result['decision_score'] == score
    assert result['class_scores'] is None  # A margin must never become a confidence percentage.
    assert result['score_kind'] == 'decision_margin'
    assert result['experiment_id'] == 'visual-source-holdout-20261113'
    assert calls == [('Guaranteed monthly returns', 'Original post caption', 'RGB', 'matching-encoder')]
    json.dumps(result, allow_nan=False)


def test_svm_runtime_should_resolve_bundle_without_legacy_checkpoints(tmp_path):
    for relative in (analysis.SVM_CHECKPOINT_RELATIVE, analysis.SVM_WEIGHTS_RELATIVE,
                     analysis.SVM_MANIFEST_RELATIVE, analysis.SVM_EXPERIMENT_RELATIVE):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'presence only')
    runtime = analysis.mode_runtime(tmp_path, 'svm')
    assert runtime['checkpoint_available']
    assert runtime['checkpoint_path'].endswith('demo_bundle/models/fused_linear_svm.joblib')
    assert runtime['unvalidated']


def test_missing_svm_bundle_should_report_missing_assets_without_loading_pickle(tmp_path):
    with pytest.raises(analysis.AnalysisError) as caught:
        analysis.load_classifier(tmp_path, 'svm')
    assert caught.value.code == 'svm_assets_missing'


def test_tampered_svm_should_be_rejected_before_deserialization(tmp_path, monkeypatch):
    joblib = pytest.importorskip('joblib')
    bundle = tmp_path / 'demo_bundle'
    (bundle / 'models').mkdir(parents=True)
    (bundle / 'weights').mkdir()
    (bundle / 'models/fused_linear_svm.joblib').write_bytes(b'not-the-saved-model')
    (bundle / 'weights/mobilenet_v3_small-047dcff4.pth').write_bytes(b'weights')
    (bundle / 'experiment.json').write_text('{}')
    (bundle / 'manifest.json').write_text(json.dumps({
        'models/fused_linear_svm.joblib': {'sha256': '0' * 64, 'bytes': 19},
        'weights/mobilenet_v3_small-047dcff4.pth': {'sha256': '0' * 64, 'bytes': 7},
        'experiment.json': {'sha256': '0' * 64, 'bytes': 2},
    }))
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('Do not deserialize changed files'))
    with pytest.raises(analysis.AnalysisError) as caught:
        analysis.load_classifier(tmp_path, 'svm')
    assert caught.value.code == 'svm_bundle_incompatible'
