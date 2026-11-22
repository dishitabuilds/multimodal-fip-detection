"""Real OCR + packaged SVM through the actual Streamlit upload/sample flow."""
from pathlib import Path
import io
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
st = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(not all((ROOT / p).is_file() for p in (
                        'demo_bundle/models/fused_linear_svm.joblib',
                        'demo_bundle/weights/mobilenet_v3_small-047dcff4.pth',
                        'demo_bundle/manifest.json', 'demo_bundle/experiment.json'))
                    or not (ROOT / 'data/models/easyocr').is_dir(),
                    reason='Real inference requires the local supplied bundle and OCR weights')
def test_real_upload_and_gallery_should_classify_with_presentation_model(monkeypatch):
    for dependency in ('torch', 'torchvision', 'easyocr', 'sklearn', 'joblib'):
        pytest.importorskip(dependency)
    # AppTest cannot set upload bytes itself. Only the uploader transport is replaced;
    # image decoding, OCR, model loading, prediction, rendering and reruns are real.
    uploaded = {'bytes': (ROOT / 'test_samples/sample_fake_1.png').read_bytes()}

    def uploader(*args, **kwargs):
        if uploaded['bytes'] is None:
            return None
        f = io.BytesIO(uploaded['bytes'])
        f.name = 'promotion.png'  # Same name also exercises changed-pixel invalidation.
        return f

    monkeypatch.setattr(st, 'file_uploader', uploader)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
        assert not app.exception
        assert app.selectbox(key='analysis_mode').value == 'Classify promotion'
        assert not app.button(key='analyze').disabled
        app.button(key='analyze').click().run()
        assert not app.exception and not app.error
        result = app.session_state['analysis_result']
        assert result['status'] == 'complete', result['error']
        assert result['label'] == 'fake'
        assert result['experiment_id'] == 'visual-source-holdout-20261113'
        assert result['artifact_sha256'] == '08dafb7407ef425f9e48601e80190cea14033a160cb0b5bae5b38a1bd82eb60c'
        assert result['class_scores'] is None
        assert result['decision_score'] == pytest.approx(0.2024, abs=0.01)
        assert any('Model prediction: Fake' in x.value for x in app.markdown)
        json.dumps(result, allow_nan=False)
        first_fingerprint = result['input_fingerprint']
        app.run()
        assert app.session_state['analysis_result']['input_fingerprint'] == first_fingerprint

        app.text_area(key='caption_input').input('Original accompanying post').run()
        assert 'analysis_result' not in app.session_state
        uploaded['bytes'] = (ROOT / 'test_samples/sample_real_with_text_1.webp').read_bytes()
        app.text_area(key='caption_input').input('').run()
        assert 'analysis_result' not in app.session_state
        app.button(key='analyze').click().run()
        assert not app.exception and not app.error
        result = app.session_state['analysis_result']
        assert result['label'] == 'real', result
        assert result['input_fingerprint'] != first_fingerprint
        assert any('Model prediction: Real' in x.value for x in app.markdown)

        app.selectbox(key='analysis_mode').select('Extract text').run()
        assert 'analysis_result' not in app.session_state
        app.button(key='analyze').click().run()
        assert app.session_state['analysis_result']['label'] is None
        assert not any('Model prediction:' in x.value for x in app.markdown)

        app.selectbox(key='analysis_mode').select('Classify promotion').run()
        app.radio(key='input_source').set_value('Sample gallery').run()
        app.button(key='sample_public_figure').click().run()
        app.button(key='analyze').click().run()
        assert not app.exception and not app.error
        assert app.session_state['analysis_result']['label'] == 'fake'
        app.button(key='reset_input').click().run()
        assert 'analysis_result' not in app.session_state
        assert app.text_area(key='caption_input').value == ''
    finally:
        st.cache_resource.clear()
