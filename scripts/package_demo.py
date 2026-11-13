"""Package the measured lab pilot with hashes for offline transfer."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'data/processed/lab/internal_smoke_only/visual-source-holdout-20261113'
TARGET = ROOT / 'demo_bundle'
FILES = {
    'models/fused_linear_svm.joblib': RUN / 'models/fused_linear_svm.joblib',
    'models/text_only_linear_svm.joblib': RUN / 'models/text_only_linear_svm.joblib',
    'models/image_only_linear_svm.joblib': RUN / 'models/image_only_linear_svm.joblib',
    'weights/mobilenet_v3_small-047dcff4.pth': ROOT / 'data/models/mobilenet_v3_small/mobilenet_v3_small-047dcff4.pth',
    'experiment.json': RUN / 'experiment.json',
    'comparison.csv': RUN / 'comparison.csv',
    'evaluations/fused_linear_svm.json': RUN / 'evaluations/fused_linear_svm.json',
    'evaluations/text_only_linear_svm.json': RUN / 'evaluations/text_only_linear_svm.json',
    'evaluations/image_only_linear_svm.json': RUN / 'evaluations/image_only_linear_svm.json',
    'comparison_with_prior_svm.json': RUN / 'comparison_with_prior_svm.json',
    'end_to_end_cpu.json': RUN / 'end_to_end_cpu.json',
    'configs/lab.yaml': ROOT / 'configs/lab.yaml',
    'split_manifest.json': ROOT / 'data/interim/fused-holdout-20261113/split_manifest.json',
}
TARGET.mkdir(exist_ok=True)
manifest = {}
for name, source in FILES.items():
    target = TARGET / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest[name] = {'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size}
(TARGET / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Packaged', len(manifest), 'files in', TARGET)
