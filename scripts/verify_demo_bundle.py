"""Verify a packaged pilot and restore its files to the local demo paths."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'demo_bundle'
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
for name, expected in manifest.items():
    item = BUNDLE / name
    if hashlib.sha256(item.read_bytes()).hexdigest() != expected['sha256']:
        raise SystemExit(f'Bundle hash mismatch: {name}')
run = ROOT / 'data/processed/lab/internal_smoke_only/visual-source-holdout-20261113'
for name in ('fused_linear_svm', 'text_only_linear_svm', 'image_only_linear_svm'):
    source = BUNDLE / 'models' / f'{name}.joblib'
    target = run / 'models' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
for name in ('experiment.json', 'comparison.csv'):
    shutil.copy2(BUNDLE / name, run / name)
for source in (BUNDLE / 'evaluations').glob('*.json'):
    target = run / 'evaluations' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
weights = ROOT / 'data/models/mobilenet_v3_small' / (BUNDLE / 'weights/mobilenet_v3_small-047dcff4.pth').name
weights.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(BUNDLE / 'weights' / weights.name, weights)
print(f'Verified {len(manifest)} files and restored the local demo')
