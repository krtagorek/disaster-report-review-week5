"""Resolve once, pin upstream revision, then download only that revision locally."""
import hashlib
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / 'work/huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from huggingface_hub import HfApi, snapshot_download

pin = ROOT / 'data/model-manifest.json'
if pin.exists():
    record = json.loads(pin.read_text())
else:
    info = HfApi().model_info('Qwen/Qwen3-1.7B-Base')
    record = {'repo_id': info.id, 'revision': info.sha, 'license': info.card_data.get('license'), 'local_directory': 'runtime/models/qwen3-1.7b-base', 'quantization': None, 'remote_code': False}
    if record['license'] != 'apache-2.0':
        raise ValueError('Unexpected license; inspect before use')
    pin.write_text(json.dumps(record, indent=2)+'\n')
path = ROOT / record['local_directory']
snapshot_download(record['repo_id'], revision=record['revision'], local_dir=path, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.md', 'LICENSE', '*.tiktoken'])
record['files_sha256'] = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.parts}
pin.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
