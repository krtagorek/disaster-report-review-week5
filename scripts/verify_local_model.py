"""Offline pin/load, raw completion boundary, and one-forward-pass checks."""
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from baseline_common import ROOT, LABELS, development_rows, prompt, sha
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(ROOT / 'work/huggingface')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load


def verified_load():
    pin = json.loads((ROOT / 'data/model-manifest.json').read_text())
    path = ROOT / pin['local_directory']
    if not pin.get('files_sha256'):
        raise ValueError('Download not complete')
    for name, expected in pin['files_sha256'].items():
        assert sha((path/name).read_bytes()) == expected, name
    start = time.monotonic()
    model, tokenizer = load(str(path), tokenizer_config={'trust_remote_code': False}, trust_remote_code=False)
    model.eval()
    return model, tokenizer, pin, time.monotonic()-start


def check_boundaries(tokenizer, rows):
    label_ids = {c: tokenizer.encode(c, add_special_tokens=False) for c in LABELS}
    eos = tokenizer.eos_token_id
    assert tokenizer.decode([eos]) == '<|endoftext|>'
    for c, ids in label_ids.items():
        assert tokenizer.decode(ids) == c
        assert eos not in ids
    lengths = []
    for r in rows:
        p = prompt(r['text'])
        prefix = tokenizer.encode(p, add_special_tokens=False)
        lengths.append(len(prefix))
        assert prefix == tokenizer.encode(p), 'Implicit special tokens differ'
        for c in LABELS:
            combined = tokenizer.encode(p+c, add_special_tokens=False)
            assert combined == prefix + label_ids[c], (r['event_id'], c, 'Boundary merge')
            training = tokenizer.encode(p+c+tokenizer.eos_token, add_special_tokens=False)
            assert training == combined + [eos], 'EOS boundary mismatch'
    return {'checked_development_prompts': len(rows), 'labels_per_prompt': len(LABELS), 'label_ids': label_ids, 'eos_id': eos, 'eos_ids': sorted(tokenizer.eos_token_ids), 'prompt_tokens_min': min(lengths), 'prompt_tokens_max': max(lengths), 'training_contract': 'encode(prompt) + encode(label) + [151643]; mask prompt loss, supervise label and EOS; no BOS or chat template', 'passed': True}


def main():
    model, tokenizer, pin, elapsed = verified_load()
    boundary = check_boundaries(tokenizer, development_rows())
    ids = tokenizer.encode(prompt('Golf ball sized hail fell across town.'), add_special_tokens=False)
    logits = model(mx.array([ids]))
    mx.eval(logits)
    assert logits.shape == (1, len(ids), 151936)
    assert bool(mx.all(mx.isfinite(logits)).item())
    params = tree_flatten(model.parameters())
    dtypes = sorted({str(p.dtype) for _,p in params})
    assert dtypes == ['mlx.core.bfloat16'], dtypes
    report = {'repo_id': pin['repo_id'], 'revision': pin['revision'], 'license': pin['license'], 'model_manifest_sha256': sha((ROOT/'data/model-manifest.json').read_bytes()), 'load_seconds': elapsed, 'device': str(mx.default_device()), 'machine': platform.machine(), 'python': platform.python_version(), 'packages': {n: importlib.metadata.version(n) for n in ['mlx','mlx-metal','mlx-lm','transformers','tokenizers','huggingface-hub']}, 'parameter_count': sum(p.size for _,p in params), 'parameter_dtypes': dtypes, 'quantized': False, 'conversion': 'MLX upstream tensor sanitization only; no saved conversion or dtype change', 'forward_shape': list(logits.shape), 'finite_logits': True, 'peak_memory_gb': mx.get_peak_memory()/1e9, 'boundary': boundary}
    out = ROOT/'data/evaluations/local-model-verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit('Refusing to overwrite verification evidence')
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
