"""Hash-checked local adapter/merged loading; no hosted inference or final data."""
import json
from baseline_common import ROOT,sha
from verify_local_model import verified_load
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters

RUN=ROOT/'data/training/lora-v1'
ADAPTER=ROOT/'runtime/lora-v1'
MERGED=ROOT/'runtime/merged-lora-v1'


def load_candidate(kind):
    result=json.loads((RUN/'training-result.json').read_text())
    assert result['completed_steps']==600 and result['frozen_base_hashes_unchanged']
    assert sha((ADAPTER/'adapters.safetensors').read_bytes())==result['adapter_sha256']
    assert sha((ADAPTER/'adapter_config.json').read_bytes())==result['adapter_config_sha256']
    if kind=='adapter':
        model,tok,_,_=verified_load()
        model=load_adapters(model,str(ADAPTER))
        fingerprint=result['adapter_sha256']
    elif kind=='merged':
        manifest=json.loads((RUN/'merged-manifest.json').read_text())
        for name,digest in manifest['files_sha256'].items():
            assert sha((MERGED/name).read_bytes())==digest,name
        model,tok=load(str(MERGED),trust_remote_code=False,tokenizer_config={'trust_remote_code':False})
        fingerprint=sha((RUN/'merged-manifest.json').read_bytes())
    else:
        raise ValueError(kind)
    model.eval()
    return model,tok,fingerprint
