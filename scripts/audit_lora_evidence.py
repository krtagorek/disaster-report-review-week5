"""Validate training exposure/provenance and freeze completed local-run evidence."""
import collections
import json
from baseline_common import ROOT,LABELS,sha
from lora_artifacts import RUN,ADAPTER,MERGED
import mlx.core as mx


def main():
    destination=RUN/'evidence-manifest.json'
    if destination.exists():
        record=json.loads(destination.read_text())
        for name,digest in record['files_sha256'].items():
            assert sha((ROOT/name).read_bytes())==digest,name
        print('Frozen evidence manifest verified; no files changed')
        return
    result=json.loads((RUN/'training-result.json').read_text())
    freeze=json.loads((ROOT/'data/splits/v1/freeze.json').read_text())
    splits={}
    for split in ('train','development'):
        data=(ROOT/f'data/splits/v1/{split}.jsonl').read_bytes()
        assert sha(data)==freeze['artifact_sha256'][split+'.jsonl']
        splits[split]={r['event_id']:r for r in map(json.loads,data.splitlines())}
    exposure=json.loads((RUN/'training-exposure-ids.json').read_text())
    monitor=json.loads((RUN/'loss-monitor-ids.json').read_text())
    assert len(exposure)==len(set(exposure))==2400
    assert set(exposure)<=splits['train'].keys()
    assert set(monitor)<=splits['development'].keys()
    assert not set(exposure)&set(monitor)
    exposure_counts=dict(collections.Counter(splits['train'][i]['reference_label'] for i in exposure))
    assert exposure_counts==result['training_exposure_class_counts']
    assert collections.Counter(splits['development'][i]['reference_label'] for i in monitor)=={c:40 for c in LABELS}
    signature=json.loads((RUN/'run-signature.json').read_text())
    assert all(sha((ROOT/name).read_bytes())==digest for name,digest in signature.items())
    weights=mx.load(str(ADAPTER/'adapters.safetensors'))
    assert len(weights)==64 and sum(v.size for v in weights.values())==917504
    assert all(bool(mx.all(mx.isfinite(v)).item()) for v in weights.values())
    b=[v for name,v in weights.items() if name.endswith('.lora_b')]
    assert len(b)==32 and all(bool(mx.any(v!=0).item()) for v in b)
    checkpoint=mx.load(str(ADAPTER/'0000600_adapters.safetensors'))
    assert set(checkpoint)==set(weights)
    assert all(bool(mx.array_equal(weights[n],checkpoint[n]).item()) for n in weights)
    assert sha((ADAPTER/'adapters.safetensors').read_bytes())==result['adapter_sha256']
    merged=json.loads((RUN/'merged-manifest.json').read_text())
    assert not (MERGED/'adapter_config.json').exists()
    assert merged['quantization'] is None
    for name,digest in merged['files_sha256'].items():assert sha((MERGED/name).read_bytes())==digest
    comparison=json.loads((RUN/'comparison.json').read_text())
    assert comparison['n']==2449 and comparison['identical_ids_labels_order'] and comparison['metrics_recomputed']
    paths=set(RUN.glob('*'))|{ROOT/'docs/LORA_RUN.md',ROOT/'requirements-model.lock',ROOT/'requirements-plot.lock',ROOT/'data/model-manifest.json',ROOT/'data/source-manifest.json'}
    paths.update(ROOT/'scripts'/name for name in ['train_lora.py','review_lora_loss.py','merge_lora.py','lora_artifacts.py','evaluate_lora.py','compare_lora.py','run_matched_lora_evaluation.py','plot_lora_loss.py','plot_lora_evaluation.py','write_lora_report.py','audit_lora_evidence.py','baseline_common.py','letter_choice.py'])
    paths.update([ROOT/'tests/test_completion_loss.py'])
    for directory in ['adapter-lora-v1-raw','adapter-lora-v1-letter','merged-lora-v1-raw','merged-lora-v1-letter']:
        paths.update((ROOT/'data/evaluations'/directory).glob('*'))
    record={'run_id':'lora-v1','training_exposure_and_monitor_membership_pass':True,'frozen_train_signature_pass':True,'adapter_tensors_finite':True,'all_32_lora_b_tensors_nonzero':True,'final600_checkpoint_tensor_equality':True,'merged_file_hashes_pass':True,'reserved_final_opened':False,'files_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(paths) if p.is_file() and p!=destination}}
    destination.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='files_sha256'},indent=2))


if __name__=='__main__':main()
