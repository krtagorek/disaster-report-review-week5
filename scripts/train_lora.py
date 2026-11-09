"""One bounded offline LoRA run with explicit raw-completion loss masking."""
import collections
import csv
import gzip
import json
import math
import os
import time
from pathlib import Path
from baseline_common import ROOT, LABELS, prompt, sha
from verify_local_model import verified_load
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import TrainingArgs, train, iterate_batches
from mlx_lm.tuner.utils import linear_to_lora_layers

OUT = ROOT/'data/training/lora-v1'
ADAPTER = ROOT/'runtime/lora-v1'


def completion_mask(batch, lengths):
    positions = mx.arange(1, batch.shape[1])
    return (positions >= lengths[:,0:1]) & (positions < lengths[:,1:2])


def completion_loss(model, batch, lengths):
    logits = model(batch[:,:-1])
    mask = completion_mask(batch, lengths)
    losses = nn.losses.cross_entropy(logits, batch[:,1:])
    ntokens = mask.sum()
    return (losses.astype(mx.float32)*mask).sum()/ntokens, ntokens


def prepare(tokenizer):
    freeze = json.loads((ROOT/'data/splits/v1/freeze.json').read_text())
    records = {}
    for split in ('train','development'):
        data = (ROOT/f'data/splits/v1/{split}.jsonl').read_bytes()
        assert sha(data) == freeze['artifact_sha256'][split+'.jsonl']
        records[split] = [json.loads(l) for l in data.splitlines()]
    assert not ({r['event_id'] for r in records['train']} & {r['event_id'] for r in records['development']})
    wanted = {r['event_id'] for rs in records.values() for r in rs}
    source = json.loads((ROOT/'data/source-manifest.json').read_text())
    path = ROOT/source['local_file']
    assert sha(path.read_bytes()) == source['sha256']
    with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as f:
        lookup = {r['EVENT_ID']:r for r in csv.DictReader(f) if r['EVENT_ID'] in wanted}
    datasets = {}
    summaries = {}
    for split, rs in records.items():
        processed = []
        for r in rs:
            raw = lookup[r['event_id']]
            assert sha(raw['EVENT_NARRATIVE'].encode()) == r['narrative_sha256']
            assert raw['EVENT_TYPE'] == r['reference_label']
            assert raw['EPISODE_ID'] == r['episode_id']
            prefix = tokenizer.encode(prompt(raw['EVENT_NARRATIVE']),add_special_tokens=False)
            target = tokenizer.encode(r['reference_label'],add_special_tokens=False)+[tokenizer.eos_token_id]
            combined = tokenizer.encode(prompt(raw['EVENT_NARRATIVE'])+r['reference_label']+tokenizer.eos_token,add_special_tokens=False)
            assert combined == prefix+target
            assert len(combined) <= 1024, 'No truncation allowed'
            processed.append((combined,len(prefix)))
        datasets[split] = processed
        summaries[split] = {'rows':len(rs),'max_tokens':max(len(t) for t,_ in processed),'completion_target_tokens':sum(len(t)-p for t,p in processed),'class_counts':dict(collections.Counter(r['reference_label'] for r in rs))}
    monitor_indices = sorted(i for c in LABELS for i in sorted((i for i,r in enumerate(records['development']) if r['reference_label']==c), key=lambda i:sha(records['development'][i]['event_id'].encode()))[:40])
    monitor = [datasets['development'][i] for i in monitor_indices]
    (OUT/'loss-monitor-ids.json').write_text(json.dumps([records['development'][i]['event_id'] for i in monitor_indices],indent=2)+'\n')
    # Preflight actual trainer batches: target token IDs must equal label+EOS,
    # never a prompt token or the extra pad slot at sequence_length.
    for data in [datasets['train'], monitor]:
        for batch,lengths in iterate_batches(data,4,1024,loop=False,seed=0):
            mask = completion_mask(batch,lengths)
            for tokens, pair, selected in zip(batch.tolist(),lengths.tolist(),mask.tolist()):
                start,end = pair
                assert [x for x,on in zip(tokens[1:],selected) if on] == tokens[start:end]
                assert tokens[end-1] == tokenizer.eos_token_id
                assert sum(selected) in (3,4)
    (OUT/'data-preflight.json').write_text(json.dumps({'source_sha256':source['sha256'],'split_sha256':{s:freeze['artifact_sha256'][s+'.jsonl'] for s in records},'summaries':summaries,'monitor_rows':len(monitor),'monitor_policy':'40 per class, SHA256(EVENT_ID) selection','all_raw_boundaries_pass':True,'actual_batch_mask_checks_pass':True,'padding_supervised':False,'final_accessed':False},indent=2)+'\n')
    return datasets['train'], monitor, records['train']


def frozen_hashes(model):
    return {name:sha(np.asarray(value.astype(mx.float32)).tobytes()) for name,value in tree_flatten(model.parameters()) if not name.endswith(('.lora_a','.lora_b'))}


class Log:
    def write(self, kind, data):
        loss = data.get('train_loss', data.get('val_loss'))
        assert loss is not None and math.isfinite(loss), 'Non-finite loss; stop run'
        with (OUT/'losses.jsonl').open('a') as f:
            f.write(json.dumps({'kind':kind,**data})+'\n')
            f.flush(); os.fsync(f.fileno())
        print(kind,json.dumps(data),flush=True)

    def on_train_loss_report(self,data): self.write('train',data)
    def on_val_loss_report(self,data): self.write('development_monitor',data)


def main():
    if ADAPTER.exists() or (OUT/'run-signature.json').exists():
        raise SystemExit('Refusing another run/overwrite; this run has no implicit resume')
    plan = json.loads((OUT/'plan.json').read_text())
    model, tokenizer, pin, _ = verified_load()
    train_data, monitor, row_ids = prepare(tokenizer)
    np.random.seed(plan['seed']); mx.random.seed(plan['seed'])
    model.freeze()
    linear_to_lora_layers(model,plan['num_layers'],plan['lora_parameters'])
    params = tree_flatten(model.trainable_parameters())
    assert len(params)==64 and all(n.endswith(('.lora_a','.lora_b')) for n,_ in params)
    before = frozen_hashes(model)
    ADAPTER.mkdir(parents=True)
    adapter_config = {'fine_tune_type':'lora','num_layers':plan['num_layers'],'lora_parameters':plan['lora_parameters']}
    (ADAPTER/'adapter_config.json').write_text(json.dumps(adapter_config,indent=2)+'\n')
    signature = {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [Path(__file__),OUT/'plan.json',ROOT/'scripts/baseline_common.py',ROOT/'data/model-manifest.json',ROOT/'requirements-model.lock']}
    (OUT/'run-signature.json').write_text(json.dumps(signature,indent=2)+'\n')
    (OUT/'frozen-base-before.json').write_text(json.dumps(before,indent=2)+'\n')
    # Capture actual batch exposure without changing the trainer's RNG path.
    used = []
    def tracked_batches(dataset, **kwargs):
        if dataset is train_data:
            ordered = sorted(range(len(train_data)),key=lambda i:len(train_data[i][0]))
            batches = [ordered[i:i+4] for i in range(0,len(ordered)-4+1,4)]
            while True:
                for index in np.random.permutation(len(batches)):
                    members=batches[index]
                    used.extend(members)
                    items=[dataset[i] for i in members]
                    lengths=[len(t) for t,_ in items]
                    width=min(1024,1+32*((max(lengths)+31)//32))
                    padded=np.zeros((4,width),dtype=np.int32)
                    for j,(tokens,_) in enumerate(items):
                        padded[j,:len(tokens)]=tokens
                    yield mx.array(padded),mx.array([(offset,len(tokens)) for tokens,offset in items])
                if not kwargs.get('loop'): break
        else:
            yield from iterate_batches(dataset,**kwargs)
    args = TrainingArgs(batch_size=4,iters=600,val_batches=-1,steps_per_report=20,steps_per_eval=100,steps_per_save=100,max_seq_length=1024,adapter_file=str(ADAPTER/'adapters.safetensors'),grad_checkpoint=True,grad_accumulation_steps=1)
    start=time.monotonic()
    train(model,optim.Adam(learning_rate=plan['learning_rate']),train_data,monitor,args=args,loss=completion_loss,iterate_batches=tracked_batches,training_callback=Log())
    after=frozen_hashes(model)
    assert before==after, 'Frozen base weights changed'
    assert len(used)==2400
    exposure=[row_ids[i]['event_id'] for i in used]
    (OUT/'training-exposure-ids.json').write_text(json.dumps(exposure,indent=2)+'\n')
    result={'completed_steps':600,'training_exposures':len(used),'unique_training_rows':len(set(used)),'training_exposure_class_counts':dict(collections.Counter(row_ids[i]['reference_label'] for i in used)),'seconds':time.monotonic()-start,'trainable_parameters':sum(p.size for _,p in params),'trainable_tensor_names':[n for n,_ in params],'frozen_base_hashes_unchanged':before==after,'peak_memory_gb':mx.get_peak_memory()/1e9,'adapter_sha256':sha((ADAPTER/'adapters.safetensors').read_bytes()),'adapter_config_sha256':sha((ADAPTER/'adapter_config.json').read_bytes()),'base_revision':pin['revision'],'selected_checkpoint':'final600','final_accessed':False}
    (OUT/'training-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
