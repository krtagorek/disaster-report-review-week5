"""Measure final-checkpoint monitor loss and summarize the one run, no retraining."""
import json
import math
from baseline_common import development_rows,prompt,sha
from lora_artifacts import RUN,load_candidate
from train_lora import completion_loss
from mlx_lm.tuner.trainer import evaluate


def main():
    path=RUN/'loss-review.json'
    if path.exists():raise SystemExit('Refusing loss-review overwrite')
    model,tok,fingerprint=load_candidate('adapter')
    ids=json.loads((RUN/'loss-monitor-ids.json').read_text())
    wanted=set(ids)
    rows=[r for r in development_rows() if r['event_id'] in wanted]
    assert [r['event_id'] for r in rows]==ids and len(rows)==240
    data=[]
    for r in rows:
        prefix=tok.encode(prompt(r['text']),add_special_tokens=False)
        target=tok.encode(r['reference_label'],add_special_tokens=False)+[tok.eos_token_id]
        data.append((prefix+target,len(prefix)))
    final_loss=evaluate(model,data,4,-1,max_seq_length=1024,loss=completion_loss)
    assert math.isfinite(final_loss)
    logs=[json.loads(l) for l in (RUN/'losses.jsonl').read_text().splitlines()]
    training=[r for r in logs if r['kind']=='train']
    monitor=[r for r in logs if r['kind']=='development_monitor']
    assert len(training)==30 and training[-1]['iteration']==600
    assert [r['iteration'] for r in monitor]==[0,99,199,299,399,499,599]
    assert all(math.isfinite(r['train_loss']) for r in training)
    assert all(math.isfinite(r['val_loss']) for r in monitor)
    report={'adapter_sha256':fingerprint,'training_log_sha256':sha((RUN/'losses.jsonl').read_bytes()),'loss_units':'mean negative log likelihood in nats per supervised label/EOS token','training_points':len(training),'training_first_20_step_mean':training[0]['train_loss'],'training_last_20_step_mean':training[-1]['train_loss'],'development_monitor_rows':240,'monitor_class_balance':'40 per class; not full-development population-weighted loss','monitor_points':monitor,'initial_monitor_loss':monitor[0]['val_loss'],'final600_monitor_loss':final_loss,'best_logged_monitor_loss':min(r['val_loss'] for r in monitor),'final_minus_best_logged_monitor':final_loss-min(r['val_loss'] for r in monitor),'final_checkpoint_selection':'predeclared final600, not loss-minimizing checkpoint selection','finite_losses':True,'trainer_iteration_note':'stock trainer callbacks report completed updates before next update:0,99,...599; separate final600 loss measured after saved adapter reload','interpretation':'Loss evidence alone does not prove classification usefulness. Inspect full-development matched metrics and all class regressions; no additional training is triggered.'}
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
