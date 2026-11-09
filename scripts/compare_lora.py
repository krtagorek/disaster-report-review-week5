"""Recompute same-ID development results and audit base/adapter/merge differences."""
import collections
import json
from baseline_common import ROOT,LABELS,development_rows,metrics,sha
from lora_artifacts import RUN


def main():
    output=RUN/'comparison.json'
    if output.exists():raise SystemExit('Refusing comparison overwrite')
    expected=[(r['event_id'],r['reference_label']) for r in development_rows()]
    names={'keyword':'keyword-development-v1','base_raw':'untuned-development-v1','base_letter':'letter-development-v1','adapter_raw':'adapter-lora-v1-raw','adapter_letter':'adapter-lora-v1-letter','merged_raw':'merged-lora-v1-raw','merged_letter':'merged-lora-v1-letter'}
    reports={};predictions={};hashes={}
    for key,name in names.items():
        path=ROOT/'data/evaluations'/name
        data=(path/'predictions.jsonl').read_bytes()
        records=[json.loads(l) for l in data.splitlines()]
        assert [(r['event_id'],r['reference_label']) for r in records]==expected
        saved=json.loads((path/'metrics.json').read_text())
        recomputed=metrics(records)
        assert all(saved[k]==v for k,v in recomputed.items())
        assert saved['predictions_sha256']==sha(data)
        if (path/'run-signature.json').exists():
            sig=json.loads((path/'run-signature.json').read_text())
            assert all(sha((ROOT/p).read_bytes())==h for p,h in sig.items() if p!='artifact_fingerprint'),name
        reports[key]=recomputed;predictions[key]=records
        hashes[key]={'predictions_sha256':sha(data),'metrics_sha256':sha((path/'metrics.json').read_bytes())}
    differences={}
    for protocol in ('raw','letter'):
        a,b=predictions['adapter_'+protocol],predictions['merged_'+protocol]
        events=[{'event_id':x['event_id'],'reference_label':x['reference_label'],'adapter_prediction':x['prediction'],'merged_prediction':y['prediction']} for x,y in zip(a,b) if x['prediction']!=y['prediction']]
        differences[protocol]={'prediction_disagreements':len(events),'events':events,'agreement':1-len(events)/len(a)}
        if protocol=='raw':
            differences[protocol]['raw_completion_differences']=sum(x['raw_completion']!=y['raw_completion'] for x,y in zip(a,b))
            differences[protocol]['token_sequence_differences']=sum(x['token_ids']!=y['token_ids'] for x,y in zip(a,b))
        else:
            drifts=[abs(x-y) for ra,rb in zip(a,b) for x,y in zip(ra['choice_logits'],rb['choice_logits'])]
            differences[protocol]['choice_logit_max_absolute_difference']=max(drifts)
            differences[protocol]['choice_logit_mean_absolute_difference']=sum(drifts)/len(drifts)
    deltas={}
    for kind in ('adapter','merged'):
        for protocol in ('raw','letter'):
            key=kind+'_'+protocol
            base=reports['base_'+protocol]
            result=reports[key]
            deltas[key]={'accuracy_vs_matched_base':result['accuracy']-base['accuracy'],'macro_f1_vs_matched_base':result['macro_f1']-base['macro_f1'],'accuracy_vs_keyword':result['accuracy']-reports['keyword']['accuracy'],'per_class_f1_vs_matched_base':{c:result['per_class'][c]['f1']-base['per_class'][c]['f1'] for c in LABELS},'per_class_f1_vs_keyword':{c:result['per_class'][c]['f1']-reports['keyword']['per_class'][c]['f1'] for c in LABELS}}
    smoke=json.loads((RUN/'merged-smoke.json').read_text())
    assert smoke['passed']==smoke['total']==5
    result={'n':2449,'split':'development','identical_ids_labels_order':True,'metrics_recomputed':True,'reports':reports,'matched_deltas':deltas,'merge_drift':differences,'evidence_hashes':hashes,'smoke_passed':5,'final_accessed':False,'no_deployment_approval':True,'limitations':'Development-only row-weighted metrics; same-label duplicates, imperfect reference labels, class imbalance and unknown pretraining overlap remain. Cross-format differences are not attributable solely to training.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'metrics':{k:{m:r[m] for m in ['accuracy','macro_f1','invalid_count']} for k,r in reports.items()},'merge_drift':differences,'matched_deltas':deltas},indent=2))


if __name__=='__main__':main()
