"""Sequential GPU evaluation of attached and reloaded merged models, no retrain."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'data/training/lora-v1'


def main():
    smoke=json.loads((RUN/'merged-smoke.json').read_text())
    assert smoke['passed']==smoke['total']==5
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for artifact,protocol in [('adapter','raw'),('merged','raw'),('adapter','letter'),('merged','letter')]:
        evidence=ROOT/f'data/evaluations/{artifact}-lora-v1-{protocol}/metrics.json'
        if evidence.exists():
            print(f'Preserving completed {artifact}/{protocol}; comparison will validate it',flush=True)
            continue
        log=ROOT/f'work/lora-v1-{artifact}-{protocol}.log'
        command=[sys.executable,'-u','scripts/evaluate_lora.py','--artifact',artifact,'--protocol',protocol]
        print(f'Starting {artifact}/{protocol}',flush=True)
        start=time.monotonic()
        with log.open('a') as f:
            subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        record={'artifact':artifact,'protocol':protocol,'seconds':time.monotonic()-start,'metrics_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}
        with (RUN/'evaluation-stages.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print('Completed '+json.dumps(record),flush=True)
    subprocess.run([sys.executable,'scripts/compare_lora.py'],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
