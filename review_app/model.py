"""A single dedicated worker owns all MLX loading and inference operations."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from .core import ROOT, Problem, valid_text, sha


class ServedModel:
    def __init__(self):
        self.worker = ThreadPoolExecutor(max_workers=1,thread_name_prefix='merged-model')
        self.metadata = self.worker.submit(self._load).result()

    def _load(self):
        os.environ['HF_HOME']=str(ROOT/'work/huggingface')
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        from mlx_lm import load
        from evaluate_lora import raw_predict
        manifest_path=ROOT/'data/training/lora-v1/merged-manifest.json'
        manifest=json.loads(manifest_path.read_text())
        directory=ROOT/'runtime/merged-lora-v1'
        for name,expected in manifest['files_sha256'].items():
            if sha((directory/name).read_bytes())!=expected:
                raise RuntimeError('Merged artifact checksum mismatch: '+name)
        self.model,self.tokenizer=load(str(directory),trust_remote_code=False,tokenizer_config={'trust_remote_code':False})
        self.model.eval()
        import mlx.core as mx
        mx.random.seed(0)
        self.predict=raw_predict
        return {'run_id':'merged-lora-v1/raw','artifact_fingerprint':sha(manifest_path.read_bytes()),
                'base_revision':manifest['base_revision'],'quantization':None,'dtype':'bfloat16',
                'prompt_sha256':sha((ROOT/'scripts/baseline_common.py').read_bytes()),
                'predictor_sha256':sha((ROOT/'scripts/evaluate_lora.py').read_bytes()),
                'decoding':{'temperature':0,'max_tokens':16,'stop':'EOS151643','parser':'exact full label after whitespace strip'},
                'serving_code_sha256':{name:sha((ROOT/'review_app'/name).read_bytes()) for name in ['core.py','model.py','server.py']},
                'serving':'one report per HTTP request; serial dedicated MLX worker'}

    def classify(self,text):
        valid_text(text)
        if len(text.strip())>2000:
            raise Problem('Classification supports up to 2,000 characters. Save this report as unresolved for manual review.')
        return self.worker.submit(self._classify,text).result()

    def _classify(self,text):
        from baseline_common import prompt
        if len(self.tokenizer.encode(prompt(text),add_special_tokens=False))>1024:
            raise Problem('This report exceeds the tested token budget. Save it as unresolved for manual review.')
        result=self.predict(self.model,self.tokenizer,[text])[0]
        return {**result,'model':self.metadata}

    def close(self):
        self.worker.shutdown(wait=True)
