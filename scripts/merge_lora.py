"""Native BF16 fusion of the final adapter into a separate standalone artifact."""
import json
import shutil
from mlx.utils import tree_flatten,tree_unflatten
from mlx_lm.utils import save
from lora_artifacts import ROOT,RUN,MERGED,load_candidate
from baseline_common import sha


def main():
    if MERGED.exists() or (RUN/'merged-manifest.json').exists():
        raise SystemExit('Refusing merged artifact overwrite')
    model,tok,fingerprint=load_candidate('adapter')
    pairs=[(n,m.fuse()) for n,m in model.named_modules() if hasattr(m,'fuse')]
    assert len(pairs)==32
    model.update_modules(tree_unflatten(pairs))
    params=tree_flatten(model.parameters())
    assert not any('.lora_' in n for n,_ in params)
    dtypes=sorted({str(v.dtype) for _,v in params})
    assert dtypes==['mlx.core.bfloat16']
    pin=json.loads((ROOT/'data/model-manifest.json').read_text())
    source=ROOT/pin['local_directory']
    config=json.loads((source/'config.json').read_text())
    save(MERGED,str(source),model,tok,config,donate_model=False)
    shutil.copyfile(source/'LICENSE',MERGED/'LICENSE')
    (MERGED/'README.md').write_text('---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-1.7B-Base\n---\n# Disaster Report Review experimental merged artifact\n\nDerived locally from pinned Qwen/Qwen3-1.7B-Base and final lora-v1 adapter.\nBF16 native MLX fusion; no quantization. Not approved for deployment.\nSee project training/evaluation records for results and limitations.\n')
    assert not (MERGED/'adapter_config.json').exists()
    from transformers import AutoTokenizer
    from baseline_common import development_rows,prompt
    from letter_choice import letter_prompt
    original_tok=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
    saved_tok=AutoTokenizer.from_pretrained(str(MERGED),local_files_only=True,trust_remote_code=False)
    rows=development_rows()
    for row in rows:
        for p in [prompt(row['text']),letter_prompt(original_tok,row['text'])]:
            assert original_tok.encode(p,add_special_tokens=False)==saved_tok.encode(p,add_special_tokens=False)
        assert letter_prompt(original_tok,row['text'])==letter_prompt(saved_tok,row['text'])
    assert original_tok.eos_token_id==saved_tok.eos_token_id==151643
    manifest={'base_revision':pin['revision'],'adapter_sha256':fingerprint,'fused_linear_layers':len(pairs),'parameter_count':sum(v.size for _,v in params),'dtypes':dtypes,'quantization':None,'tokenizer_parity_development_rows':len(rows),'tokenizer_protocols':['raw','letter'],'rounding':'MLX native fuse casts delta to BF16 and adds to BF16 base; potential merge drift must be measured','files_sha256':{str(p.relative_to(MERGED)):sha(p.read_bytes()) for p in sorted(MERGED.rglob('*')) if p.is_file()}}
    (RUN/'merged-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
