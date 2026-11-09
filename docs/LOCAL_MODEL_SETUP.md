# Local model setup and completion contract

Verified on Apple M5, 24 GB, macOS 26.6, CPython 3.11.16. Use the repository's
own `.venv`; do not activate another project's environment. Packages are fully
version-pinned in `requirements-model.lock` (including MLX 0.32.4 and MLX-LM
0.32.2). No hosted inference is used.

```sh
UV_CACHE_DIR="$PWD/work/uv-cache" uv venv --python /opt/homebrew/bin/python3.11 .venv
UV_CACHE_DIR="$PWD/work/uv-cache" uv pip install --python .venv/bin/python -r requirements-model.lock
.venv/bin/python scripts/download_model.py
```

The downloader resolves the model only when `data/model-manifest.json` does not
exist. The checked-in manifest pins Qwen/Qwen3-1.7B-Base revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, with SHA-256 of every downloaded
file. It downloads into ignored `runtime/models/`, with caches in ignored
`work/`. It never uploads. The upstream model card and included LICENSE declare
Apache 2.0:
https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/README.md

Subsequent verification/evaluation scripts set Hugging Face offline mode and
load the local files after hash checks. No remote model code is trusted. MLX
loads all 1,720,574,976 parameters as bfloat16; no quantization, adapter, saved
conversion or dtype change was introduced. A real GPU forward pass produced
finite logits of expected vocabulary dimension 151,936. This is loading and
inference evidence, not proof that training or merging works.

## Raw completion contract

`scripts/baseline_common.py:prompt` renders the six allowed names, a JSON-quoted
original event narrative, and ends with `\nCategory:\n`. Narrative text is the
only report-specific input. Do not inject source ID, reference label, episode
narrative or structured source fields. No chat template or few-shot examples
are used for this raw completion path.

For every one of 2,449 development prompts and all six labels, verified:

```
encode(prompt + label) == encode(prompt) + encode(label)
encode(prompt + label + eos) == encode(prompt) + encode(label) + [151643]
```

No implicit BOS is added. EOS is `<|endoftext|>` (151643). Development prompts
span 59–483 tokens, below the model context limit; none were truncated.

| Exact label | Label token IDs |
| --- | --- |
| Flash Flood | 23876, 56127 |
| Tornado | 51, 62943 |
| Thunderstorm Wind | 95912, 26413, 22087 |
| Hail | 39, 604 |
| Winter Storm | 66188, 21862 |
| Wildfire | 40603, 10796 |

Future training must mask prompt loss and supervise the entire label plus EOS.
Verify the actual trainer implements this contract before launching LoRA; the
boundary check is not evidence that a future dataset/trainer already does so.
Future adapter, merged and served raw-completion evaluation must use this same
prompt, tokenizer, greedy decoding and exact-label parser. All labels are
multi-token; scoring only the first token is not full-label likelihood.

## Handout-style letter contract

The handout's baseline is a separate, explicit multiple-choice method. See
`scripts/letter_choice.py`. It uses the shipped chat template for the handout's
system/user prompt, adapts the task and mapping to six disaster classes, and
compares only first-step logits for A–F. The notebook's preference for a
single-token spaced letter is retained, with single-token, uniqueness and
prompt/choice boundary assertions. This explicit comparison protocol does not
silently change the raw training format. Both protocols have separate plans
and saved development results. Six-way normalized scores are not calibrated
confidence and cannot detect out-of-scope reports.

## First-run and verification commands

After the environment and pinned files exist, first-run commands are:

```sh
.venv/bin/python scripts/verify_local_model.py
.venv/bin/python scripts/run_keyword_baseline.py
.venv/bin/python scripts/run_model_baseline.py
.venv/bin/python scripts/run_letter_baseline.py
.venv/bin/python scripts/validate_baselines.py --check-letter-batching
```

Completed v1 evidence refuses overwrite. Model baseline runners can resume an
interrupted run only if the saved code/config/model signatures match and saved
IDs match the development order. To independently repeat a completed experiment,
preserve v1 and create an explicitly versioned run rather than delete evidence.
The validator recomputes saved metrics and checks signatures; its own evidence
also refuses overwrite. Unit tests can be rerun without model loading:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Do not run the split builder as part of baseline evaluation: the development
loader checks only the pinned source and development manifest, and never opens
the reserved final manifest. The source CSV is scanned only to select the
explicit development IDs. Final data is excluded from all model calls here.
