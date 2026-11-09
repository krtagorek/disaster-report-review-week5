# Development baselines — November 6, 2026

All three runs cover the same frozen 2,449 development records, in identical
EVENT_ID order. Only event narrative text is supplied as report-specific input.
No adapter, training, quantization, prompt search or final-set inference occurred.
Reserved final manifests were not opened by these evaluation scripts.

## Results

| Method | Accuracy | Macro-F1 | Invalid outputs |
| --- | ---: | ---: | ---: |
| Keyword | 89.02% | 88.10% | 0 |
| Raw completion diagnostic | 71.74% | 77.41% | 9 |
| Handout letter-choice | 26.46% | 26.85% | 0 |

The keyword baseline is the strongest measured method here. The handout-style
letter protocol predicts Winter Storm on 1,706 of 2,449 records, including 947
Thunderstorm Wind references. Its valid output rate is guaranteed by restricting
the choice to six letters; this is not evidence of reliable classification.
Do not use the weak letter score alone to manufacture a fine-tuning success.
Report the keyword and raw-completion comparisons alongside it.

## Protocols and boundaries

- Keyword: count predeclared regex matches per class; ties follow manifest label
  order. No match falls back to training-majority Thunderstorm Wind. No tuning
  followed development predictions.
- Raw diagnostic: one narrative-only raw prompt, no chat template or examples,
  greedy generation, 16-token maximum, model EOS stop. Strip surrounding
  whitespace and accept only a complete exact category name. Extra text or
  unknown categories count as invalid, not a fallback label.
- Handout baseline: notebook cell 16 first-step choice-logit scoring, adapted to
  A–F and six disaster classes. The shipped chat template is explicitly used
  for this reference protocol; it does not replace the raw training contract.
  Prefer a spaced letter if it is one token, as the notebook does. Verified
  all 14,694 prompt/choice boundaries and distinct IDs: A=362, B=425, C=356,
  D=422, E=468, F=434. These normalized choice probabilities are uncalibrated.

The raw diagnostic was declared and started before the user supplied the
original handout. After reading it, a separate letter plan was fixed before
letter predictions. Both results are preserved; this was a requirements
correction, not a search for a better development score. See
`data/baseline-plan-v1.json`, `data/letter-baseline-plan-v1.json` and
`docs/HANDOUT_ALIGNMENT.md`.

All six exact category names are multi-token. All 14,694 raw prompt/label
boundaries and EOS boundaries passed; no development inputs were truncated.
The raw diagnostic returned nine invalid outputs: five Microburst, three
Landspout and one Forest City. There were zero max-token-budget completions.
Do not infer that other input types will always yield a valid category.

## Per-class reports

Precision, recall and F1 below are percentages. Support is the reference count.
Invalid outputs contribute false negatives to the reference class.

### Keyword

| Reference | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Flash Flood | 99.38% | 73.96% | 84.81% | 434 |
| Tornado | 98.74% | 77.12% | 86.61% | 306 |
| Thunderstorm Wind | 83.10% | 99.60% | 90.60% | 1239 |
| Hail | 98.52% | 91.32% | 94.79% | 219 |
| Winter Storm | 100.00% | 70.48% | 82.68% | 210 |
| Wildfire | 80.39% | 100.00% | 89.13% | 41 |

Full confusion matrix and metrics: `data/evaluations/keyword-development-v1/metrics.json`.

### Raw completion diagnostic

| Reference | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Flash Flood | 46.71% | 94.93% | 62.61% | 434 |
| Tornado | 90.28% | 94.12% | 92.16% | 306 |
| Thunderstorm Wind | 94.61% | 51.01% | 66.28% | 1239 |
| Hail | 99.44% | 80.37% | 88.89% | 219 |
| Winter Storm | 63.22% | 99.05% | 77.18% | 210 |
| Wildfire | 63.08% | 100.00% | 77.36% | 41 |

Full confusion matrix and metrics: `data/evaluations/untuned-development-v1/metrics.json`.

### Handout letter-choice

| Reference | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Flash Flood | 84.24% | 32.03% | 46.41% | 434 |
| Tornado | 63.55% | 88.89% | 74.11% | 306 |
| Thunderstorm Wind | 100.00% | 0.81% | 1.60% | 1239 |
| Hail | 100.00% | 7.31% | 13.62% | 219 |
| Winter Storm | 12.19% | 99.05% | 21.71% | 210 |
| Wildfire | 2.42% | 7.32% | 3.64% | 41 |

Full confusion matrix and metrics: `data/evaluations/letter-development-v1/metrics.json`.

## Error patterns inspected

Keyword errors most often default to Wind: 102 Flash Flood, 70 Tornado and 62
Winter Storm references. Raw completion instead assigns 462 Wind references to
Flash Flood and 92 to Winter Storm. The letter protocol has very low Wind and
Hail recall and overwhelmingly selects Winter Storm. These observations describe
reference-label confusions; they do not establish that every source label is
semantically correct. No rows or labels were changed after predictions.

The letter baseline is protocol-sensitive and poor on this task. The current
checks establish faithful execution and consistent batching, not that the
shipped chat template is an optimal prompt for a pretrained base model. No
alternative letter token or prompt sweep was run to improve its result.

## Validation and runtime evidence

`data/evaluations/local-model-verification.json` records the exact model and
environment, finite GPU forward pass and token boundaries. The original
bfloat16 upstream weights were loaded locally, with no saved conversion or
quantization. `data/evaluations/validation-v1.json` confirms exact recomputation
of all metrics, signatures and prediction hashes, and complete unique
development ID alignment. First eight letter-scoring cases matched unbatched
predictions and logits exactly (maximum absolute logit difference 0.0).

Five unit tests passed for grouping, strict parsing, invalid-error accounting,
keyword tie/fallback behavior and narrative-only prompt construction. Weights,
source CSV, caches and the environment are ignored by Git.

Raw-completion inference took 132.0 seconds with 4.43 GB peak MLX memory;
letter inference took 148.0 seconds with 5.07 GB peak MLX memory. These are
local measured inference-loop times excluding model loading, source preparation
and verification, not a deployment latency/cost benchmark or a comparison
against a hosted service. See `docs/LOCAL_MODEL_SETUP.md` for setup/commands.

## Limits and next bounded milestone

Development is available for model decisions and is not the reserved final.
Class imbalance, imperfect reference labels, same-label duplicates and unknown
upstream pretraining overlap limit interpretation. Wildfire has only 41
development records from 27 components. Row-weighted scores do not imply that
all rows are independent, and closed-set accuracy does not measure arbitrary
input coverage. No confidence calibration or out-of-scope evaluation is claimed.

Next, complete one bounded LoRA cycle with frozen base weights, saved adapter
and real training/validation loss evidence. Verify completion-only loss masking
in the actual trainer, and declare the first configuration and any project
quality gates before candidate evaluation. Resolve the literal handout stack
versus existing local-MLX adaptation explicitly in the training plan; do not
claim LLaMA Board execution. Merge the adapter, run the five predeclared obvious
smoke cases before full merged evaluation, then evaluate base/adapter/merged
under both matched protocols on the same development IDs. Preserve the
reserved final for the predeclared final comparison; do not promise a gain.
Training or merge completion alone does not approve a model for the review app.
