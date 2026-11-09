# Start the next chat here

## User decision

On November 6, 2026 the user chose to leave Curator's Studio as it is and start
this separate Disaster Report Review project. The user wants a new Git project.
Do not resume Curator repairs or training, and do not revisit the shortlist
unless the user asks.

## What to build

A local historical disaster report classification/review application:

1. Load an archived report or paste report text.
2. Suggest one of the six declared event categories with a local open-weights model.
3. Let the reviewer confirm or correct the suggestion.
4. Save and export the review outcome, keeping the original report and source ID.

No generated impact summaries, inferred casualties or emergency dispatch.
The AI contribution is the classifier. Review state, queue operations and source
facts are handled by application code.

## Current foundation

- The source NOAA 2024 CSV snapshot is present at
  `data/raw/stormevents-2024.csv.gz`, ignored by Git.
- `data/source-manifest.json` records its official URL and SHA-256.
- `scripts/audit_source.py` is a standard-library source audit.
- The source has 69,801 rows. A preliminary 80-2,000-character narrative filter
  gives 16,332 candidate rows across the six selected types. This is only a
  length filter, not a reviewed training set.
- Wildfire is the smallest candidate class: 276 rows, 251 distinct normalized
  narratives in the preliminary check. Plan splits and sampling accordingly.
- Three identical normalized narratives occur under multiple selected labels;
  all six conflicting rows are now excluded in the frozen split manifests.
- Pre-split spot inspection and product review contract are complete; see
  `docs/CLASSIFICATION_CONTRACT.md` and `data/audits/candidate-inspection.json`.
  `scripts/inspect_candidates.py` reproduces 30 inspection examples and records
  six conflicting-label row exclusions before predictions. 16,326 candidates
  remain; this is not exhaustive semantic approval.
- Frozen grouped splits now exist at `data/splits/v1/`: 11,430 train,
  2,449 development, 2,447 reserved final; 3,646 connected components.
  See `docs/DATA_SPLITS.md` for policy, counts, limitations and reproduction.
  `python3 scripts/prepare_splits.py` verifies frozen bytes; never overwrite v1
  in response to predictions. Episode/exact/declared-near overlap checks passed.
- Pinned model loading, finite GPU inference and all development label/EOS
  boundaries verified. See `docs/LOCAL_MODEL_SETUP.md`.
- Full development baselines completed and validated: keyword 89.02% accuracy /
  88.10% macro-F1; raw-completion diagnostic 71.74% / 77.41% (9 invalid);
  handout-style constrained letter baseline 26.46% / 26.85% (valid by design).
  See `docs/BASELINES.md` and `data/evaluations/`. Do not hide the weak letter
  result or use it alone to claim training value. No prompt sweep occurred.
- Original handout supplied at `/Users/tagore/Downloads/Week 5 Project Handout
  (Oct 2026).pdf`; all four pages and linked notebook reviewed. Reference hashes
  and exact example/local differences: `data/handout-reference.json` and
  `docs/HANDOUT_ALIGNMENT.md`. Do not claim LLaMA Board execution.
- Reserved final was not opened or evaluated during the baseline milestone.
- One actual local LoRA run now completed: `data/training/lora-v1/plan.json`,
  600 updates, batch size 4, rank 8 q/v adapters in the last 16 layers; 2,400 distinct training
  examples. Frozen base tensor hashes stayed unchanged. Adapter lives in ignored
  `runtime/lora-v1/`; standalone BF16 artifact in `runtime/merged-lora-v1/`.
- Loss reviewed: fixed 240-report class-balanced development monitor decreased
  0.201500 to 0.046790 after saved final600 reload, with fluctuations and a lower
  logged point at update 499. No checkpoint search or second run occurred.
- Saved merged artifact passed all five predeclared smoke cases before full
  classification evaluation. Adapter and merged raw development results both
  98.73% accuracy / 98.79% macro-F1, zero invalid, zero label disagreements.
- Matched letter evaluations and independent recomputation finished: adapter
  64.03% accuracy / 56.54% macro-F1, merged 63.33% / 56.20% (base letter
  26.46% / 26.85%). BF16 native fusion changed 48 letter labels, max absolute
  choice-logit drift 0.25; raw labels, completion text and tokens all matched.
  This protocol is weaker and has class regressions; do not substitute it for
  the declared raw application protocol or hide its result.
- Full per-class reports, confusion matrices, merge drift and reviewed error
  examples are saved. Read `docs/LORA_RUN.md`. Figures and provenance are in
  `data/training/lora-v1/`; candidate predictions in `data/evaluations/`.
- Training/merge/evaluation are complete; do not start a second run. The run
  evidence audit verifies hashes and actual training/monitor membership.
- Local review app is complete in `review_app/`. Start with
  `.venv/bin/python -m review_app` or `start-review.command`, then open
  `http://127.0.0.1:8877`. Bind is loopback-only; do not use occupied port 8765
  or stop other applications. Normal DB is `runtime/review-app.sqlite3`.
- Real model classification, explicit confirm/correct/unresolved/pending,
  saved/reopened reviews and filtered CSV/JSON exports are verified. Original
  text/source metadata and first model suggestion are preserved. Pending reports
  without a suggestion can receive one after reopening. Source catalog is
  development-only; pasted text never inherits similar reports' facts.
- Actual served endpoint evaluated one report per request over all 2,449 development
  rows: 98.69% accuracy / 98.76% macro-F1, zero invalid. It differs from offline
  merged batch size 8 on one label: 1213326 (Flash Flood reference and offline prediction,
  served Thunderstorm Wind). All five served smoke cases passed first.
- Twelve tests passed. Actual HTTP exports preserved text/source/model provenance;
  three QA saved reviews survived a full process restart unchanged. Browser
  workflow/desktop/narrow layout checks passed; screenshot in
  `docs/assets/review-desk.png`. Read `docs/REVIEW_APP.md` for details.
- Tests/evaluation used only `runtime/review-qa.sqlite3`; the normal workspace DB
  was not populated. Owned test server is stopped at completion. Curator's
  files/service were not modified. No hosted publication occurred.
- App/serving provenance is frozen by `scripts/audit_review_app.py` in
  `data/evaluations/review-app-evidence.json`. Existing LoRA evidence remains
  hash-identical. No training or dataset relabeling happened in the app milestone.
- Reserved final is still untouched; no universal input coverage, calibrated
  confidence or real-time warning performance is claimed.

## Model and machine

Apple M5, 24 GB memory. `Qwen/Qwen3-1.7B-Base` revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1` is downloaded in ignored
`runtime/models/qwen3-1.7b-base`, SHA-256 verified, Apache 2.0 license checked.
Use this repository's `.venv` (Python 3.11.16), pinned in
`requirements-model.lock`. No quantization or dtype change; parameters are BF16.
GPU peak memory: 4.43 GB raw baseline / 5.07 GB letter baseline / 5.82 GB training.
The saved LoRA tensors are FP32; base and merged weights remain BF16.

Raw training contract: narrative-only prompt from `scripts/baseline_common.py`,
no chat template, target is the full exact label plus EOS 151643; mask prompt
loss. Actual trainer masking was verified before this run. The handout-style letter
baseline explicitly uses the shipped chat template and separate choice-logit
scoring. Keep both comparison protocols matched across base/adapter/merged/served.
The user explicitly authorized the local MLX/custom-data adaptation on November 6,
provided submission criteria are met. The one-run training/loss/merge/smoke/matched
evaluation milestone is now complete; no LLaMA Board execution is claimed.

## Next bounded milestone

1. Completed: bounded label/text spot inspection and six-category review contract.
2. Completed: grouped splits frozen and validated; see docs/DATA_SPLITS.md.
3. Completed: local loading, boundaries and keyword/untuned development baselines.
4. Completed: one bounded LoRA cycle, actual loss review, merged artifact,
   five smoke tests and matched full-development evaluation. Preserve v1; no
   sweep or additional training is authorized by completion of this milestone.
5. Completed: local classify/human review/save/reopen/export workflow with the
   actual saved merged raw model and source IDs/attribution preserved.
6. Completed: full development evaluation through the actual single-report HTTP
   endpoint, including smoke, per-class metrics, confusion matrix and batch drift.
7. Next: prepare final submission assets and licensing/reproducibility instructions,
   review the GitHub package for ignored raw data/weights/runtime DBs, and record
   Loom/demo/submission links. Do not publish without user authorization. Preserve
   reserved final until a separately authorized, predeclared evaluation.

The first useful local workflow checkpoint is complete. Do not expand features
or launch further model experiments while preparing the submission.

## Submission

Deadline: Sunday November 8, 2026, 11:59 PM Pacific. The custom-project option is
GitHub assets plus a Loom video. Real LoRA execution, loss review,
merge, five smoke tests and matched development per-class evaluation now have
local evidence. App and served evaluation are also complete. Final packaging/licensing,
publication and Loom remain to be done.
The handout does not prescribe an 85% deployment threshold; agree any project
quality gates before seeing candidate results. Never guarantee a training gain.

The user values clear instructions, stable goals and usable product behavior.
They are willing to pay for useful data, but no purchase is needed for the
currently selected NOAA source. Do not create unnecessary approval loops.

## Checkpoint before prospective enhancements

At the user's request, preserve the current working version in a local Git
commit tagged `working-review-app-v1`. See `docs/CHECKPOINTS.md`. Do not move the
tag or overwrite v1 model artifacts. Saved review data has a separate ignored
SQLite backup under `runtime/checkpoints/working-review-app-v1/`. The user is
considering small functionality additions; no specific new feature is authorized
yet. Keep the original submission candidate available. Nothing has been pushed.

## Bounded demo polish branch

The user authorized only three UI improvements on `feature/demo-polish-v1`:
live state counts/filters, explicit oldest-next-pending navigation and a recorded
model evidence panel. See `docs/DEMO_POLISH.md`. The original tag/main remain at
`0297d9d`; do not move them or overwrite v1 evidence. Backend/model/prompt/data
are byte-identical. Five Node helper tests, the twelve Python tests, five real
merged-model smoke cases and source-snapshot/protection checks pass.

QA used `runtime/demo-polish-qa.sqlite3` and port 8891. Browser checks passed
empty/populated counts, state-filter export links, explicit save→next,
no-suggestion pending, no-pending disabled state, unresolved counting and the
recorded evidence panel. Native Escape canceled the unsaved-change dialog and
preserved the note; the earlier automation limitation was resolved. See
`docs/demo-polish/verification.json` and screenshots. The QA server is stopped
at completion; normal port 8877 and its database are untouched. No new training
or benchmark is authorized. Next work remains submission packaging and Loom.
