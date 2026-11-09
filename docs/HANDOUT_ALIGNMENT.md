# Week 5 handout alignment

Read all four pages of the user-supplied Week 5 Project Handout (Oct 2026).pdf
on November 6, 2026, including visual verification. Its SHA-256 and the linked
notebook snapshot SHA-256 are recorded in `data/handout-reference.json`.
Notebook reference: https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router

This is the custom-project route: GitHub assets and a Loom video. This repository
already specifies NOAA reports, six categories and local Apple M5/MLX execution.
The example handout instead uses IT tickets, seven categories, an 80/20 split,
Colab Tesla T4, ShareGPT dataset registration and LLaMA Factory/LLaMA Board.
Those example-specific steps were not executed literally. On November 6, the user
explicitly accepted the local MLX/custom-data adaptation as long as submission
criteria are met. Describe the implementation accurately; do not claim LLaMA Board
execution. The required actual LoRA, loss review, merge, five smoke cases and
matched development evaluation now have local evidence in `LORA_RUN.md`.

| Handout requirement or example step | Disaster Report Review evidence / remaining work |
| --- | --- |
| Labelled classification data with a held-out evaluation set | Frozen episode/duplicate-grouped NOAA train/development/final manifests. Grouped 70/15/15 replaces example 80/20 to avoid event leakage and preserve a reserved final. |
| Qwen/Qwen3-1.7B-Base | Exact upstream revision, file hashes and license in data/model-manifest.json. |
| Baseline without task-specific training | Full development constrained-letter baseline follows notebook cell 16, adapted to six disaster labels. |
| LoRA training with frozen base weights | Completed one MLX run: 600 updates, 2,400 distinct train reports, 917,504 LoRA parameters; frozen base hashes unchanged. No LLaMA Board training claimed. |
| Review actual loss curve | Recorded losses and visually inspected loss-curve.png; fixed monitor 0.201500 → 0.046790, fluctuating; final checkpoint selected before training. |
| Merge saved adapter into standalone model | Completed native BF16 fusion; standalone saved artifact hashed and reloaded; tokenizer parity and full matched evaluation recorded. |
| Five obvious smoke cases before merged full validation | All five predeclared cases passed on actual saved merged artifact before full candidate evaluation. |
| Accuracy, class precision/recall/F1/support, confusion matrix | Saved development baseline reports include all these plus macro-F1 and invalid outputs. Rows=true, columns=predicted. |
| Compare same base model against fine-tuned model on same held-out examples | Full base/adapter/merged development comparisons completed under both matched protocols. Raw merged 98.73% accuracy, zero invalid; letter 63.33%, with regressions and 48 merge-label changes documented. |
| Practical downstream workflow | Classifier plus human confirm/correct/save/export; local app now verified, including unresolved review, save/reopen and exports. No generated disaster briefings. |
| Custom submission GitHub assets and Loom | Pending; remain local until publication is requested. Deadline November 8, 2026, 11:59 PM Pacific. |

The handout's notebook uses constrained letter logits for the baseline and
prefix-matched label generation for the merged model. Here the handout-style
letter baseline is preserved, with explicit single-token and boundary checks.
An additional strict raw-completion diagnostic was already declared and started
before the original handout arrived; retain it openly rather than discard a weak
result. There was no prompt search. Do not treat six-choice normalized scores as
calibrated confidence, and do not copy the example's fallback-to-general behavior
into a six-class application that has no general class.

Reserved final remains excluded from this milestone. No final manifest was read
by the baseline code. No data was relabeled based on predictions. One actual
training/merge/development
cycle and actual local served-model evaluation are complete. See `REVIEW_APP.md`
for workflow/served evidence. Hosted publication and final submission remain
incomplete.
