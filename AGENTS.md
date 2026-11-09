# Project instructions

## Keep the project separate

This is **Disaster Report Review**, a new project and Git repository. Curator's
Studio is parked. Do not modify its files, reuse its SQLite database, import its
backend, or continue its experiments. Do not work from the old Codex research
folder. Make changes only in this project unless explicitly authorized.

## Stable product scope

Build a historical disaster report classifier and human-review desk. A report
receives one of the six initial NOAA event-type categories. A reviewer can
confirm/correct the classification, save it, and export a review queue.

This is a deliberate, bounded version of the earlier disaster-project idea.
Do not expand into narrative impact extraction, casualty triage, live warnings,
autonomous emergency response, general chat, or generated factual briefings.

Show the original report beside the classification. Preserve source IDs and
attribution. Source metadata can accompany its own linked record; similarity
does not establish that a newly submitted report has the same facts. Support
human correction and manual review without claiming universal input coverage.

## Data and evaluation discipline

- Input to the model is event narrative text. `EVENT_TYPE` is the reference
  target and must not be supplied as an input field. Keep structured answer
  columns and the broader episode narrative out of the first classifier input.
- Natural hazard names in the narrative are legitimate evidence. Include a
  simple keyword baseline rather than hiding easy examples to manufacture gains.
- Group related events by `EPISODE_ID` before splitting. Connect exact and
  near-duplicate narratives across episodes as needed to prevent leakage.
- Check label/text alignment and ambiguous or conflicting records. Record
  exclusions before predictions. NOAA labels are references, not infallible truth.
- Freeze development and reserved final sets. Do not train on or relabel final
  examples in response to predictions. Report class imbalance and dataset limits.
- Measure accuracy, macro-F1, per-class precision/recall/F1 and confusion matrices.
  Training loss alone does not prove product usefulness.
- Separate the assignment's training/merge/evaluation objectives from a decision
  to deploy a particular candidate. An experimental merged artifact is not
  automatically an approved app model.

## Model and delivery

Proposed model: `Qwen/Qwen3-1.7B-Base`; proposed framework: MLX on Apple M5/24 GB.
Use a separate environment in this repository. Pin revisions and verify raw
completion/label token boundaries before training a base model. Do not blindly
apply an instruction-model chat template.

Keep baseline, adapter, merged and served comparisons matched. Record changes
caused by conversion or quantization. Run five predeclared obvious smoke cases
and evaluate the actual merged/served artifact. Keep model weights, credentials,
raw data and runtime databases out of Git.

Do not start unrelated model sweeps or extra features. Establish one working
end-to-end path first. Report failures precisely, with files and evidence; do
not portray incremental component scores as a finished product.

The Week 5 deadline is November 8, 2026 at 11:59 PM Pacific. Required delivery
includes a GitHub project with assets and a Loom demonstration. All work remains
local unless the user explicitly approves a separate hosted comparison.
