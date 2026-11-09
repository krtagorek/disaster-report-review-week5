# Classification and review contract

## Scope and input

The six exact reference labels are Flash Flood, Tornado, Thunderstorm Wind,
Hail, Winter Storm and Wildfire, in the order in data/source-manifest.json.
Only EVENT_NARRATIVE is model input. EVENT_TYPE is the target; episode text,
source metadata and structured answer columns are excluded from model input.
These are project reference categories, not comprehensive hazard coverage.

For review purposes: Flash Flood concerns rapid flooding; Tornado concerns
reported tornado occurrence; Thunderstorm Wind concerns storm wind or associated
damage; Hail concerns hail reports; Winter Storm concerns winter precipitation
or storm conditions; Wildfire concerns vegetation fire. These descriptions guide
review; they are not a replacement for NOAA's full event definitions or a rule
that every selected narrative contains enough evidence to distinguish its label.

## Output and reviewer behavior

A valid classifier completion is exactly one of the six labels, with boundary
handling to be verified against the pinned tokenizer before training. Invalid
output is a classification failure and requires manual review; never silently
map it to a label. Preserve the raw completion for diagnosis locally.

Display the original report beside the suggestion. Suggestions are unconfirmed
until a reviewer confirms or corrects them. Do not present model scores as
calibrated confidence without evidence. Review states are pending, confirmed,
corrected and unresolved. Confirmed/corrected records require a selected label;
unresolved records may have no selected label and a reason such as ambiguous,
out of scope, insufficient text or invalid model output. These reasons are review
states, not extra model classes. Empty input does not invoke the model.

Pasted text may mention multiple hazards, contain too little evidence or concern
an unrepresented category. Allow unresolved review without forcing a category.
The six-class model alone cannot reliably detect out-of-scope input. Do not claim
automatic coverage or abstention performance before evaluating it.

Save/export the original text, source ID and attribution when linked, suggestion,
review state, reviewer-selected label, review note and timestamp, and model/run
identifier. Keep NOAA reference label distinct from the suggestion and reviewed
label. Pasted reports have no NOAA ID unless explicitly linked; similarity does
not transfer metadata. Preserve source facts verbatim without generating impacts.

## Pre-split inspection completed November 6, 2026

Run `python3 scripts/inspect_candidates.py` to reproduce the inspection against
the pinned checksum. The report is data/audits/candidate-inspection.json.
Five examples per class are selected by SHA-256 of EVENT_ID, independently of
predictions. All 30 examples were inspected as a bounded spot check.

Observations:

- Flood descriptions 1199121 and 1192387 describe inundation without explicit
  rapid onset. Keep their source labels, flag the text-only ambiguity in dataset
  limitations, and do not invent timing from metadata.
- Wind examples 1203975 and 1156490 describe damage without explicitly naming a
  thunderstorm. Damage alone cannot uniquely establish this category.
- Hail example 1182181 also describes wind. Multi-hazard narratives need human
  review; a single source label need not cover every hazard in the text.
- Winter Storm example 1158307 ends with an incomplete wind-gust sentence.
  Length filtering does not establish completeness or semantic quality.
- Wildfire examples include repeated updates about named fires. Episode grouping
  alone may miss related narratives across episodes; near-duplicate grouping is
  required before freezing any split.

Three normalized narratives have both Hail and Thunderstorm Wind labels:
1180057/1180058, 1191568/1191570 and 1193768/1193769. Exclude all six rows from
supervised splits before predictions rather than choose an arbitrary target.
Their IDs and reasons are recorded by the inspection script. Other sampled rows
are not relabeled or individually excluded based on this small spot check.

Remaining candidate counts after these exclusions: Flash Flood 2,890; Tornado
2,042; Thunderstorm Wind 8,257; Hail 1,460; Winter Storm 1,401; Wildfire 276.
This is 16,326 candidates, not a fully approved training set. Wind dominates;
Wildfire is small. Report class counts and macro/per-class metrics alongside
accuracy. No model quality gate or deployment approval is implied.

## Implementation status

Grouped splits, model boundaries, one LoRA cycle, merged smoke/development
comparisons and the local review app are complete. See `DATA_SPLITS.md`,
`LORA_RUN.md` and `REVIEW_APP.md`. The app keeps model suggestions pending until
explicit human review, preserves source/reference versus reviewed labels,
allows unresolved review without a forced category, and exports original text
and provenance. Actual served development inference is evaluated separately
from offline batching. Reserved final remains untouched.
