# Frozen dataset splits v1

Prepared November 6, 2026 without predictions or model calls. Source checksum
and artifact hashes are in `data/splits/v1/freeze.json`. These are ID manifests;
original narratives remain in the ignored pinned source, not duplicated into
training files in Git. Join by EVENT_ID and verify narrative_sha256 before use.
Only EVENT_NARRATIVE becomes input; reference_label is a separate target.

## Eligibility and grouping policy

Use the six manifest classes and stripped narrative lengths 80–2,000 inclusive.
Exclude all six rows in the three exact-normalized cross-label conflicts
identified by the pre-split audit. No additional semantic exclusions or label
changes were introduced. Require unique event IDs and nonempty episode IDs.

Connect eligible rows sharing EPISODE_ID or exact casefolded, whitespace-normalized
narrative. Also connect near duplicates across all eligible rows, regardless of
label or episode: replace numeric strings (including decimal/comma components)
with #, tokenize ASCII words and #, form sets of consecutive three-token shingles,
and require Jaccard similarity >= 0.80. Numeric masking conservatively connects
reports differing in amounts or dates. It can group unrelated templated reports;
this sacrifices independence within a group rather than allowing template leakage.

The matcher uses a frequency-ordered prefix index with length filtering, followed
by exact integer Jaccard comparison. It is exhaustive for this declared rule,
not an approximate nearest-neighbor search. Union-find takes transitive closure
of all episode/exact/near edges. Every connected component stays in one split.
The edge artifact permits inspection of all 3,195 near-duplicate pairs, including
1,882 across episodes. Spot inspection of the first three cross-episode edges
showed shared winter-storm descriptions with local wording changes. Largest
connected component: 225 rows; no component was broken to improve balance.

Allocate components largest first, tie-breaking by SHA-256 of sorted event IDs.
Greedily minimize incremental per-class squared deviation from 70/15/15 targets,
scaled by target count. No random seed, predictions or classifier scores are used.
All eligible rows remain, including same-label duplicates; report row-weighted
metrics with this limitation. Future group-weighted metrics can supplement them.

## Counts and validation

| Reference class | Train | Development | Reserved final |
| --- | ---: | ---: | ---: |
| Flash Flood | 2,023 | 434 | 433 |
| Tornado | 1,430 | 306 | 306 |
| Thunderstorm Wind | 5,780 | 1,239 | 1,238 |
| Hail | 1,022 | 219 | 219 |
| Winter Storm | 981 | 210 | 210 |
| Wildfire | 194 | 41 | 41 |
| Total | 11,430 | 2,449 | 2,447 |

There are 3,646 components: 2,554 train, 547 development and 545 final.
Wildfire spans 133/27/29 components respectively. Before model evaluation, the
builder required >=50 training rows, >=20 rows in each evaluation split, and >=5
components per class per split. All passed; these are data support checks, not
model deployment quality gates. Small-class estimates will remain uncertain.

Checks passed: complete unique eligible assignment; no episode overlap; no exact
normalized narrative overlap; no declared near-duplicate edge across splits;
per-class row and component support. Rebuilding reproduces all frozen bytes and
checks the builder hash. Tests compare prefix search to exhaustive pairwise
matching on synthetic sets and check numeric updates and transitive connectivity.

## Reproduce and preserve the freeze

From the repository root:

```sh
python3 -m unittest discover -s tests -v
python3 scripts/prepare_splits.py
```

The default command recomputes and verifies without changing frozen artifacts.
`--freeze` creates a new directory and refuses to overwrite an existing one.
Do not delete or regenerate v1 in response to model results. Any justified future
data revision needs an explicitly versioned dataset and documented reason.
Do not train on development or final. Use development for model decisions and
reserve final for the predeclared final evaluation of the selected artifacts.
Earlier pre-split source spot checks were available to the developer; final is
reserved from prediction-driven decisions, not a claim of unseen human access.

## Limits and next step

This is a single-year source, not a guarantee of generalization to new years,
locations, unseen fires, paraphrases or arbitrary pasted reports. Lexical grouping
cannot detect all related incidents with dissimilar descriptions, and transitive
closure may group different incidents sharing boilerplate. No external incident
identity resolution is claimed. NOAA labels remain imperfect references; this
split process is not an exhaustive semantic audit. No final predictions exist.

Next: pin and verify local model/tokenizer loading and the raw completion label
contract; establish the keyword and untuned-model development baselines. Keep all
model calls local, and keep reserved final out of exploratory evaluation.
