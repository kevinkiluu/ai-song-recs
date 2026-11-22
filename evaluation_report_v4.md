# Pipeline v4 audit and evaluation

**Status: insufficient evidence for a ranking change.** The evaluation code is
ready for the frozen inputs. The checkout has the v2 liked-feature snapshot
and the v3 score audit, but lacks `data/negatives_features.json` and
`data/pool_features.json`. The v4 runner reports both missing paths and exits
with code 2. The v2 ranker remains in service.

## What the PR #3 replay establishes

Base commit `13604951419ad199cf8639f2e811f5b99fdad3d9`. The v3 integration
checks replayed the published metrics and verdict from all 306 scored rows.
The legacy unit script ran 84 passing checks and one missing-data check
(`test_t01`). Rebuilding the full v3 score pipeline requires the frozen
caches and original model inputs.

The same 306 stored rows yield the figures below. The rightmost column changes
only the NDCG denominator. These are historical rankings with known v3
limitations.

| Published ranking | Known-like P@10 | Legacy NDCG@10 | Standard NDCG@10 |
| --- | ---: | ---: | ---: |
| v2 | 0.50 | 0.726 | 0.471036 |
| v6 per-bucket | 0.40 | 0.654 | 0.368658 |
| Embeat full | 0.60 | 0.829 | 0.603262 |
| Embeat with channel 1 off | 0.60 | 0.948 | 0.689829 |

The liked-feature file contains 650 positions, 475 records with preview
features, and zero verified add dates. On this file alone the v4 manifest
retains 506 training and 98 eligible test representatives. It excludes 30
rows with missing shared features, one with unresolved identity, nine other
aliases, and six cross-boundary aliases on the training side. These counts
can change when the frozen catalog, rejects, ISRCs and cached preview hashes
are added to the identity graph.

## Decision-changing findings

| v3 location | Reproduction and effect | v4 behavior |
| --- | --- | --- |
| `integrate_v3.py:111`, `integrate_v3.py:420` | Five feature-covered holdout rows share a training Deezer ID. Sampled evaluation removes them. The 701-row path selects all 99 feature-covered holdout likes again. | Build one recording-level eligibility manifest and refit using disjoint recording groups. Full and sampled rankings share it and reuse scores from the same fold. |
| `model_v2.py:155`, `integrate_v3.py:427` | Training and negative-control evaluation load the same reject file. A low reject rank can measure memorization. | Rotate disjoint historical reject groups. Fresh labels attach to frozen catalog features and never enter their own fit. |
| `integrate_v3.py:47`, `embeat_port.py:869` | The artist bucket map and Embeat's eligible seed artists use all likes. Changing a held-out taste bucket can change scores. | Build the bucket map, model and seed from training likes. Use train-derived artist buckets for each comparison. |
| `evaluate_v3.py:272`, `evaluate_v3.py:282` | One and two consecutive top hits can both receive legacy NDCG 1.0. The item bootstrap can give one recording multiple top-10 slots. | Use standard NDCG and fixed, without-replacement distractor draws. Position windows and pool draws measure sensitivity for one user. |
| `evaluate_v3.py:616` | With passing offline inputs, 10/10 returns GO even though the documented first checkpoint is 42. | Compute checkpoints from unique, rated challenger recordings. The v2 controls have their own arm. The first 42 challenger ratings require 32 keeps. The threshold at 50 is 37. |

The historical v3 entry point and artifacts are kept for replay. Use
`integrate_v4.py` for new evaluations. v6's published scorer constants are
frozen. Its artist bucket lookup uses the current training fold. Embeat's
popularity proxy and candidate provenance retain the limits documented in
v3.

## Run v4

Use the existing audio environment or Python with NumPy, SciPy and
scikit-learn. From the checkout:

```bash
python -m unittest -v test_evaluate_v4 test_integrate_v4 test_check_v4
python integrate_v4.py --output /tmp/evaluation-v4.json
python check_v4.py /tmp/evaluation-v4.json
```

Place the two original caches under `data/` or point `--rejects` and
`--catalog` to their frozen copies. The runner expects lists of the repo's
existing feature records. `--liked-export` verifies the frozen source
order against the feature snapshot. When available, pass
`--matches data/deezer_matches.json --previews previews` to use the cached
ISRC mappings and hash original preview bytes. Cache-only runs use the
provider identities available in their records. Raw audio extraction and
catalog discovery are outside this run.

The default diagnostic uses positions 0–129 and reject fold 0. Rerun with
`--holdout-start 130 --cutoff 260 --reject-fold 1` and
`--holdout-start 260 --cutoff 390 --reject-fold 2` and distinct local
output paths for position sensitivity. Those folds omit newer positions
from training. The five fixed distractor draws retain all judged rows and
reuse the full-pool scores, so they measure the effect of catalog
composition.

For confirmatory frozen-pool ranking, supply `--fresh path/to/judgments.json`.
Each judgment supplies provider IDs for a frozen catalog recording,
`"label": "like"` or `"reject"`, and a `"label_source"` that identifies
the record of the untouched judgment. The runner joins those labels to the
frozen features. Historical judgments or entries without an eligible
catalog match cannot qualify. Unlabeled catalog tracks remain distinct
from confirmed rejects. Known-like P@10 measures labeled recovery.
Catalog retrieval quality and future preferences remain unverified.

The predeclared screening rule requires at least ten eligible fresh likes
and two fresh rejects. The ablation must improve standard NDCG without
losing P@10 against intact Embeat. Against both v2 and historical v6 it
must improve NDCG and either gain at least 0.10 P@10 or tie P@10 with a
0.02 MAP@10 gain. It must also beat chance with P@10 of at least 0.50,
place zero rejects in the top 10, and rank rejects below the midpoint on
average. A reversal versus v2 or v6 on fixed distractor draws blocks
eligibility. This is an offline screening rule. Future improvement
requires the live checkpoint.

If the offline result is eligible, `--blind-output /path/to/deck.json`
writes a listener copy, with up to five tracks per arm. Supply
`--liked-export`, `--history`, and `--verified-spotify` (a JSON map from
manifest recording ID to independently verified Spotify track ID). If
the current likes differ from the source snapshot, supply
`--current-likes` for shortlist exclusions. Keep the audit private
until ratings are locked. For each completed round, append an object with
`trial_id`, `reveal` (copied from the audit's `reveal_after_ratings`),
and `ratings` (token to boolean) to a local JSON list. Pass that list via
`--live` to replay the checkpoint or prepare another round. Duplicate
recordings and mismatched rounds are rejected. Two consecutive blocks
of ten challenger ratings at seven keeps or fewer trigger a stop. The
runner emits files locally and has no Spotify write path.

A published v4 evaluation report with actual metrics requires the two
frozen feature caches. A promotion result also requires untouched
catalog judgments and the prospective blind ratings. The report and
per-item audit from those inputs should stay local.
