# Streaming history and later judged recordings

The local availability check on **November 23, 2026 UTC** returned `awaiting_nov_26_check`. The repository data directory, local audio-pipeline data directory, personal and shared downloads, and workspace library files contained no extended-history export. The real comparison is pending. The checked-in liked snapshot contains no verified judgment times.

From the repository root, the following command checks availability on the day it runs and prints JSON. It reads local files and returns exit 2 for missing or unusable evidence. On or after November 26, an absent export yields `export_unavailable`. For a date-stamped run on November 26 use `--as-of 2026-11-26`. The CLI rejects future check dates. A historical cutoff restricts the observed events and evaluation dates, and does not reconstruct the files that were actually on disk at that time.

```bash
python -m audit.streaming_history
```

To import an export in another location, pass `--history /private/my_spotify_data.zip`. The option also accepts a single extended-history JSON file or a directory and can be repeated. Directory and ZIP discovery recognizes both `Streaming_History_Audio_*.json` and `endsong_*.json`. An Account data export alone produces an error. ZIP members are read in memory. Per-event metadata, row identifiers and fitted weights stay local. Use exports from one listener. Parts identifying different listeners produce an error.

## What the import measures

Raw ID, timestamp and duration coverage each use **all raw rows** as their denominator, including overlaps across files. A valid track ID is a 22-character base62 Spotify track URI on a music event. A timestamp needs an explicit offset and is normalized to UTC. Duration here means nonnegative integer milliseconds listened. Exposures count distinct valid event ends on or before the UTC cutoff date, including brief listens. Episodes, local files and unidentified items contribute no exposures. The summary reports both the unique-track and the event join fractions against the reference catalog. The default reference is the 649 valid Spotify IDs in the checked-in liked-feature cache. Supply `--catalog` for another versioned array of records with `spotify_id` and optional `duration_ms`, or an object containing that array in `tracks`.

Events with the same track and UTC end time in different parts count once when duration, skip flag and playback context agree. Ambiguous same-part collisions or conflicting cross-part records block the import. An invalid event remains visible in the raw coverage counts. If its ID is identifiable but its timing or playback is unresolved and it could affect a judged recording’s features, the comparison stops. Unknown IDs supply no joins. Coverage and model inputs describe the portion of playback observable in the supplied export.

| Quantity | Rule |
|---|---|
| Qualifying play | At least 30,000 ms, including events flagged as skipped by Spotify. |
| Repeated qualifying play | Each qualifying play after the first for the same Spotify ID. A separate count gives the number of IDs with two or more qualifying plays. |
| Short skip | Under 30,000 ms. The export’s optional `skipped` flag gets its own count. |
| Completion | At least 120,000 ms, or at least 80% of the track duration if a verified duration is supplied. A very short track can meet the short-skip and completion definitions on the same exposure. |
| Repeated short skips | At least three short skips and zero completions at scoring time. This is a behavioral feature. All judgments, including rejects, come from the explicit outcome source. |

The play and behavioral thresholds come from the repository’s [extended-history specification](../research/implicit-spec.md) and [history skip/completion definition](../research/troi-spec.md).

## Running the paired comparison

```bash
python -m audit.streaming_history --history /private/spotify-export.zip \
  --panel /private/judged-blocks.json --model-dir /private/frozen-v2
```

The model directory holds the original `model_v2.npz` and `model_v2.json`. The audit checks each byte stream against the panel’s SHA-256 digest, then passes those same bytes to the production `V2Model` loader. Obtain the digests, freeze time and training inventory from the original model records. Digest agreement alone cannot establish when a model was frozen.

The panel is a local JSON object. Build it by joining archived candidate features to independent, timestamped listener judgments. Preserve the frozen cohort for each block and supply every judged recording in it. Document the rating source and establish the shared recording identities before using the panel. Current library membership, inferred skip proxies and unjudged candidates are ineligible.

| Field | Required content |
|---|---|
| `frozen_at` | Timestamp when the original v2 model and training inventory were fixed, preceding the first block. |
| `model_sha256` | Object keyed by `model_v2.npz` and `model_v2.json`, each containing its SHA-256 hex digest. |
| `training_recordings` | One identity entry per original v2 training row, including negatives. A string gives a canonical Spotify ID. An object gives `recording_id`, `spotify_id`, any verified `spotify_aliases` and optional `deezer_id`, `isrc`, `preview_sha256`. Use consistent recording keys across the entire panel. The production config’s positive and negative training counts provide a row-count check. |
| `blocks` | Ordered array of nonoverlapping blocks. Each has `role` (`train` or `test`), `start`, `end`, `judgment_source` and `rows`. Supply offset-aware timestamps and put every training block before the first test block. Each test block requires at least ten judged recordings. The full test panel must contain a like and a known reject. |
| Each row | A unique `recording_id`, valid `spotify_id`, explicit `label` (`like`, `ok`, `reject`), `judged_at` at or after its block start and strictly before its end, and the `reccobeats` and/or `librosa` features available at scoring time in the v2 feature-cache schema. Optional identity fields as above and a verified positive integer `duration_ms` are supported. One row represents one recording even if the history uses several Spotify aliases. |
| `independence_basis` | Optional text identifying the predeclared independent test sampling units and the evidence for their independence, conditional on the fixed training fit. Omit this when sessions share relevant time effects or the sampling design is unknown. |

Both arms score every judged recording in each test block, with ties broken by the shared recording key. The fixed arm uses the original v2 score. The challenger adds a ridge-logistic adjustment with that score as a fixed offset. It fits three weights on the earlier training blocks using log(1 + qualifying plays), log(1 + completions) and the repeated-short-skips indicator. Its objective is the sum of binary logistic losses plus half the squared weight norm, without an added intercept. Each training and test row uses only events ending strictly before its own block starts. The fit freezes before the first test and stays fixed across later blocks.

The training target is an explicitly judged **like** versus an explicitly judged **OK or reject**. An OK remains eligible to rank and contributes zero to binary like relevance. Only an explicit reject counts in the top-ten reject control. The report gives each block’s judged and exposed counts, both arms’ P@10, standard NDCG@10 and known rejects in the top ten, and their paired differences (history minus v2). Negative differences in reject counts favor the challenger. A block with no likes contributes zero standard NDCG under the repository convention. Every prespecified block enters the mean, including a block with no likes or no rejects. These results apply to the supplied judged cohort.

The primary summary gives each test block equal weight. Eight or more test blocks and a documented independent design enable simultaneous 95% two-sided Hoeffding intervals for the three mean paired differences. P@10 and NDCG@10 differences lie in [−1, 1], and reject-count differences lie in [−10, 10]. The Bonferroni-adjusted intervals can span most of the possible range. Chronology and disjoint recording IDs alone do not establish independence. With less evidence the report includes the observed differences and leaves uncertainty pending.

The synthetic CLI test executes the importer, model loader, before-boundary fit and both ranking arms. On its 12-row test block P@10 changes from 0 to 0.2 and NDCG@10 from 0 to 1. Each arm places one known reject in its top ten. Those numbers exercise the code and do not estimate listener performance.
