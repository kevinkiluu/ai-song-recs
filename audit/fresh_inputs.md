# Reconstruct the fresh count gate

From the checkout, point the gate at a private inventory of archived sources. Supply the SHA-256 of that inventory from a separate trusted record. A successful run writes two files with owner-only permissions in a new directory at the path you choose. The example keeps listener data outside the checkout.

```sh
python audit/fresh_gate.py /private/inventory.json \
  --sha256 "$TRUSTED_INVENTORY_SHA256" --output-dir /private/fresh-run
```

`gate.json` contains the recording-level decisions, per-round exposure and judgment counts, source digests, eligible counts for the original replay and later rounds, and the remaining shortfall. `fresh.json` is the binary judgment input for `integrate_v4.py --fresh`. Supply that runner with the **same** catalog, likes, rejects, optional Deezer matches, and optional preview files. An expanded identity graph can change eligibility. A count of ten eligible likes and two rejects permits the offline ranking comparison. Use its independently checked metrics for that decision. Run `python -m unittest test_fresh_gate` for generated examples.

## The private inventory

This example shows the metadata format. Replace digest placeholders with the full original checksums. Generation records, archived snapshots or receipts must establish each assignment, collection, timestamp and completeness declaration. Rehashing a present-day playlist establishes its current contents alone. The inventory digest binds the operator's source assignments to the run. Retain the provenance independently.

```json
{
  "dataset": "original_v5",
  "mode": "historical_replay",
  "challenger_revision": "<frozen challenger commit>",
  "frozen_at": "2026-11-21T10:00:00Z",
  "library_id": "spotify:user:OWNER:collection",
  "sources": {
    "catalog": {"role":"catalog", "path":"data/pool_features.json", "sha256":"<digest>"},
    "rejects": {"role":"rejects", "path":"data/negatives_features.json", "sha256":"<digest>"},
    "likes": {"role":"likes", "path":"features_v2.json", "sha256":"<digest>"},
    "crosswalk": {"role":"crosswalk", "path":"data/reccobeats_pool.json", "sha256":"<digest>"},
    "original_likes": {"role":"original_likes", "path":"original/spotify-liked-songs.txt", "sha256":"<digest>", "at":"2026-11-21T08:00:00Z", "collection":"spotify:user:OWNER:collection", "complete":true},
    "v9_offer": {"role":"offered", "round":"v9", "path":"v9/offered.tsv", "sha256":"<digest>", "at":"2026-11-21T16:00:00Z", "collection":"<offer playlist ID>", "complete":true},
    "v9_after": {"role":"retained", "round":"v9", "path":"v9/rated.tsv", "sha256":"<digest>", "at":"2026-11-22T12:00:00Z", "collection":"<rated playlist ID>", "complete":true},
    "nov22_likes": {"role":"cutoff_likes", "path":"nov22/spotify-liked-songs.txt", "sha256":"<digest>", "at":"2026-11-22T13:00:00Z", "collection":"spotify:user:OWNER:collection", "complete":true},
    "v9_history": {"role":"prior_history", "round":"v9", "path":"v9/prior-recommendations.tsv", "sha256":"<digest>", "at":"2026-11-21T15:59:00Z", "collection":"prior-recommendations", "complete":true}
  },
  "rounds": [{
    "name":"v9", "cohort":"historical_replay", "untouched":true,
    "playlist_id":"<offer playlist ID>", "rated_playlist_id":"<rated playlist ID>",
    "cutoff":"2026-11-22T13:00:00Z", "offered":"v9_offer", "retained":"v9_after",
    "cutoff_likes":"nov22_likes", "prior_history":"v9_history"
  }]
}
```

All times in the example are illustrative. Use the actual collection IDs and times from the archived events. Add every round in the prespecified cohort. Resolve a different rated-copy playlist ID only from an authenticated role assignment. For a single playlist omit `rated_playlist_id`. Each dated history snapshot represents recommendations that preceded that round's offers. A later undated 90-entry file containing those same offers cannot establish prior exposure. The original 650-row liked TSV must agree with every frozen feature position, including the known placeholder. A shared cutoff liked export may serve multiple historical rounds when its observation time equals each declared cutoff.

The original 590-row catalog and 13 training rejects must match the separately retained **full** SHA-256 digests in the locked inventory. Original mode also checks the 590/13/650 row counts and the full hash of the tracked liked-feature blob. Fixture runs are explicitly identified. Fragments quoted in the old reports cannot authenticate private files.

## Input shapes and recording identity

- Feature sources use the same JSON as the v4 runner. The likes object contains `tracks`. Catalog and rejects are lists of feature records. Add optional source `matches` with role `matches` for the original `data/deezer_matches.json`. For archived preview audio, include each available file as a source named `preview/123` with role `preview`, path ending in `123.mp3`, and its full SHA-256. Substitute the actual Deezer ID. When comparing to a v4 run with preview enrichment, include every audio file used by that run.
- Track exports accept `title<TAB>artist<TAB>spotify:track:ID`, the pipeline's JSON list of track objects, and Spotify Web API playlist or saved-track pages. Preserve raw bytes. A complete Spotify page needs an exhausted continuation and the full advertised count. For a multi-page collection use the complete archived export with its original pagination provenance. A recommendation output can serve as the offer list only if its creation or delivery receipt authenticates that exact playlist and round.
- The frozen ReccoBeats pool cache maps an ISRC to a record with `href` and `isrc`. Its Spotify track URL, ISRC key, returned ISRC, catalog identity, title including version, and first artist must agree. Exact Spotify identities on frozen rows also work. The v4 component graph deduplicates aliases and excludes historical like or reject identities. Prior exposure also checks the original v2 normalized title and first artist, so a different Spotify ID can still be excluded by the historical policy. Raw text alone cannot establish a catalog join.
- For later judgments set `cohort` to `post_freeze`. Pin the challenger revision and freeze time, and identify the **pre-offer** liked library used at generation in the round's `prior_likes` field. Its source uses role `prior_likes`, the round name and library collection, or references `original_likes` if that exact export was used. Post-freeze offers must occur strictly after the freeze. An absent or incomplete prior library leaves unknown freshness for unobserved members. Set `untouched` true only for judgments withheld from tuning or selection. Historical replay is for the originally held-out historical cohort. The top-level `mode` selects the cohort for the gate and v4 judgment file. It defaults to `post_freeze`. Set it explicitly to `historical_replay` to reproduce an earlier measurement. The report shows counts for both cohorts. Each recording belongs to its first offer's cohort.

## Counts and missing evidence

Membership in cutoff Liked Songs makes an offered recording **kept**. A retained but unsaved offer is **left**. An offer absent from both complete sources is **removed**. A partial snapshot can support observed presence. A missing ID, uncertain alias, or incomplete snapshot prevents an unsupported inference from absence. Raw offer rows and unresolved rows are counted separately. Each round counts distinct identified recordings plus unresolved positions. When an offer list is incomplete its denominator covers only the recovered offers. Retained additions outside that set have no offer in the denominator.

The recording funnel counts catalog joins and previously known identities before eligibility exclusions. Those counts can overlap each other and the exclusion reasons. Mutually inconsistent tiers, disputed identities, missing catalog features, OK judgments, previous labels or prior offers cannot become eligible binary labels. The first observation determines freshness for repeated recordings. The private audit identifies the round for each supported tier. Missing required files, ambiguous provenance, and failed locks stop the command with exit 2. A valid run with insufficient counts returns the measured shortfall and exit 0.
