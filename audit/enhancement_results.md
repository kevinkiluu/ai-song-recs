# Enhancement measurements, November 23, 2026

**v2 remains the production ranker. The Embeat channel-1 challenger still has insufficient fresh evidence.** The six plan items now have executable measurement paths. The available inputs support recording-identity checks, a provider-coverage measurement, and a sampled Bliss comparison. The historical full-pool and fresh-judgment estimates still require the private source bundle.

| Priority | Implementation | Observed result |
|---:|---|---|
| 1 | Reconstruct the fresh gate from hash-locked offers, post-rating playlists, cutoff likes, historical exclusions and the recording crosswalk. Count eligible likes and rejects separately for historical replay and post-freeze observations. | Missing original offer/retained snapshots and catalog crosswalk. The reported nine-like shortfall remains conditional on verifying one eligible like. A controlled 10-like/2-reject case cleared the count gate and returned `NO_GO_OFFLINE` after scoring. [Private inventory and commands](fresh_inputs.md) |
| 2 | Resolve connected recordings by Spotify ID, Deezer ID, ISRC and preview hash before fitting. Compare v2 and both Embeat arms on the same eligible panel, with and without the older training aliases. | Across the three liked-song folds, verified component overlaps fell from 5, 1 and 6 to zero. All three fixed panels contained distinct recordings. The original reject features and full candidate pool are missing, so the personal ranking deltas are unmeasured. |
| 3 | Retry missing ReccoBeats panels, validate independent Spotify/ISRC evidence for recovered rows and keep the derived cache separate from the baseline. | The provider returned zero panels across 11 successful batches covering all 202 valid missing Spotify IDs. One additional missing row had no valid Spotify ID. Coverage stayed at 33/130 among the newest likes and 414/520 among older likes. The independent-ISRC fallback needs the original identity crosswalk. [Coverage run and cache replay](coverage_inputs.md) |
| 4 | Distinguish no match, unavailable preview, failed transfer and failed decode. Retry an existing match by provider ID and cache only the recovered feature panel and audio digest. | The snapshot records 173 no-match rows. Two more rows have a Deezer match and lack decoded features. Both Nov 23 metadata requests received HTTP 403 from the network proxy, before transfer or decode. Zero of the 55 rows lacking both sources gained a feature. The original failure stage for the two matches remains unknown. [Coverage details](coverage_inputs.md) |
| 5 | Score one fixed panel, slice it for seeds 42–46, and compare standard P@10 and NDCG@10 against v6 per-bucket with a shared tie rule. Save the score digest, membership and top-ten changes. | On 305 distinct archived recordings the Bliss/v2 blend’s NDCG advantage was +0.175262. It ranged from +0.120152 to +0.271569 across five distinct draws of 100 of the 200 archived distractors. Fixed and adaptive Bliss each trailed v6 on NDCG in every draw. The original full catalog and Bliss vectors remain unavailable. [All five draws and the full-pool command](bliss_inputs.md) |
| 6 | Import Extended Streaming History locally, count valid exposures, plays, skips and completions, then measure a history adjustment against frozen v2 on later explicit judgments. | The Nov 23 availability check found no extended history export. The one-time Nov 26 availability check is scheduled. Real P@10, NDCG@10 and reject-control deltas are pending. [Importer and temporal evaluation](streaming_inputs.md) |

## Recording folds

The original top-level feature schema contains zero ISRC fields with values. The embedded ReccoBeats payloads contain **447** ISRCs whose Spotify link agrees with the row and whose code has valid syntax after normalization. Combining those fields with the existing IDs reduces the 649 identified liked rows from 634 to 629 connected recordings. This corrects the earlier assessment of ISRC availability. Stored preview bytes are still unavailable here.

| Liked positions held out | Overlapping recordings before grouping | Older alias rows | After grouping, overlap / duplicate test recordings | Usable training likes | Eligible held-out likes |
|---|---:|---:|---:|---:|---:|
| 0–129 | 5 | 6 | 0 / 0 | 472 | 98 |
| 130–259 | 1 | 1 | 0 / 0 | 357 | 103 |
| 260–389 | 6 | 6 | 0 / 0 | 235 | 89 |

These are liked-snapshot counts. Additional frozen provider identities can connect more rows. The fold CLI writes JSON to stdout, including source digests, identity coverage on the active train and test sets, exclusion counts and a scored-panel digest. Missing required scoring files produce exit 2 and null paired results. With complete inputs, the before fit restores just the older, feature-bearing training rows linked to an eligible held-out like. Both fits hold the candidate records and labels fixed. This isolates the effect of those training aliases. A three-fold controlled run restored one alias per fold and scored v2 and both Embeat arms in all three.

```sh
python -m audit.recording_folds
python -m audit.recording_folds --likes /private/features_v2.json \
  --rejects /private/data/negatives_features.json --catalog /private/data/pool_features.json \
  --matches /private/data/deezer_matches.json --previews /private/previews
```

The 98 eligible newest recordings are a position-based holdout from the training-era library. Future comparisons use the separate fresh gate and require untouched observations dated after the challenger freeze. [Original plan](plan.md) · [Earlier acceptance ledger](verification.md)
