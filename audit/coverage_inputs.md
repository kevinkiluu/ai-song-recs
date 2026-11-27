# ReccoBeats and preview recovery

On November 23, 2026, the read-only retry recovered **zero verified ReccoBeats panels and zero previews** for the November 22 liked snapshot. The newest 130 positions retained 33 ReccoBeats panels and 99 usable previews. No recording among the original 55 rows lacking both sources gained either source.

| Fixed position cohort | Rows | ReccoBeats, before → after | Preview, before → after | Neither, before → after |
|---|---:|---:|---:|---:|
| Newest, 0–129 | 130 | 33 → 33 (25.4%) | 99 → 99 | 26 → 26 |
| Older, 130–649 | 520 | 414 → 414 (79.6%) | 376 → 376 | 29 → 29 |
| All | 650 | 447 → 447 (68.8%) | 475 → 475 | 55 → 55 |

Of 203 rows missing ReccoBeats, 202 had a valid Spotify ID. Eleven provider requests covered all 96 valid IDs in the newest group and all 106 in the older group. All requests completed, returning zero candidate panels. A separate positive control on a previously covered recording returned panels for its Spotify ID and ISRC. The independent Spotify track-to-ISRC export was unavailable, so the alternate-edition ISRC search for the missing tracks is still open. The frozen list order defines recency. The snapshot lacks verified save times. [Frozen feature rows](../features_v2.json)

Of 175 rows missing preview features, the frozen export recorded “no match” for 173. This is the export’s historical classification, which can include an absent search result. Two rows had previously matched Deezer IDs without features. Their historical failure stages are unknown because the match, transfer and decoder logs are missing. Both rows belong to the older cohort and already had ReccoBeats. On November 23 the HTTPS proxy rejected both metadata requests to Deezer with HTTP 403 before transfer or decoding. This run measured zero successful redecodes. Recovery under working provider access remains unresolved.

There were 98 preview-eligible held-out liked representatives before and after, and zero newly evaluable liked recordings. Fixed mixed-pool P@10 and standard NDCG@10 deltas are **unavailable** because the frozen catalog and historical reject feature rows are missing. No rank improvement follows from these coverage measurements. [Recording split and evaluation](../evaluate_v4.py)

## Reproduce the aggregate

Run from the repository root. The first command reads the coverage snapshot offline. The second reads providers and writes an owner-only, create-once cache to an existing private directory. Keep the emitted cache digest with the aggregate from that run.

```bash
python -m audit.coverage
python -m audit.coverage --recover /private/coverage-new-v1.json --rb-limit 202 --preview-limit 2
python -m audit.coverage --cache /private/coverage-new-v1.json --cache-sha256 ORIGINAL_CACHE_SHA256
```

The November 23 cache was observed at 14:10:04 UTC. Its SHA-256 is `e821a1a059dd669aab8ece36056fa310f7b37f8229475de65fd3326f3dd097de`. It binds to the November 22 04:08:03 UTC snapshot with SHA-256 `d8db0af163807d9c40fa8fcc2903ac70db059cd1492efa639e92e7d7a4488ed1`. The cache and optional private inputs stay local. The CLI prints aggregate counts, metrics and input digests. Preview bytes exist only in a temporary directory during decoding. [Recovery and replay](coverage.py)

| Optional input | Contents and use |
|---|---|
| `--spotify-tracks FILE` | An independent JSON array of Spotify track objects with `id` and `external_ids.isrc`. Use a trustworthy source export. Conflicting valid codes for one ID stop the run. ReccoBeats can answer via the exact Spotify ID or an alternate edition bearing the independently confirmed ISRC. A direct contradiction, malformed feature panel or conflicting feature values among eligible editions disqualifies the candidate. |
| `--matches FILE`, `--deep FILE` | Original `deezer_matches.json` keyed by Spotify ID and `librosa_deep.json` keyed by Deezer ID. A recorded download error or failed analysis supports the corresponding historical stage. A missing preview file alone leaves the cause unknown. |
| `--rejects FILE`, `--catalog FILE` | The original arrays of historical reject and catalog feature records. The audit fits the existing v2 implementation once on the original training partition and reuses that fitted model. The shared pool needs at least ten positions, a held-out like, a held-out reject and an eligible catalog distractor. |

The model comparison uses the repository’s recording split for positions 0–129 and reject fold 0. Both arms share the same identity evidence, training features, held-out recording representatives and score tie order. Source-specific deltas change one feature panel at a time. Newly eligible recordings are counted outside the paired ranking. The coverage tables count snapshot rows. The recovered-recording count deduplicates accepted rows by verified ISRC. [Regression tests](../test_coverage_plan.py)
