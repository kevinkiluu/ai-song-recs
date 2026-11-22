# ai-song-recs

Spotify recommendation pipeline for Kevin's Liked Songs. Two generations
live here side by side.

## v1 (Sep 2026, still working)

- `pipeline.py` — Deezer 30s preview fetch + librosa analysis of the
  first 100 recent liked tracks; writes `analysis.json`.
- `recommend.py` — per-bucket (E/R/H) median-profile scoring.
- `retry_candidates.py` — retry helper for failed candidates.

v1 method: score = |bpm-med|/IQR + |energy-mean|/0.10 +
|brightness-mean|/500 - 0.5 if minor, per genre bucket.

## v2 (Nov 2026, current)

Deeper, not bigger. Same goal: better analysis of Liked Songs and
better recommendations. See `evaluation_report_v2.md` for the numbers
and the honest limits.

- `features_v2.py` — ReccoBeats panel + Deezer match + deep librosa
  features for all 650 liked tracks; writes `features_v2.json`.
- `scoring.py` — the single shared feature-vector definition, the v2
  model scorer, and the v6 baseline scorer (one definition of
  `feat_vector`, imported by everything else).
- `model_v2.py` — PCA(15) -> Gaussian-mixture taste clusters (k by
  BIC) + logistic regression on likes vs 13 confirmed negatives;
  writes `model_v2.json` + `model_v2.npz`.
- `negatives_features.py` — feature extraction for the 13 confirmed
  negatives (9 dropped v5 tracks + 4 v6 H-bucket thumbs-down).
- `recommend_v2.py` — candidate pool from Deezer related artists of
  the 30 most-liked artists (`build-pool`), MMR selection + Spotify-ID
  verification (`recommend`).
- `pool_reccobeats.py` — ISRC + ReccoBeats fill for pool tracks
  (Deezer top-tracks responses carry no ISRC).
- `evaluate.py` — evaluation harness (run BEFORE recommendations):
  precision@10 and NDCG@10, new method vs v6, on a held-out set of
  the most recent 20% of likes + negatives + 200 distractors.
- `gtzan_check.py` — GTZAN genre sanity check for the feature
  extractor.
- `test_pipeline_v2.py` — acceptance checks (37 checks, all green).
  Run it after any change: `audio-env/bin/python test_pipeline_v2.py`.

### Results (2026-11-22)

- NDCG@10: v2 0.726 vs v6 0.666 (per-bucket) / 0.664 (global).
- P@10: v2 0.50 vs v6 0.50 / 0.60 (lower bound; distractors unlabeled).
- Mean rank of the 13 confirmed negatives: v2 292.9 of 311 (bottom
  6%) vs v6 173-178 (mid-pack). v2's biggest win is rejecting
  known-bad tracks.
- GTZAN genre accuracy: 0.600 (10 genres, chance 0.10).
- v6 keep rate: 6/10 (all four drops were H-bucket picks).

### Hard limits

- All audio is 30s Deezer previews. Kevin owns no full-length audio
  files; full-track structure and Demucs stems are not possible.
- No streaming-history export, so no skip/play labels; the negative
  set is 13 confirmed tracks.
- ReccoBeats covers 69% of liked tracks (25% of the newest 130).
- 27% of liked tracks have no Deezer preview.

### Run order

```text
python features_v2.py        # ~5 min cached, longer cold
python negatives_features.py # ~1 min
python model_v2.py           # ~1 min
python gtzan_check.py 10     # ~9 min
python recommend_v2.py build-pool   # ~25 min (Deezer API)
python pool_reccobeats.py    # ~8 min
python evaluate.py           # ~1 min
python recommend_v2.py recommend  # ~1 min
python test_pipeline_v2.py   # acceptance checks, must be all green
```

Do not modify `~/workspace/spotify-liked-songs.txt` (read-only) and do
not append to `~/workspace/spotify-recommended-tracks.txt` until Kevin
has rated the new recommendations. Dedup is by Spotify track ID.

## v3 (Nov 2026, integration — NO-GO offline)

Harness + three experimental feature ports, integrated on a
decontaminated 306-item pool (94 held-out + 12 negatives + 200
distractors; 5 duplicate-audio held-out likes removed). Verdict: **no
candidate passes Legs 1–3 of the pre-registered 4-leg GO rule, so v3
ships the harness and the ports as experimental modules with v2
unchanged as the ranker. No new recommendation list was built.**
See `evaluation_report_v3.md` and `evaluation_v3.json` for the numbers.

- `evaluate_v3.py` — eval harness v3 (P-01..P-11 ports + 4-leg verdict;
  `test_evaluate_v3.py`, 97 checks). Run the verdict with
  `audio-env/bin/python integrate_v3.py`.
- `troi_patch.py` — Troi-style patch (audio neighbours + same-artist,
  zipper). 306-pool: P@10 0.80, NDCG 0.820; fails Leg 1 (ΔNDCG CI
  includes 0) and Leg 3 (mean neg rank 146.7 < 153.5).
- `embeat_port.py` — Embeat architecture port (4 channels, medoid
  seed). 306-pool: P@10 0.60, NDCG 0.829; minus-channel-1 variant
  NDCG 0.948, P@10 flat. Channel 4 KEPT (removal costs 0.10 P@10).
- `bliss_features.py` — 23-dim bliss vectors + distance scorer.
  Alone below random (P@10 0.20); equal-weight bliss+v2 blend P@10
  0.40, NDCG 0.956. Sampled-pool and full-pool (701) deltas disagree
  in sign for bliss alone — unresolved, forces NO-GO on its own.
- `integrate_v3.py` — the integration (decontaminate, re-score every
  candidate on the 306 pool, 4-leg verdicts via `verdict_4leg`).
  `checks_v3_integration.py` re-derives every number from the
  per-item audit file `v3_scores_306.json`.

Tests (all re-run fresh 2026-11-22): `test_pipeline_v2.py` 37 green;
`test_evaluate_v3.py` 97 green; `test_troi_patch.py` 20 passed +
1 xpassed; `test_embeat_port.py` 20 passed + T-16 FAIL (honest eval
failure, not fixed by weakening the test); `test_bliss_features.py`
20 passed + Z2 FAIL + 1 skipped.

## v4 evaluation

`integrate_v4.py` runs the recording-disjoint comparison of refit v2,
historical v6, and Embeat with channel 1 on and off. `check_v4.py`
independently verifies the local audit. See [the v4 evaluation
report](evaluation_report_v4.md) for the historical replay, frozen input
requirements, and blinded rating protocol.
