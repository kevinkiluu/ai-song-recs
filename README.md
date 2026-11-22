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
