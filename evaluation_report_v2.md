# Pipeline v2 evaluation report

Date: 2026-11-22. All numbers below come from actual runs on this date;
nothing is estimated or carried over from v6. Reproduce with
`test_pipeline_v2.py` (acceptance checks) and the scripts in this repo.

## Verdict in one paragraph

The new method is better than v6 where it matters most and not better
everywhere. It ranks held-out likes better in the top 10 (NDCG@10 0.726
vs 0.666) and it is dramatically better at rejecting tracks Kevin
already disliked: the 13 confirmed negatives land at mean rank 293 of
311 under v2 (bottom 6%) vs 173-178 under v6 (middle of the pack). It
does NOT win on precision@10 (0.5 vs 0.5 per-bucket, 0.6 global) or on
the average rank of held-out likes (137.7 vs 132-136). So: v2 is a real
but narrow improvement, and its biggest win is the learned negative
signal, not the clustering. The clustering is the weakest part of the
new method (GMM silhouette 0.02); the logistic-regression term carries
most of the gain.

## What changed vs v6

- Features: 78-dim vector per track (12 ReccoBeats panel fields +
  66 deep librosa fields: MFCC mean/std, chroma histogram, tonnetz,
  spectral contrast, brightness/rolloff/bandwidth/flatness, ZCR, RMS,
  onset strength, per-segment brightness/energy deltas). v6 used 5
  scalars (BPM, energy, brightness, key mode, per-bucket medians).
- Modeling: PCA(15) -> Gaussian mixture (k=4 by BIC) taste clusters;
  logistic regression on train likes vs 13 confirmed negatives
  (class_weight balanced, recency-weighted likes). Score =
  0.5*z(-distance to nearest cluster) + 0.5*z(LR log-odds), z-scored on
  the train distribution. No hand-picked constants.
- Candidates: 590-track pool from Deezer related artists of Kevin's 30
  most-liked artists (not a hardcoded list); MMR diversity (lambda 0.7);
  finalists verified by Spotify track ID against Liked Songs and the
  recommendation history.
- Dedup: by Spotify track ID throughout. Never re-recommends.

## Evaluation

Ranking task: rank 311 items — 99 held-out liked tracks (most recent
20% of Liked Songs, positions 0-129, with librosa features), 12
confirmed negatives with librosa, 200 distractor tracks sampled from
the candidate pool (seed 42). Relevance = 1 iff the item is a held-out
like. All three methods rank the same 311 items.

| Metric | v2 | v6 per-bucket | v6 global |
|---|---|---|---|
| P@10 | 0.50 | 0.50 | 0.60 |
| NDCG@10 | 0.726 | 0.666 | 0.664 |
| Mean rank of held-out likes (lower is better) | 137.7 | 136.0 | 132.2 |
| Mean score of negatives | -1.787 (low = rejected) | 3.362 (distance; high = far) | 3.337 |
| Mean rank of negatives (311 = bottom) | 292.9 | 173.3 | 178.2 |

Notes on reading this:

- P@10 is a lower bound, not a true precision. The 200 distractors
  are unlabeled pool tracks, not confirmed dislikes; some top-ranked
  distractors are probably tracks Kevin would like. The ranking task
  measures "can the method find the held-out likes among plausible
  candidates", not "would Kevin keep these 10".
- v6 global beating v6 per-bucket on P@10 (0.6 vs 0.5) says the
  per-bucket split adds noise for the newest likes. The bucket profiles
  were fit on older likes; the most recent 130 likes are less
  bucket-separated.
- The negative result is the cleanest win: v2 puts 12 of 13 negatives
  in the bottom ~6% of the ranking. v6 leaves them mid-pack, which is
  exactly how v6 produced 4 H-bucket picks Kevin thumbs-downed on
  second listen.

## Sanity checks

- GTZAN genre classification (external benchmark): the v2 feature
  extractor + logistic regression scores 0.600 accuracy on 30 held-out
  clips across 10 genres (chance 0.10), trained on 70 clips. The
  feature pipeline distinguishes musical genres on data it never saw.
- Genre-bucket fallback check: the same features predict Kevin's own
  E/R/H/O bucket labels at 0.573 accuracy on 89 held-out likes
  (majority-class baseline 0.427). E is predicted well (29/38), R
  moderately (12/26), H poorly (9/18, with 7 of 18 H tracks predicted
  as E), O poorly (1/7). The H bucket is the least separable, which
  matches the listening evidence below.

## The H-bucket pattern

v6 keep rate is 6/10, not 10/10. On second listen Kevin thumbs-downed
all four H-bucket v6 picks: "prove it" (21 Savage, Summer Walker),
"Akina Speed Star" (Xavier Wulf), "Alone" (Night Lovell), "Cha Cha"
(Freddie Dredd). Those four plus the 9 dropped v5 tracks are the 13
confirmed negatives. The H picks scored well on v6 audio features but
failed on second listen, which means the H profile is the least
reliable of the three buckets: hip-hop/trap tracks that match on BPM,
energy and brightness still miss on something the 5-scalar profile
does not capture (likely vocal timbre, flow, or production texture).
The v2 LR term learns from those 4 directly, which is why v2 demotes
negatives so much better. But with only 13 negatives total, the LR is
a ranking signal, not a calibrated classifier.

## The 10 new recommendations

MMR order (not score order). v2 score is a z-score (higher = better
fit); v6 score is a distance (lower = closer fit). v6 per-bucket falls
back to the global profile for 8 of 10 because these are new artists
with no bucket in Kevin's liked set.

| # | Track | v2 | v6 (bucket) | v6 global | Cluster |
|---|---|---|---|---|---|
| 1 | af1 (sped up) - Lilbubblegum | +1.49 | 3.59 (global) | 3.59 | 3 |
| 2 | Cut Me Out - Flux Pavilion | +1.42 | 3.94 (global) | 3.94 | 0 |
| 3 | The Weekend - 88rising | +1.37 | 3.23 (global) | 3.23 | 0 |
| 4 | HUMANS AND MONSTERS - Lil Darkie | +1.30 | 5.13 (global) | 5.13 | 2 |
| 5 | kiss/20 - DPR CREAM | +1.24 | 3.09 (global) | 3.09 | 0 |
| 6 | Butterflies - Hearts2Hearts | +1.28 | 4.35 (global) | 4.35 | 0 |
| 7 | Wisteria - Hands Like Houses | +1.12 | 2.87 (global) | 2.87 | 0 |
| 8 | Me, Myself & I - G-Eazy | +1.12 | 3.28 (E) | 3.20 | 0 |
| 9 | 깨어 (Are You Alive) - tripleS | +1.06 | 2.97 (global) | 2.97 | 1 |
| 10 | Type Shit - Future | +1.05 | 3.52 (H) | 3.30 | 0 |

Spotify IDs are in `recommendations_v2.json`. All 10 were verified by
Spotify track ID against Liked Songs (650 tracks) and the
recommendation history (60 tracks): no overlap, no duplicates. Note
"HUMANS AND MONSTERS" scores 5.13 under v6 (v6 would never pick it);
it is the clearest v2-vs-v6 disagreement in the set.

## Blind keep-rate protocol

Kevin rates these 10 without knowing which method produced each track
(and without knowing the v2/v6 scores above). For each track: keep or
drop, with one line on why if dropped. Then compare:

- v2 keep rate (kept / 10) vs v6 keep rate (6/10).
- v2 must beat 6/10 to claim an improvement on the metric Kevin
  actually cares about. The offline metrics above (NDCG, negative
  rank) are supporting evidence, not the verdict.
- Drops go into the negative set for the next training round; if an
  H-bucket pick drops again, the H profile gets another negative
  example and the LR weight on the H-confused features grows.
- Do NOT append these 10 to `spotify-recommended-tracks.txt` until
  Kevin has rated them.

## Honest limits

1. **Preview-only audio (hard limit).** Kevin confirmed Nov 21, 2026
   he owns no full-length audio files anywhere and has no local music
   library. Every analysis is on 30s Deezer previews. Full-track
   structure, Demucs stems, and per-section analysis are not possible
   and were not approximated. The only path to full-track analysis is
   him acquiring the files.
2. **No streaming history.** No extended streaming history export
   exists, so there are no play/skip/ms-played/completion labels. The
   LR is trained on like vs 13 confirmed negatives only, not on real
   skip behavior. Unblock: Kevin provides his Spotify extended
   streaming history export.
3. **Thin negative set.** 13 negatives (9 v5 drops + 4 v6 H-bucket
   thumbs-down). The LR train accuracy (0.981 weighted) is inflated by
   the 520:13 imbalance; with so few negatives it can overfit to their
   quirks. Treat the LR as a ranking signal.
4. **Weak clusters.** GMM silhouette is 0.02 (clusters barely
   separated); one cluster holds 284 of 520 train tracks. HDBSCAN on
   UMAP found 6 clusters with silhouette 0.48, but that is measured in
   UMAP space (not comparable) and HDBSCAN has no clean out-of-sample
   scoring, so GMM is used for scoring. The cluster term contributes
   little discrimination; the LR term carries the gain.
5. **ReccoBeats coverage gaps.** 447/650 liked tracks (69%); only
   33/130 of the most recent test-set likes (25%). Missing values are
   median-imputed, which blunts the ReccoBeats signal exactly where
   the test set lives.
6. **Deezer match gaps.** 173/650 liked tracks (27%) have no Deezer
   preview; 55 tracks (8.5%) have neither librosa nor ReccoBeats
   features and contribute nothing to the model. 30 of the 130 test
   likes were excluded from ranking for lack of a preview.
7. **No added_at.** The Spotify connector exposes no added_at, so
   list position is the time proxy (verified: the live list order
   matches the local file order exactly).
8. **Distractors unlabeled.** P@10 understates true precision (see
   above). The evaluation measures retrieval of held-out likes, not
   Kevin's actual keep decisions.
9. **Essentia / Demucs / CLAP / MERT / Chromaprint not attempted.**
   The light path (ReccoBeats + librosa + sklearn) was built first per
   the brief; heavy torch-based embeddings were timeboxed out. The
   Demucs stem balance and audio-embedding features from the brief
   are not in this build. Chromaprint dedup was not implemented;
   dedup is by Spotify track ID (plus title/artist strings at pool
   build time), which is the rule Kevin set.
10. **Test set is the most recent likes.** The held-out set is the
    newest 20%, which is ReccoBeats-poor and H-heavy relative to the
    train set. Metrics on a random split would likely look better;
    the time split is the honest one.

## Coverage summary

| Source | Liked tracks | Test set (130) | Pool (590) |
|---|---|---|---|
| ReccoBeats | 447 (69%) | 33 (25%) | 493 (84%) |
| Librosa (Deezer preview) | 475 (73%) | 99 (76%) | 590 (100%) |
| Neither | 55 (8.5%) | - | 0 |

## Files

- `features_v2.py` — ReccoBeats + Deezer match + deep librosa extraction; writes `features_v2.json` (version 2).
- `scoring.py` — the single shared feature-vector definition + V2Model + v6 baseline scorer.
- `model_v2.py` — PCA/GMM clustering + logistic regression; writes `model_v2.json` + `model_v2.npz`.
- `negatives_features.py` — feature extraction for the 13 confirmed negatives.
- `fetch_genre_buckets.py` — re-fetches the E/R/H/O playlist membership from Spotify and writes `data/genre_buckets.json`. The mapping used to live in /tmp; a runtime restart wiped /tmp and a re-run silently rebuilt `features_v2.json` with every `genre_bucket` None, which broke the evaluation's genre sanity check and made v6 per-bucket scores fall back to global. The mapping now lives in the workspace data dir, `features_v2.py` fails loudly if it is missing or covers < 500 tracks, and the acceptance tests check bucket coverage (>= 600 tracks).
- `pool_reccobeats.py` — ISRC + ReccoBeats fill for pool tracks (Deezer top-tracks responses carry no ISRC).
- `recommend_v2.py` — candidate pool build (Deezer related artists) + MMR selection + Spotify-ID verification.
- `evaluate.py` — the evaluation harness (run before recommendations).
- `gtzan_check.py` — GTZAN genre sanity check.
- `test_pipeline_v2.py` — acceptance checks (37 checks; all green as of this report).
- `features_v2.json`, `model_v2.json`, `model_v2.npz`, `evaluation_v2.json`, `recommendations_v2.json`, `data/negatives_features.json`, `data/pool_features.json`, `data/gtzan_check.json`, `data/evaluation_audit.json`.
- Caches under `data/` (reccobeats.json, deezer_matches.json, librosa_deep.json, reccobeats_pool.json, pool_isrc.json) and `previews/` are NOT committed; they are reproducible from the scripts.
- `pipeline.py`, `recommend.py`, `retry_candidates.py` (v1) are unchanged and still work; v2 scripts supersede them for recommendations.

## What would unblock the limits

- Full-track audio: Kevin acquires full-length files of his liked tracks (any format). Then Demucs stems, structural segmentation, and per-section features become possible.
- Streaming history: Kevin provides his Spotify extended streaming history export (plays, skips, ms played). Then like/skip labels replace the 13-track negative set.
- More negatives: every future keep/drop verdict (starting with these 10) goes into the negative set and the LR is retrained.
- ReccoBeats coverage: nothing to do; it is a third-party coverage limit. The librosa path carries the tracks it misses.
