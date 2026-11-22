# evaluation_v3 API note — how integration should call the verdict function

Module: `~/workspace/audio-pipeline/evaluate_v3.py` (new; `evaluate.py` untouched).
Tests: `~/workspace/audio-pipeline/test_evaluate_v3.py` (97 checks, exit 0).
No network, no Spotify/Deezer calls. Pure functions over `(score, relevance, kind)` rows.

## Grounded constants (recomputed from the landed v2 files, not the spec)

- Ranked pool is **311**, not the spec's 343: 99 held-out likes with librosa
  (130 by position, 31 excluded incl. the unresolvable `'None'` id at
  position 85), **12 negatives with librosa** (13 defined; `Touch` by
  KATSEYE, `6aJn7Cst74cj4lNIiPRgav`, has no librosa), 200 distractors, seed 42.
- Random baseline: **P@10 = 0.318** (99/311). Random NDCG@10 is **~0.628**,
  NOT 0.318 — see the G9 note below. Random mean rank **156.0**.
- Negative control: P(0 of 12 in top-10 | random) = 0.676; n required for
  alpha = 0.05 is **92**; mean-rank 95% half-width at n = 12 is **50.8 ranks**,
  labelled UNDERPOWERED.
- The split is **position-based, never time-based**: `added_at` coverage is
  0/650. `evaluation_v2.json` `counts.test_likes` is 129, not 130, because
  `evaluate.py` drops the `'None'`-id track before counting.

## G9 — the one spec contradiction, resolved

Spec P-01 claims random NDCG@10 == random P@10. That equality holds only
under the *standard* NDCG ideal (min(k, n_relevant) ones). The spec's own
T-07 hand values (e.g. rel = [1,0,1,...] -> NDCG 0.920) and the landed
`evaluate.py` `ndcg_at_k` both build the ideal from the hits *inside* the
top-k window. Under that convention a random ranking scores NDCG@10 ~0.63,
and any ranking whose top-10 hits sit above its misses scores 1.0.
T-07 + `evaluate.py` govern: `random_baseline()` reports random NDCG@10 as
a deterministic Monte-Carlo expectation under the v2 convention
(seed 42, 2000 reps, cached per pool shape). Consequence: v2's NDCG@10
0.726 is only ~0.09 above the chance floor under this convention, while
P@10 0.50 vs 0.318 looks stronger. The GO gate's Leg 1 still uses the
bootstrap CI on the NDCG *difference*, which is unaffected.

## Calling the verdict (the integration contract)

```python
import sys; sys.path.insert(0, '/home/hatch/workspace/audio-pipeline')
import evaluate_v3 as ev3

# 1. Score ONE identical pool with both models (already true in v2).
#    items: dicts with 'spotify_id', 'kind' ('heldout'|'negative'|
#    'distractor'), 'relevant' (0/1), 'artist', 'bucket', and a score
#    column per model. v6 scores are lower-better.
m_v3 = ev3.rank_metrics(items, 'score_v3', higher_better=True)
m_v6 = ev3.rank_metrics(items, 'score_v6', higher_better=False)

# 2. Paired bootstrap on the difference (P-04). B=10,000, seed 42 by
#    default; ~3.5 s on this VM for a 311-item pool.
boot = ev3.paired_bootstrap(items, 'score_v3', 'score_v6',
                            higher_a=True, higher_b=False)

# 3. Leakage guards BEFORE trusting any metric (P-06, fail-fast).
ev3.check_leakage(top10_ids=m_v3['top10_ids'], train_ids={...},
                  heldout_ids={...}, negative_ids={...},
                  distractor_ids={...})

# 4. The verdict. Leg 4 inputs come from Kevin's judged lists; pass
#    live_keeps=None / live_n=None until judging reaches n=42.
verdict = ev3.verdict_4leg({
    'delta_ndcg_ci_low': boot['delta_ndcg_ci'][0],
    'delta_ndcg': boot['delta_ndcg'],
    'delta_p10': boot['delta_p10'],
    'delta_map10': m_v3['MAP@10'] - m_v6['MAP@10'],
    'p10_v3': m_v3['P@10'],
    'p10_v3_ci_low': boot['p10_a_ci'][0],
    'random_p10': m_v3['random_P@10'],          # 0.318 on the v2 pool
    'n_neg_top10': m_v3['n_neg_top10'],
    'n_neg_top10_H': m_v3['n_neg_top10_H'],
    'mean_rank_negative_v3': m_v3['mean_rank_negative'],
    'random_mean_rank': 156.0,                  # from random_baseline()
    'sampled_disagreement': ev3.sampled_metric_disagreement(
        delta_sampled=boot['delta_ndcg'], delta_full=...),  # P-10
    'live_keeps': None, 'live_n': None,         # until n=42 judged
    # optional: 'live_per_list_keeps': [8, 9, ...] — two consecutive
    # lists <= 7/10 force NO-GO even if the cumulative count passes.
})
print(verdict['verdict'])
# 'GO — v3 beats v6'  |  'NO-GO — offline'  |
# 'NO-GO (marginal) — binomial passes, Wilson fails; ...'  |
# 'NO-GO — offline signal did not transfer to listening'  |
# 'NO-GO — unresolved measurement disagreement ...'  |
# 'PENDING — live leg not yet judged (n=0/42); ...'
```

`verdict_4leg` returns `{'verdict', 'legs': {...}, 'offline_pass', 'note'}`.
Each leg entry carries `passed` plus the numbers it was decided on.
Leg 4's pass rule is generic, not hardcoded to n=42: Wilson 95% lower
bound > 0.60 AND exact one-sided binomial p <= 0.05 vs p0 = 0.60
(verified: 32/42 -> GO, 31/42 -> NO-GO (marginal), 37/50 -> GO).
If Legs 1-3 fail, do not spend a live list: the verdict is NO-GO offline.

## The other P-ports (all pure functions, see docstrings)

| Port | Function |
|---|---|
| P-01 random baseline | `random_baseline(n_heldout, n_ranked, k=10)` |
| P-02 date guard | `assert_split_allowed(split, added_at_coverage, total_liked)` — raises `SystemExit('TIME_SPLIT_REFUSED: ...')`; `split_label(split)` |
| P-03 metrics | `metrics_from_rel(rel, n_relevant, k=10)` (P/NDCG/MAP/MRR/Recall/Hit + `defined` + `n_hits` + warning); `METRIC_DEFINITIONS` states the log2 discount and the `min(k, n_relevant)` MAP normalisation |
| P-04 bootstrap + sweep | `paired_bootstrap(...)`, `draw_distractors(cands, n, seed)`, `seed_sweep_summary(per_seed)` where each entry is `{'p10_a','p10_b','ndcg_a','ndcg_b'}` |
| P-05 popularity | `popularity_audit(items, score_key, higher_better)` — flag if top-10 mean popularity exceeds pool mean by > 15 points; pool SD printed beside it; `popularity_source: 'unavailable'` + skip (never zero-fill) when no values exist |
| P-06 leakage | `check_leakage(...)` (raises `AssertionError`), `artist_overlap_fraction(train, test)` |
| P-07 ties | `rank_items(items, score_key, higher_better)` — sort `(-score, spotify_id)`, average ranks for exact ties; `rank_metrics` reports `n_tied_scores` |
| P-08 exclusions | `missing_feature_audit(n_heldout_total, excluded_ids, ranked_metrics, n_heldout_ranked)` — NOTE: the spec's literal worst-case imputation (excluded ranked last as relevance 0) leaves the top 10 unchanged, so `P@10_all == P@10` by construction; the function follows the spec literally and states this degeneracy in `imputation_note` rather than inventing another imputation. Judge exclusions by the count/fraction + audit list. |
| P-09 silent zeros | `metrics_from_rel` returns `defined: False`, `n_hits: 0`, and a `warning` string whenever a metric has zero hits — never a bare `0.0` |
| P-10 full-pool | `sampled_metric_disagreement(delta_sampled, delta_full)` — True only on opposite nonzero signs; feeds `verdict_4leg` and forces NO-GO on disagreement |
| P-11 artist split | `artist_disjoint_split(records, target_test_n)` — greedy whole-artist assignment, largest artists first; returns `(train_ids, test_ids)` |
| Power (T-06) | `negative_control_stats(n_ranked, n_ranked, k, n_defined)` |
| Real-data audit (T-01) | `audit_real_pool()` — replicates `evaluate.py`'s distractor draw and returns grounded counts + disjointness + split label; `python evaluate_v3.py` prints it as JSON |
| Live-leg stats | `wilson_ci(k, n)`, `binom_p_ge(k, n, p0=0.60)` |

## What this module deliberately does NOT do

- No recommender feature, no model change, no implicit/CF.
- No paired t-test (undefined at |U| = 1); bootstrap CI instead.
- No `Hit@K`/`Recall@K`/`AUC`/`novelty`/`coverage` gates (spec §(b) "explicitly NOT ported" — theater at this scale). Recall@10 and Hit@10 are reported by `metrics_from_rel` for the record only.
- No time-based split until `added_at` coverage >= 95% (618/650); until then every output says `position-based holdout (NOT time-based)`.
