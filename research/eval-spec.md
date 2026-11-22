# Eval-harness strengthening spec (v3): ported from RecBole + microsoft/Recommenders

- **Date:** 2026-11-21
- **Status:** Research spec for the v3 build. It strengthens the v2 harness; it does not replace it.
- **Scope:** Evaluation methodology only. No model changes, no new recommendations, no file edits outside this spec.
- **Upstream repos (cloned 2026-11-21 to `/tmp/RecBole` and `/tmp/Recommenders`, depth-1 clones):**
  - RecBole: `https://github.com/RUCAIBox/RecBole`
  - microsoft/Recommenders: `https://github.com/recommenders-team/recommenders`
- **Local harness under test:** `~/workspace/audio-pipeline/evaluate.py` (v2, read 2026-11-21),
  `~/workspace/audio-pipeline/negatives_features.py`, `~/workspace/audio-pipeline/gtzan_check.py`.
- **Hard limits this spec is built around:** 650 liked tracks with NO `added_at` dates;
  13 confirmed negatives (9 dropped v5 + 4 v6 thumbs-down, all H bucket); recommendation list
  length 10; one user (`|U| = 1`); VM 1.5 shared CPUs, ~6 GB RAM; default stack ReccoBeats +
  librosa on Deezer 30 s previews + scikit-learn; no full-length audio files.
- **Metrology rule for this spec:** every verdict is a named metric, a threshold, and a sample
  size. No adjective stands in for a number. Every number below was computed 2026-11-21
  (Python `math`/`scipy`; scripts at `/tmp/eval_numbers.py`, `/tmp/eval_numbers2.py`) or read
  verbatim from the cited source file.

## 0. Audit of the v2 harness as found (what the spec builds on)

Read `evaluate.py` in full on 2026-11-21. Verified facts (not assumptions):

| # | Fact | Source |
|---|------|--------|
| 0.1 | Test set = first 130 liked tracks by file `position` (`TEST_N = 130`, `test = [t for t in tracks if t['position'] < TEST_N]`). The docstring calls this "most recent 20% by list position". That label is **unverified**: it is true only if `spotify-liked-songs.txt` is ordered newest-first, and the file carries no dates to check against. | `evaluate.py`, `TEST_N`, `main()` |
| 0.2 | Ranking pool = 130 held-out likes + negatives + 200 distractors sampled from the candidate pool with `np.random.RandomState(42)`, `N_DISTRACTORS = 200`. Pool size at full coverage = 343 items. | `evaluate.py` |
| 0.3 | Metrics computed: `P@10`, `NDCG@10`, mean rank of held-out likes, mean score/rank of negatives, `top10_kinds`. No confidence interval, no random baseline, no significance test, no repeated split, no seed sweep. | `evaluate.py`, `rank_metrics()` |
| 0.4 | Negative set is loaded from `data/negatives_features.json`. The docstring says "(b) the 9 dropped v5 negatives", but `negatives_features.py` defines **13** negatives (9 v5 drops + 4 v6 thumbs-down). On 2026-11-21 `data/negatives_features.json` **did not exist** (directory contained only `deezer_matches.json`, `librosa_deep.json`, `reccobeats.json`), so whether the harness will rank 9 or 13 negatives is **unverified at spec time**. The spec requires an assertion (test T-01). | `evaluate.py` docstring; `negatives_features.py` `NEGATIVES`; `data/` listing 2026-11-21 |
| 0.5 | Only items WITH librosa features are ranked; excluded counts are printed but do not enter any metric denominator. A model that scores well only on the preview-available subset is not penalised. | `evaluate.py`, `main()` |
| 0.6 | `ndcg_at_k` builds the ideal DCG from `min(k, rel.sum())` ones, i.e. standard binary NDCG with log2 discount. `precision_at_k` is the plain mean of the top-k relevance vector. Both are single-ranking values, not means over users. | `evaluate.py`, `ndcg_at_k()`, `precision_at_k()` |
| 0.7 | Genre sanity check: multinomial logistic regression on train features predicts E/R/H/O bucket of held-out likes; accuracy + confusion matrix reported. GTZAN check is a separate file. | `evaluate.py`, `main()` tail; `gtzan_check.py` |
| 0.8 | Both methods rank the SAME pool (v2, v6-per-bucket, v6-global). This is the correct paired design; the spec keeps it and adds the paired statistics it lacks. | `evaluate.py` |

Consequence of 0.2 + 0.6: the random-ranking baselines for this pool are **not 0**.
With 130 positives in 343 pool items, relevance density `p = 130/343 = 0.379`, so
`E[P@10 | random] = 0.379`, `E[NDCG@10 | random] = 0.379`, `E[hits in top-10 | random] = 3.79`,
`E[Recall@10 | random] = 0.029`. Any v2/v3 P@10 below 0.379 is worse than chance, and
`P(hits >= 5 | random) = 0.315`, `P(hits >= 6 | random) = 0.133`: a single 5/10 or 6/10
offline result is weak evidence. The v2 harness reports no baseline, so a reader cannot
tell a real signal from this floor. Fix is port P-01.

---

## (a) How the upstream evaluation mechanisms work, with sources

### (a.1) RecBole — `https://github.com/RUCAIBox/RecBole`

RecBole is a PyTorch framework. Its evaluation is a config-driven pipeline:
config (`eval_args`) -> data split -> collector gathers per-batch resources ->
`Evaluator` runs every registered metric -> optional significance test across seeds.
No part of it runs on our stack (torch is a stretch goal, not the default); the
spec ports the *protocol*, not the code.

**A1. Metric registry and the two metric types.**
`recbole/evaluator/register.py` auto-discovers every metric class in
`recbole/evaluator/metrics.py` and records, per metric, `metric_need` (which
collector resources it consumes) and `metric_type` (`RANKING` = grouped by user,
`VALUE` = ungrouped). `recbole/evaluator/base_metric.py` defines `TopkMetric`
(needs `rec.topk`: a bool matrix of whether each top-k slot is a positive, plus
per-user positive counts) and `LossMetric` (needs `rec.score` + `data.label`).
`recbole/evaluator/evaluator.py` (`Evaluator.evaluate`) runs each configured
metric over the collected `DataStruct` and returns an `OrderedDict`.
Default config `recbole/properties/overall.yaml`: `metrics: ["Recall","MRR","NDCG",
"Hit","Precision"]`, `topk: [10]`, `valid_metric: MRR@10`, `metric_decimal_place: 4`,
`seed: 2020`, `reproducibility: True`.

**A2. Top-k metric definitions (exact formulas, from `recbole/evaluator/metrics.py`).**
- `Hit@K`: fraction of users with >= 1 positive in top-K.
- `MRR@K`: mean reciprocal rank of the first positive.
- `MAP@K`: average precision normalised by `min(|R-hat(u)|, K)` (not by the
  number of positives). Note this normalisation differs from Recommenders' `map`
  (normalised by number of relevant items) — see A9. Report which one is used.
- `Recall@K`: hits / `|R(u)|`.
- `NDCG@K`: binary DCG with `1/log2(rank+1)` discount, normalised by the ideal
  DCG over `min(|R(u)|, K)` positives.
- `GAUC`: per-user AUC weighted by positive count; **drops users with zero
  positives or zero negatives and logs a warning** (`metrics.py`, `GAUC.metric_info`).
- `AUC` (loss metric): global ROC AUC via scikit-learn on all scores/labels.
- Loss metrics: `MAE`, `RMSE`, `LogLoss` (predictions clipped to `[1e-15, 1-1e-15]`).

**A3. Beyond-accuracy metrics (same file).**
- `ItemCoverage@K = |union of recommended items| / |I|`.
- `AveragePopularity@K`: mean train-interaction count of recommended items; `smaller = True`.
- `ShannonEntropy@K`: entropy of the recommended-item distribution, **divided by the
  number of distinct recommended items** (non-standard normalisation; do not port blindly).
- `GiniIndex@K`: inequality of the recommendation-count distribution; `smaller = True`.
- `TailPercentage@K`: fraction of recommended items in the long tail; tail = bottom
  `tail_ratio` (default 0.1) of items by train count, or items with count <= `tail_ratio`
  when `tail_ratio > 1`.

**A4. Evaluation settings: split x order x mode (the core port).**
`recbole/properties/overall.yaml`, `eval_args`: `split: {'RS':[0.8,0.1,0.1]}`,
`group_by: user`, `order: RO`, `mode: full`. Docs (`https://recbole.io/docs/v1.0.1/user_guide/train_eval_intro.html`,
`https://recbole.io/docs/v0.1.2/recbole/recbole.config.eval_setting.html`):
- Order: `RO` random ordering, `TO` temporal ordering (sort by timestamp).
- Split: `RS` ratio-based (default 0.8/0.1/0.1 train/valid/test), `LS` leave-one-out
  (last record per user = test; `leave_one_num` default 2 in v0.2.0 config docs).
- Presets: `RO_RS`, `TO_RS`, `RO_LS`, `TO_LS`; records grouped by user by default.
- Mode (ranking candidates): `full` = rank all non-ground-truth items;
  `uniN` = each positive paired with N uniformly sampled negatives;
  `popN` = each positive paired with N popularity-sampled negatives.
  Defaults in v0.2.0 config docs: `eval_setting 'RO_RS,full'`, `group_by_user True`,
  `split_ratio [0.8, 0.1, 0.1]`, `metrics ['Recall','MRR','NDCG','Hit','Precision']`.
- Cited basis: Zhao et al., "Revisiting Alternative Experimental Settings for
  Evaluating Top-N Item Recommendation Algorithms", CIKM 2020 (named in the
  `EvalSetting` docstring).

**A5. Negative sampling (train + eval) and the leakage guard.**
`recbole/sampler/sampler.py` (`Sampler`): distributions `uniform` (default) and
`popularity` (alias-table sampling, `alpha = 1.0` power on popularity;
`train_neg_sample_args` in `overall.yaml`: `distribution: uniform, sample_num: 1,
alpha: 1.0, dynamic: False, candidate_num: 0`). The sampler is constructed with
the datasets of **all phases** and `get_used_ids()` accumulates positives
phase by phase, so train positives cannot be sampled as negatives in valid,
and train+valid positives cannot be sampled as negatives in test
(`sample_by_key_ids` re-samples any draw that lands in `used_ids`).
`RepeatableSampler` samples only items that never appeared in any phase.
Port: the *rule* (negatives must exclude every positive from every phase), not the class.

**A6. Reproducibility and significance testing.**
- `overall.yaml`: `seed: 2020`, `reproducibility: True`, `shuffle: True`.
- `significance_test.py` (repo root): runs each model `run_times = 10` times with
  random seeds drawn from `st_seed = 2023`, then `scipy.stats.ttest_rel(ours,
  baseline, alternative="less")` per metric, writing `statistic` and `pvalue` to
  `significant_test.txt`. It is a **paired t-test over repeated runs** (same seed
  list for both models), not over users.
- `recbole/evaluator/collector.py` (`Collector._average_rank`): tied scores get
  **average ranks** (SciPy-style), not first/last. Our harness sorts by score and
  takes order as-is; ties are currently order-dependent. Port the tie rule (P-07).

**A7. What RecBole does NOT solve for us.**
- Every top-k metric is a **mean over users**. With `|U| = 1` the mean is a single
  observation: there is no cross-user variance to average away, and a paired
  t-test over 10 seeds tests *seed noise*, not generalisation to new users.
  RecBole has no single-user mode and no confidence interval on a metric.
- `full` mode assumes the full catalogue is rankable. Our pool is 343 sampled
  items, i.e. RecBole `uni200`-flavoured sampled ranking, which is exactly the
  regime Krichene & Rendle show to be inconsistent (see A10).
- RecBole has no negative-control concept (confirmed human rejects) and no
  keep-rate / live-judgement metric. Those are added here, not ported.

### (a.2) microsoft/Recommenders — `https://github.com/recommenders-team/recommenders`

A framework-agnostic toolkit (pandas + Spark). Its evaluation module is
`recommenders/evaluation/python_evaluation.py` (pandas) and
`spark_evaluation.py`; splits live in `recommenders/datasets/python_splitters.py`
and `split_utils.py`; defaults in `recommenders/utils/constants.py`.
Docs stub: `docs/evaluation.rst` (autodoc only). Worked examples:
`examples/03_evaluate/evaluation.ipynb`,
`examples/03_evaluate/als_movielens_diversity_metrics.ipynb`.

**A8. Splitting.**
- `python_random_split(data, ratio=0.75, seed=42)`: global random split
  (`sklearn.model_selection.train_test_split`); multi-ratio lists are normalised
  by `process_split_ratio` (`split_utils.py`) and cut with `split_pandas_data_with_ratios`.
- `python_chrono_split(data, ratio, min_rating=1, filter_by="user", ...)`: per-user
  (or per-item) chronological split — sorts by `timestamp` within each user and
  cuts the last `(1-ratio)` fraction into test. **Raises `ValueError("Schema of data
  not valid. Missing Timestamp Col")` if the timestamp column is absent.**
  This is the direct precedent for our date rule (P-02): no timestamp column, no
  chrono split, no exception.
- `python_stratified_split(...)`: same per-user cut but on a random uniform
  (`np.random.rand`, seeded) instead of timestamp.
- `numpy_stratified_split(X, ratio, seed=42)`: per-user ratio preserved on the
  affinity matrix (each user's rated items cut by the same ratio, rounded).
- `min_rating_filter_pandas` / `filter_k_core` (`split_utils.py`): drop users/items
  below a minimum interaction count (warm-start filter). Not applicable at
  `|U| = 1`, but the *idea* (state the minimum-count rule explicitly) is ported
  as the librosa-availability filter audit (P-08).

**A9. Ranking metrics (`python_evaluation.py`).**
`DEFAULT_K = 10`, `DEFAULT_THRESHOLD = 10`, `SEED = 42` (`constants.py`).
All ranking functions share `merge_ranking_true_pred`, which restricts to users
present in both truth and prediction and computes hits via `get_top_k_items`
(rank = within-user `cumcount()+1` after sorting by prediction desc).
- `precision_at_k`: `sum(hit/k) / n_users`. Note: denominator is `k`, so the max
  is < 1 when a user has fewer than k predictions.
- `recall_at_k`: `sum(hit/actual) / n_users`.
- `r_precision_at_k`: precision at R = number of relevant items per user.
- `ndcg_at_k`: `score_type` in `binary | raw | exp` (default `binary`),
  `discfun_type` in `loge | log2` (default `loge` — **not log2**; our harness uses
  log2, so the two NDCGs differ by a constant factor and are not comparable
  unless the discount is stated). IDCG is computed from the user's own relevance
  scores, capped at k.
- `map` (Spark-MLlib convention): per-user average precision normalised by
  `n_relevant`. `map_at_k`: normalised by `min(k, n_relevant)`, always in [0,1].
  RecBole's MAP normalises by `min(|R-hat|, K)` instead. **Three MAPs exist; the
  spec fixes one** (P-03).
- Rating-error metrics: `rmse`, `mae`, `rsquared`, `exp_var` (all via
  `merge_rating_true_pred` inner join on user+item).
- `auc`, `logloss`: global (not per-user) ROC AUC / log-loss on binary ratings.
- **Silent-zero trap:** every ranking function returns `0.0` when `df_hit` is
  empty. A model with zero hits and a broken model with zero hits both report
  0.0 with no flag. The spec ports the opposite (P-09): empty-hit results are
  reported as `0.0` **plus** an explicit `n_hits = 0` field and a warning.

**A10. Diversity / novelty / serendipity / coverage (`python_evaluation.py`, lower half).**
All of these enforce a leakage guard in their decorators
(`_check_column_dtypes_diversity_serendipity`,
`_check_column_dtypes_novelty_coverage`): they **raise** if `reco_df` contains
any user-item pair already in `train_df`. Port that guard verbatim in spirit (P-06).
- `diversity` / `user_diversity`: `1 -` mean intra-list cosine similarity
  (Ziegler et al. 2005; Zhang et al., Auralist, WSDM 2012). Similarity modes:
  `item_cooccurrence_count` (default; needs multi-user co-occurrence — **not
  computable at |U| = 1**) and `item_feature_vector` (cosine over supplied
  feature vectors — **computable here**: librosa feature vectors are item features).
- `serendipity` / `user_serendipity`: `(1 - mean similarity of the recommended
  item to the user's interacted history) x relevance` (Zhang et al. 2012; Yan 2020).
  Directly portable with librosa cosine similarity and binary relevance.
- `historical_item_novelty` / `novelty`: item novelty `-log2(count / n_records)`
  (Castells et al., ECIR 2011, choice model), averaged over recommendations.
  **Degenerate at |U| = 1**: every train item has count 1, every unseen item has
  count 0, so novelty is a constant for train items and undefined for unseen ones.
  Port only with a different popularity proxy (P-05).
- `catalog_coverage`: distinct recommended items / distinct train items
  (Shani & Gunawardana, Recommender Systems Handbook 2010).
  `distributional_coverage`: entropy of the recommended-item distribution.
  At one user and one 10-list these are `10/650 = 0.015` and a single-list
  entropy: reportable, not decision-grade (section e).

**A11. Sampled-metric inconsistency (external result both frameworks ignore).**
Krichene & Rendle, "On Sampled Metrics for Item Recommendation", RecSys 2020
(`http://walid.krichene.net/papers/KDD-sampled-metrics.pdf`,
`https://cacm.acm.org/research/on-sampled-metrics-for-item-recommendation/`,
arXiv:1912.02263): sampled ranking metrics (rank positives among N sampled
negatives) are **inconsistent with their exact versions — they do not preserve
"A beats B" orderings, not even in expectation**, and as the sample shrinks all
metrics collapse toward AUC. Our v2 pool (200 sampled distractors + 13 negatives)
is a sampled ranking. Consequence: v2/v3 P@10 and NDCG@10 are **sampled-metric
estimates whose ordering of v2 vs v6 need not match the full-catalogue ordering**.
The spec's mitigations are P-01 (report the random baseline of the sampled pool),
P-04 (repeat the measurement over 5 distractor seeds and report the range), and
P-10 (full-pool re-rank as a sensitivity check, scored but not the primary metric).

### (a.3) Source index (every claim above traces to one of these)

| Claim area | Source |
|---|---|
| RecBole metric formulas (Hit/MRR/MAP/Recall/NDCG/GAUC/AUC/MAE/RMSE/LogLoss/Coverage/AvgPopularity/ShannonEntropy/Gini/TailPercentage) | `/tmp/RecBole/recbole/evaluator/metrics.py` |
| Metric types, `TopkMetric`/`LossMetric`, `metric_need` | `/tmp/RecBole/recbole/evaluator/base_metric.py` |
| Registry auto-discovery | `/tmp/RecBole/recbole/evaluator/register.py` |
| Evaluator loop, collector resources (`rec.topk`, `rec.meanrank`, `rec.score`, `rec.items`, `data.count_items`) | `/tmp/RecBole/recbole/evaluator/evaluator.py`, `/tmp/RecBole/recbole/evaluator/collector.py` |
| Defaults: seed 2020, RS [0.8,0.1,0.1], RO, full, metrics, topk [10], MRR@10, train neg-sample 1 uniform | `/tmp/RecBole/recbole/properties/overall.yaml` |
| Eval presets RO/TO x RS/LS, full/uniN/popN | `https://recbole.io/docs/v1.0.1/user_guide/train_eval_intro.html`, `https://recbole.io/docs/v0.1.2/recbole/recbole.config.eval_setting.html` |
| Negative sampler, uniform/popularity, used-ids leakage guard | `/tmp/RecBole/recbole/sampler/sampler.py` |
| Significance test: 10 runs, paired t-test | `/tmp/RecBole/significance_test.py` |
| Recommenders ranking metrics, silent 0.0, NDCG loge default, MAP vs MAP@K | `/tmp/Recommenders/recommenders/evaluation/python_evaluation.py` |
| Chrono/stratified/random splits, timestamp-required error, k-core filter | `/tmp/Recommenders/recommenders/datasets/python_splitters.py`, `/tmp/Recommenders/recommenders/datasets/split_utils.py` |
| Defaults K=10, threshold=10, SEED=42 | `/tmp/Recommenders/recommenders/utils/constants.py` |
| Diversity/novelty/serendipity/coverage + reco∩train guard | `/tmp/Recommenders/recommenders/evaluation/python_evaluation.py` (lower half) |
| Sampled metrics inconsistent, collapse to AUC | Krichene & Rendle 2020, links above |

---

## (b) Exact evaluation features to port, mapped onto our stack and VM limits

Stack facts the mapping relies on: numpy, pandas, scikit-learn already in use;
no torch, no Spark, no FAISS by default. All ports below are numpy/pandas/
scikit-learn only, run in seconds on 1.5 CPUs for a 343-item pool (the only
non-trivial cost is a 10,000-resample bootstrap over 343 items, estimated < 60 s).
Nothing here requires full-length audio; every port has a Deezer-preview-only path.
Full-audio features, if the friend's API ever lands, change only the feature
vectors, not the harness: the harness consumes `(score, relevance, kind)` rows.

Priority order is by effect on the v3 go/no-go decision (P-01 first).

### P-01. Report the random baseline beside every ranking metric (RecBole `full` vs sampled-mode lesson + A10). Priority 1 — decision-changing.
- **Port:** the concept that a sampled ranking has a non-zero chance floor.
- **Exact implementation:** in `evaluate.py`, after computing P@10 / NDCG@10 /
  Recall@10, also compute and print `random_P@10 = 130/343 = 0.379`,
  `random_NDCG@10 = 0.379`, `random_Recall@10 = 0.029`, `random_hits = 3.79`,
  derived from the *actual* counts in the run (`n_heldout / n_pool`), not hardcoded.
  General formula: `random_P@10 = random_NDCG@10 = n_heldout_with_features / n_ranked`.
- **Gate:** any model with `P@10 <= random_P@10` is reported as `BELOW_RANDOM`
  and cannot pass the GO gate (section d), whatever its margin over v6.
- **Why it changes the decision:** without it, a P@10 of 0.40 looks like a win
  over v6's hypothetical 0.30; with it, 0.40 is 0.02 above chance on a metric
  whose single-ranking SD under random is 0.153 (Binomial(10, 0.379)).

### P-02. Timestamp-or-it-isn't-chrono: hard date guard + position-based relabel (Recommenders `python_chrono_split` raises without a timestamp column). Priority 1.
- **Port:** Recommenders' refusal to run a chrono split without timestamps.
- **Exact implementation:**
  1. Re-fetch `added_at` for all 650 liked tracks via the Spotify API (separate
     data task, not this harness). Until then, `evaluate.py` must label the split
     `position-based holdout (NOT time-based)` in every output line and JSON field.
  2. Add an assertion: if any code path requests `split == "time-based"` while
     `added_at` coverage is < 95% of liked tracks (i.e. < 618 of 650), exit
     non-zero with `TIME_SPLIT_REFUSED: added_at coverage = X/650`.
  3. Anchor check for the position assumption (one-time, manual): Kevin names 3
     tracks he knows he liked in Nov 2026; if all 3 are in file positions 1-130,
     the file is newest-first and position<130 is a defensible *proxy* for recent;
     otherwise the proxy is rejected and the split is reported as `arbitrary
     position-based`. Record the anchor result in `evaluation_v2.json`
     (`position_order_anchor: {tracks: [...], positions: [...], verdict: ...}`).
  4. Once dates exist: switch to a true RecBole `TO_RS` split — sort by `added_at`,
     train on the oldest 80% (520 tracks), validate on the next 10% (65), test on
     the newest 10% (65) — and keep the position-based split as a sensitivity run.
     Note the test set then shrinks 130 -> 65, so `random_P@10` rises to
     `65 / (65 + 13 + 200) = 0.234` only if the pool is rebuilt at that size; the
     harness must recompute the baseline from actual counts every run (P-01).
- **Deezer-preview-only fallback:** unaffected; dates are metadata, not audio.

### P-03. Fix the metric definitions in writing and add MAP@10 + MRR@10 (RecBole A2 + Recommenders A9). Priority 1.
- **Port:** RecBole's explicit top-k set + Recommenders' `map_at_k` normalisation.
- **Exact implementation:** add to `rank_metrics()`:
  - `MAP@10` with `min(k, n_relevant)` normalisation (Recommenders `map_at_k`),
    where `n_relevant = 130` (held-out likes with features). State the formula in
    the JSON (`metric_definitions` block). Do NOT report RecBole's
    `min(|R-hat|, K)` MAP or Recommenders' `map` without naming which one it is.
  - `MRR@10` = reciprocal rank of the first held-out like (RecBole default
    `valid_metric` is `MRR@10`; adopting it makes the early-stopping metric and
    the reported metric the same object).
  - `NDCG@10` keeps log2 discount (v2 behaviour); record `discount: log2`
    explicitly, because Recommenders defaults to `loge` and the two are not interchangeable.
  - Keep `P@10` as the headline offline metric (it is the direct analogue of the
    10-track keep rate) and `NDCG@10` as the ranking-quality secondary.
- **Why:** MAP@10 and MRR@10 punish models that bury the first hit deep even when
  P@10 ties; on a 343-item pool two models can tie at P@10 = 0.50 while differing
  by > 0.15 in MRR@10. The GO gate (section d) uses P@10 + NDCG@10; MAP@10/MRR@10
  are tie-breakers, not gatekeepers.

### P-04. Paired bootstrap CI on the v3-minus-v6 difference + 5-seed repeated measurement (RecBole `significance_test.py`, adapted for |U| = 1). Priority 1.
- **Port:** RecBole's paired comparison with identical seeds for both models, and
  its 10-run repetition. **Adaptation (mandatory):** RecBole's paired t-test over
  runs tests seed noise; with one user there is no cross-user variance, so the
  resampling unit is the *ranked pool item*, not the user, and the test is a
  bootstrap, not a t-test (n = 1 user makes a t-test over users undefined).
- **Exact implementation:**
  1. Both models score the identical pool (already true in v2; keep it).
  2. Paired bootstrap: `B = 10,000` resamples, `seed = 42`, resampling the 343
     pool rows with replacement (stratified by kind: held-out / negative /
     distractor, so each resample keeps 130/13/200 counts); recompute P@10 and
     NDCG@10 for v3 and v6-per-bucket on each resample; report the 95% percentile
     CI of `delta_P@10` and `delta_NDCG@10`. Runtime on 1.5 CPUs: each resample
     is a sort of <= 343 rows; 10,000 resamples is < 60 s in numpy.
  3. Repeated measurement: re-run the whole harness with distractor seeds
     `42, 43, 44, 45, 46` (5 runs, same 200-distractor count, different draws)
     and report `min / median / max` of `delta_NDCG@10` and `delta_P@10`.
     The v2 single-seed result is one draw; test T-05 bounds how much that draw can move.
  4. Report all four numbers in `evaluation_v2.json` under `paired_bootstrap`
     and `seed_sweep`. No p-value is reported from a t-test (it would be a
     category error at |U| = 1); the bootstrap CI is the uncertainty statement.
- **Gate use:** section (d).

### P-05. Popularity-bias audit: AveragePopularity + TailPercentage with a Spotify-popularity proxy (RecBole A3, Recommenders A10 adapted). Priority 2.
- **Port:** RecBole `AveragePopularity@K` (`smaller = True`) and `TailPercentage@K`.
- **Adaptation (mandatory):** RecBole/Recommenders define popularity as train
  interaction count. At |U| = 1 every train item has count 1, so that definition
  is a constant and useless. Substitute Spotify `popularity` (0-100, from the
  ReccoBeats/Spotify record if present; else Deezer `rank` if present; else the
  field is `null` and the metric is skipped with `popularity_source: unavailable`).
- **Exact implementation:** for each model's top-10, report
  `mean_popularity_top10`, `mean_popularity_pool`, `tail_share_top10` where tail
  = bottom 10% of pool by popularity (`tail_ratio = 0.1`, RecBole default).
  Flag `POPULARITY_BIAS` if `mean_popularity_top10 - mean_popularity_pool > 15`
  popularity points (a 15-point gap is ~1 SD on the 0-100 scale for typical
  pools; the harness must print the pool SD beside the flag so the 15 is auditable).
- **Why it matters:** a model that wins P@10 by recommending only famous tracks
  is exploiting the pool, not Kevin's taste; the v6 failure mode (all 4 thumbs-down
  in the H bucket) is a taste failure, and popularity is the cheapest confound to check.
- **Theater warning:** do not report Recommenders `novelty` (degenerate, A10) or
  RecBole `ShannonEntropy` with its divide-by-distinct-count normalisation.
  If entropy is wanted, report plain Shannon entropy of the top-10 artist
  distribution with the formula stated.

### P-06. Leakage guards: reco ∩ train = empty, and negatives exclude all positives (Recommenders decorator guard + RecBole `Sampler.get_used_ids`). Priority 2.
- **Port:** Recommenders raises if any recommended pair is already in train;
  RecBole never samples a positive of any phase as a negative.
- **Exact implementation (3 assertions, all fail-fast, exit non-zero):**
  1. `top10 ∩ train_likes = ∅` for every model (spotify_id level). The v3 list
     must never contain a liked track; a violation is a harness/model bug, not a metric.
  2. `negatives ∩ heldout_likes = ∅` and `negatives ∩ train_likes = ∅`
     (the 13 negatives are rejected recommendations, not likes; overlap means a
     data-labelling error).
  3. `distractors ∩ heldout_likes = ∅` and `distractors ∩ negatives = ∅`
     (sampled distractors are re-drawn until disjoint; record `n_redraws`).
  4. Artist-level leakage audit (report-only, not fail-fast): report
     `artist_overlap_fraction` = share of held-out likes whose artist also appears
     in train. If it exceeds 0.50, the harness prints `ARTIST_LEAKAGE_WARNING`:
     P@10 is partly artist memorisation, and the artist-disjoint split in P-11
     becomes the binding sensitivity check.

### P-07. Deterministic tie-breaking: average ranks (RecBole `Collector._average_rank`). Priority 2.
- **Port:** tied scores share the average of their ranks.
- **Exact implementation:** in `rank_metrics()`, sort by `(-score, spotify_id)`
  (score desc, spotify_id asc as the deterministic breaker) and compute ranks
  with average-rank handling for exact score ties (numpy: `scipy.stats.rankdata`
  with `method="average"`, or a 10-line manual implementation — no new dependency
  beyond scipy, which scikit-learn already pulls in). Record `n_tied_scores` in
  the JSON. Re-running the harness on identical inputs must produce bit-identical
  `evaluation_v2.json` (test T-08).
- **Why:** GMM/LR scores are floats and ties are rare, but the v6 distance score
  is a sum of small integers-ish bucket deviations; ties at the top-10 boundary
  flip P@10 by 0.10 with no model change. Unrecorded tie order is unmeasured variance.

### P-08. Missing-feature exclusion audit (Recommenders `min_rating_filter` idea: state the filter, count what it removes). Priority 2.
- **Port:** the discipline of stating the filter and its yield, not the filter itself.
- **Exact implementation:** report, per model and overall:
  `n_heldout_total = 130`, `n_heldout_with_librosa`, `n_heldout_excluded`,
  and the same triple for negatives and distractors; plus `excluded_spotify_ids`
  (audit list). Add a sensitivity metric: `P@10_all` recomputed with excluded
  held-out likes counted as relevance 0 and placed at the bottom of the ranking
  (worst-case imputation). If `P@10 - P@10_all > 0.10`, print
  `MISSING_FEATURE_WARNING`: the headline metric is flattered by exclusions.
- **Deezer-preview-only note:** exclusions happen when a 30 s preview is missing
  or unmatchable; full-audio availability would shrink exclusions but the audit
  stays identical in form.

### P-09. No silent zeros: empty-hit and degenerate-metric reporting (fix for Recommenders' `return 0.0`). Priority 3.
- **Port:** the inverse of Recommenders' silent-zero behaviour.
- **Exact implementation:** whenever a metric's hit count is 0, or a metric is
  undefined (e.g. GAUC-style per-user AUC with no negatives, NDCG with zero
  positives in the ranked set), emit the numeric value **and**
  `n_hits`, `defined: false`, and a one-line warning. Never let `0.0` appear
  without its `n_hits`. This applies to P@10, NDCG@10, MAP@10, MRR@10, and the
  negative-control counts.

### P-10. Full-pool sensitivity re-rank (RecBole `full` mode as a check on the sampled pool; Krichene & Rendle A11). Priority 3.
- **Port:** RecBole `mode: full` — rank everything, not a sample.
- **Exact implementation:** once per harness run, score **all** pool candidates
  the v2 build has features for (the full candidate pool file, not just the 200
  sampled distractors) plus the 130 held-out likes and 13 negatives, and report
  `P@10_fullpool` and `NDCG@10_fullpool` for v3 and v6. This is a sensitivity
  check, not the primary metric (the pool is still a candidate subset, not the
  Spotify catalogue). If `sign(delta_fullpool) != sign(delta_sampled)` the
  harness prints `SAMPLED_METRIC_DISAGREEMENT` and the GO gate (section d)
  cannot pass on the sampled metric alone — the disagreement must be resolved
  by inspection before any GO.
- **Cost:** scoring is one forward pass per item per model over a pool of
  low-thousands of rows; trivial on this VM. No new data fetch.

### P-11. Artist-disjoint split as a stress test (RecBole `group_by` + LS idea, adapted). Priority 3.
- **Port:** RecBole's grouping logic: the split unit need not be the interaction.
- **Exact implementation:** build a second split in which no artist appears in
  both train and test (greedy assignment of artists to test until >= 130 test
  tracks, or as close as possible; report the achieved test size). Report
  `P@10_artist_disjoint` and `NDCG@10_artist_disjoint` for v3 and v6.
  This is diagnostic only: a model that only memorises artists collapses here,
  and the collapse size is the honest measure of artist memorisation. It is not
  a gate, because Kevin genuinely likes some artists repeatedly and an
  artist-disjoint test understates production value.
- **Blocked until:** nothing — artist strings are already in the data. Do it in v3.

### Explicitly NOT ported (theater on this data)
| Upstream feature | Why not |
|---|---|
| RecBole `Hit@K`, `Recall@K` as headline metrics | `Recall@10 = hits/130`, max 0.077 at 10 hits; it compresses every real difference into the third decimal. Report it (cheap) but never decide on it. `Hit@K` is 1.0 whenever any held-out like cracks the top 10 — saturated, uninformative. |
| RecBole `GAUC` / Recommenders `auc` as headline | Global AUC over a 343-item pool with 130 positives measures "better than random ordering", not top-10 quality; Krichene & Rendle: sampled metrics collapse toward AUC, so a good AUC with a bad P@10 is the expected failure shape, not a consolation. Report AUC once as a diagnostic; gate on P@10/NDCG@10. |
| Recommenders `novelty` (`-log2(count/n)`) | Degenerate at \|U\| = 1 (A10). Replaced by P-05's popularity proxy. |
| RecBole `ShannonEntropy` (divided by distinct count) | Non-standard normalisation; would mislead. Plain entropy or nothing. |
| RecBole `ItemCoverage@K` / Recommenders `catalog_coverage` as a gate | One user, one 10-list: coverage = 10/650 = 0.015 by construction. Report the raw unique-artist count of the top 10 instead (a 10-list with 10 artists vs 3 artists is a real diversity signal at this scale). |
| RecBole `significance_test.py` paired t-test | Undefined / misleading at \|U\| = 1. Replaced by P-04 bootstrap. |
| Torch / Spark evaluation code | Out of scope for the light stack; the ports above are numpy/pandas/scikit-learn. |

---

## (c) TDD acceptance tests written FIRST

All tests live in `~/workspace/audio-pipeline/research/test_eval_harness.py`
(new file, v3 build) and run with the repo's Python (`pytest` or plain asserts;
no network, no Spotify/Deezer calls — all fixtures are synthetic).
**Each test must fail on the unmodified v2 harness and pass after the port.**
That is the TDD contract: write the test, watch it fail against `evaluate.py`
as found, then implement.

### Fixture (shared by T-02–T-07)
Synthetic pool: 130 held-out likes (ids `H000`–`H129`), 13 negatives
(`N00`–`N12`), 200 distractors (`D000`–`D199`), each with a `score_v3` and
`score_v6` column. A `make_pool(seed)` helper builds it. All tests use
`seed = 42` unless the test itself varies the seed.

### T-01. Negative-count and pool-shape assertion (P-01, P-06)
- **Setup:** run the harness entry point on the real `data/` directory.
- **Assert:** `counts.test_likes == 130`, `counts.negatives == 13`,
  `counts.distractors == 200`, `counts.n_ranked == 343` (minus documented
  librosa exclusions, each counted in the audit list).
- **Fail condition (must fail pre-port):** the v2 harness has no such assertion;
  its docstring says 9 negatives while `negatives_features.py` defines 13, and
  `data/negatives_features.json` did not exist on 2026-11-21. The test fails
  until the assertion exists and the file contains 13 records.
- **Also assert:** `negatives ∩ heldout = ∅`, `negatives ∩ train = ∅`,
  `distractors ∩ heldout = ∅`, `top10 ∩ train = ∅` for both models.

### T-02. Random-baseline calibration — the harness knows chance (P-01)
- **Setup:** assign i.i.d. uniform random scores to all 343 pool items
  (seed 42), compute P@10 and NDCG@10. Repeat 200 times with seeds 42–241.
- **Assert:** `mean(P@10) ∈ [0.359, 0.399]` and `mean(NDCG@10) ∈ [0.359, 0.399]`
  (true value 0.379; ±0.02 allows Monte-Carlo error: SE of the mean over 200
  reps is 0.153/sqrt(200) = 0.011, so ±0.02 is ~1.8 SE — tight but attainable;
  if flaky, widen to ±0.025 and record why).
- **Assert:** the JSON contains `random_P@10 == 0.379` (±0.001, recomputed from
  actual counts) and a `BELOW_RANDOM` flag is present and false for a model
  scoring P@10 = 0.50 on the same fixture.
- **Fail condition pre-port:** v2 reports no random baseline at all.

### T-03. Degraded-model detection — the harness convicts a bad model (P-04)
- **Setup A (inverted scores):** `score_bad = -score_v3` on the fixture where
  v3 is constructed to rank 8 held-out likes in its top 10. **Assert:**
  `P@10_bad <= 0.20`, `NDCG@10_bad < NDCG@10_v3 - 0.20`, and
  `mean_rank_negative_bad < 172` (negatives float up when scores are inverted;
  172 is the random mean rank over 343 items), and the GO gate returns False.
- **Setup B (shuffled scores):** randomly permute v3's scores (seed 7).
  **Assert:** `|P@10_shuffled - 0.379| <= 0.25` on a single draw (wide because
  one draw is noisy — the point is it is near chance, not near v3's 0.80),
  and over 50 shuffles `mean(P@10_shuffled) ∈ [0.33, 0.43]`.
- **Setup C (popularity-only):** score = synthetic popularity column, constructed
  so the top 10 are the 10 most popular distractors. **Assert:** GO gate returns
  False and `POPULARITY_BIAS` flag is raised (`mean_popularity_top10 -
  mean_popularity_pool > 15`).
- **Fail condition pre-port:** v2 has no GO gate and no popularity flag, so
  setups A–C produce numbers with no verdict; the test asserting a verdict fails.

### T-04. Positive control — the harness detects memorisation (leakage test)
- **Setup:** fit the scoring profile on the held-out likes themselves (train =
  test: the "model" is the test-set centroid).
- **Assert:** `P@10 >= 0.90` and `NDCG@10 >= 0.90`. If the harness reports less,
  the harness is broken (it cannot see a model that has memorised the answers),
  and the test fails. This is the mirror image of T-03: a harness that cannot
  convict a bad model *and* cannot crown a cheating model is measuring nothing.
- **Also assert:** the leakage guard (P-06) fires on this fixture if any test
  item is present in the declared train set — i.e. the test runs the guard in
  `expect_raise` mode and confirms the exit is non-zero.

### T-05. Seed sensitivity — one distractor draw must not decide the verdict (P-04)
- **Setup:** same models, distractor seeds 42–46 (5 draws of 200 from a fixed
  5,000-item synthetic candidate pool).
- **Assert:** `max(P@10) - min(P@10) <= 0.20` (2 hits in the top 10) for each
  model, and `sign(delta_NDCG@10)` is the same in >= 4 of 5 seeds.
- **Fail condition pre-port:** v2 runs one seed, so the range is unmeasured;
  the test asserting a 5-seed sweep fails.

### T-06. Negative-control power — 13 negatives cannot clear the bar, 102 can (P-01 + section d)
- **Setup (analytic, no fixture):** pool 343, random ranking, `p_top10 = 10/343 = 0.0292`.
- **Assert:** `P(0 of 13 negatives in top-10 | random) = 0.681` (±0.001) —
  i.e. observing zero negatives in the top 10 with n = 13 is *expected* under
  chance and proves nothing.
- **Assert:** the harness computes `n_negatives_required = 102` (smallest n with
  `(1 - 10/343)^n <= 0.05`; exact value 101.2, so 102) and prints it in the JSON
  as `negative_control.n_required_for_alpha_0.05`.
- **Assert:** with the 13 real negatives, the harness reports the mean-rank
  95% half-width `1.96 * 99.0 / sqrt(13) = 53.8` ranks (uniform-rank SD over
  343 items is 99.0) and labels the negative control `UNDERPOWERED`
  (half-width > 25 ranks is the labelling rule).
- **Fail condition pre-port:** v2 reports a mean negative rank with no
  uncertainty and no required-n.

### T-07. Metric-definition unit tests (P-03)
Hand-computed vectors, no fixture needed:
- `rel = [1,1,0,0,0,0,0,0,0,0]` -> `P@10 = 0.20`, `NDCG@10 = 1.000`, `MRR@10 = 1.000`.
- `rel = [0,1,0,0,0,0,0,0,0,0]` -> `P@10 = 0.10`, `NDCG@10 = 0.631`
  (`(1/log2 3) / 1`), `MRR@10 = 0.500`.
- `rel = [1,0,1,0,0,0,0,0,0,0]` -> `P@10 = 0.20`,
  `NDCG@10 = 0.920` (`DCG = 1 + 1/log2 4 = 1.500`; `IDCG = 1 + 1/log2 3 = 1.631`).
- `rel = [0]*10` -> `P@10 = 0.0`, `NDCG@10 = 0.0`, `defined: false`,
  `n_hits = 0` with a warning emitted (P-09), not a bare 0.0.
- `MAP@10` on `rel = [1,0,1,0,0,0,0,0,0,0]` with `n_relevant = 130`:
  average precision = `(1/1 + 2/3) / min(10, 130)` = `(1 + 0.667) / 10 = 0.167`.
  (States the `min(k, n_relevant)` normalisation explicitly.)
- **Assert** each value to ±0.001. **Fail condition pre-port:** v2 has no
  MAP/MRR and no `defined` flag; the test fails on missing keys.

### T-08. Determinism and tie handling (P-07)
- **Setup:** fixture with 5 items sharing the exact top score.
- **Assert:** two harness runs on identical inputs produce bit-identical
  `evaluation_v2.json` (byte comparison), and tied items receive average ranks
  (the 5 tied items at positions 1–5 each get rank 3.0).
- **Fail condition pre-port:** v2 sorts by score only; order among ties follows
  input order, which is not recorded as a tie-break rule.

### T-09. Date guard (P-02)
- **Setup:** call the harness with `split = "time-based"` and a liked-tracks
  file with no `added_at` column.
- **Assert:** exit code non-zero, stderr contains `TIME_SPLIT_REFUSED`, and no
  `evaluation_v2.json` is written.
- **Assert:** with `split = "position-based"`, the JSON contains
  `split_label: "position-based holdout (NOT time-based)"`.
- **Fail condition pre-port:** v2's docstring calls the split "most recent 20%"
  with no date check anywhere.

### T-10. Missing-feature sensitivity (P-08)
- **Setup:** fixture where 10 of the 130 held-out likes have `librosa = None`.
- **Assert:** JSON reports `n_heldout_excluded == 10` with their ids listed,
  and `P@10_all` (excluded counted as relevance 0, ranked last) is reported
  beside `P@10`; if `P@10 - P@10_all > 0.10`, `MISSING_FEATURE_WARNING` is set.
- **Fail condition pre-port:** v2 prints exclusion counts to stdout only; they
  are not in the JSON and no worst-case metric exists.

### T-11. Full-pool disagreement flag (P-10)
- **Setup:** fixture where the sampled-pool delta and the full-pool delta have
  opposite signs (construct by planting high-scoring distractors outside the
  sampled 200).
- **Assert:** JSON contains `SAMPLED_METRIC_DISAGREEMENT: true` and the GO
  gate returns False.
- **Fail condition pre-port:** v2 has no full-pool run.

### Test-run contract
- Every test above names the port it covers; a port with no failing-then-passing
  test is not done. `pytest research/test_eval_harness.py` must exit 0 on the
  v3 harness and non-zero on the v2 harness (verified by running the suite
  against a `git stash` of the v2 files — the v3 builder records the pre-port
  failure list in the test file header).

---

## (d) The metric that proves v3 beats the v6 6/10 baseline: exact decision rule

### The problem in numbers
- v6 baseline: **6/10 keeps** on one 10-track list, second listen.
  Wilson 95% CI for 6/10 = **[0.313, 0.832]**; Clopper-Pearson exact 95% CI =
  **[0.262, 0.878]**. The baseline's own uncertainty spans 0.31–0.88: a single
  new 10-list cannot separate "same as v6" from "much better" without a rule
  that says so.
- One new 10-list, one-sided binomial test vs `p0 = 0.60`:
  `P(X >= 7) = 0.382`, `P(X >= 8) = 0.167`, `P(X >= 9) = 0.046`, `P(X >= 10) = 0.006`.
  Only **9/10 or 10/10** rejects `p = 0.60` at `alpha = 0.05` on a single list.
  8/10 does not (p = 0.167). 7/10 does not (p = 0.382).
- Sample size to detect `0.60 -> 0.80` at 80% power, `alpha = 0.05` two-sided:
  **42 judged tracks** (one-sample vs fixed 0.60; normal approximation; exact
  binomial power at n = 42 with critical value 31 is 0.882). Two-sample
  (fresh v6 list vs fresh v3 list) needs **82 judged tracks per arm**.
  Detecting `0.60 -> 0.70` needs **181 judged tracks** one-sample — out of reach;
  the harness is not designed to detect a 10-point keep-rate gain, and the spec
  says so rather than pretending otherwise.
- At n = 42 the Wilson-lower-bound rule (lower bound > 0.60) requires **k >= 32
  keeps (0.762)**; Wilson CI at 32/42 = [0.615, 0.865]; one-sided binomial
  p = 0.021. At n = 50 it requires **k >= 37 (0.740)**; Wilson [0.604, 0.841],
  p = 0.028. (31/42 has p = 0.045 but Wilson lower bound 0.589 < 0.60, so it
  fails the Wilson leg — the two legs together are the rule.)

### The decision rule (all four legs must pass; a single leg failing = NO-GO)

The v3 GO decision is made on **two independent evidence streams** — offline
ranking on the 130-item holdout, and live keep rate on judged 10-track lists —
because neither alone has the power. The rule is pre-registered here so the
v3 builder cannot move the goalposts after seeing the numbers.

**Leg 1 — Offline ranking beats v6 (primary).**
`delta_NDCG@10 = NDCG@10(v3) - NDCG@10(v6-per-bucket)` and
`delta_P@10 = P@10(v3) - P@10(v6-per-bucket)`, both on the identical 343-item pool.
- **Pass:** lower bound of the 95% paired-bootstrap CI (P-04, B = 10,000,
  seed 42) on `delta_NDCG@10` is **> 0.00**, AND point estimate
  `delta_P@10 >= +0.10` (at least 1 extra held-out like in the top 10).
- v6-per-bucket is the comparator (it is the stronger v6 variant in the v2
  design: per-bucket profile vs global profile). v6-global is reported but
  does not gate.
- **Tie handling:** if `delta_P@10` is exactly 0.00 and the NDCG CI excludes 0,
  the leg is a **tie on P@10, win on NDCG**: it passes Leg 1 only if
  `delta_MAP@10 >= +0.02` as well (MAP@10 breaks the tie). Otherwise it fails.
- **Uncertainty handling:** if the bootstrap CI for `delta_NDCG@10` includes 0,
  Leg 1 fails regardless of the point estimate. No "trending positive" verdict exists.

**Leg 2 — Offline ranking beats chance (sanity floor).**
- **Pass:** `P@10(v3) >= 0.50` (at least 5 of 10 top slots are held-out likes,
  vs 3.79 expected under random ranking) AND the lower 95% bootstrap bound of
  `P@10(v3)` is **> 0.379** (the random baseline, P-01).
- Note `P(hits >= 5 | random) = 0.315`: the 0.50 point threshold alone is weak,
  which is why the CI leg is attached to it.

**Leg 3 — Negative control does not incriminate v3 (guardrail).**
- **Pass:** **0 of the 13 gold negatives** in v3's top 10, AND
  `mean_rank_negative(v3) >= mean_rank_negative(v6-per-bucket)` is NOT required
  (v6's negative ranks are themselves noisy) — instead:
  `mean_rank_negative(v3) > 172` (above the random mean rank over 343 items),
  reported with its 95% half-width of **±53.8 ranks** (n = 13) and labelled
  `UNDERPOWERED` in the JSON. The leg passes on the point conditions; the
  label records that the control cannot prove much (section e, T-06).
- **Fail (hard):** >= 1 gold negative in v3's top 10, OR
  `mean_rank_negative(v3) <= 172` (negatives rank better than chance).
- The 4 v6 thumbs-down negatives (all H bucket) are reported separately from
  the 9 v5 drops: `n_neg_top10_H` must be 0 as its own sub-condition, because
  the v6 failure mode was H-specific and a v3 that repeats it in H has failed
  even if its overall negative count is 0 elsewhere. (With 13 negatives this
  sub-condition is the same count split two ways; it becomes independently
  meaningful as more H negatives accumulate.)

**Leg 4 — Live keep rate (accumulated, not single-list).**
- A single v3 10-list is judged by Kevin (keep / drop per track, second listen,
  same protocol as v6):
  - **9/10 or 10/10:** live evidence of improvement (one-sided binomial
    p <= 0.046 vs 0.60). Counts toward the cumulative total below.
  - **8/10:** promising, not proof (p = 0.167). Counts toward the cumulative total.
  - **7/10 or below:** no live evidence of improvement (p >= 0.382). Counts
    toward the cumulative total; two consecutive lists at <= 7/10 trigger a
    NO-GO review even if Legs 1–3 pass.
- **Cumulative rule (binding):** accumulate judged v3 tracks across lists until
  **n = 42** (5 lists of 10 = 50 tracks covers it; use the first 42 judged).
  **Pass:** keeps **k >= 32 of 42** (keep rate >= 0.762), i.e. one-sided exact
  binomial p <= 0.021 vs 0.60 AND Wilson 95% lower bound > 0.60.
  If judging continues to n = 50, the threshold becomes **k >= 37 of 50**.
- **Sequencing:** Legs 1–3 are computed before any new list is judged (they use
  only the holdout). If Legs 1–3 fail, no live list is spent on v3: the verdict
  is NO-GO on offline evidence and the live leg is not run. If Legs 1–3 pass,
  the live leg is the confirmation: GO requires Leg 4 to pass at n = 42.
  A single 9/10 or 10/10 first list is **not** a GO by itself — it is a
  green light to keep judging to n = 42. The spec forbids declaring v3 better
  than v6 on one list.

### Verdict table

| Legs 1–3 (offline) | Leg 4 at n = 42 (live) | Verdict |
|---|---|---|
| Pass | k >= 32/42 | **GO — v3 beats v6** |
| Pass | k = 31/42 (p = 0.045, Wilson lower 0.589) | **NO-GO (marginal)** — binomial passes, Wilson fails; do not ship on a split decision |
| Pass | k <= 30/42, or two lists <= 7/10 | **NO-GO — offline signal did not transfer to listening** |
| Fail (any leg) | Not run | **NO-GO — offline** |
| `SAMPLED_METRIC_DISAGREEMENT` (P-10) | Any | **NO-GO — unresolved measurement disagreement**; investigate before re-deciding |

### What the rule deliberately does not do
- It does not use a paired t-test (RecBole's `significance_test.py`): undefined
  at |U| = 1. Bootstrap CI instead (P-04).
- It does not declare victory from `delta_P@10` alone: with a single ranking,
  P@10 moves in steps of 0.10 and its random SD is 0.153.
- It does not average the v6 6/10 with new data: 6/10 is the fixed null `p0 = 0.60`,
  not a sample to pool. Pooling would double-count Kevin's judgement history.
- It does not set the bar at 7/10 or 8/10 for a single list: those p-values
  (0.382, 0.167) are not evidence at alpha = 0.05.

---

## (e) What it cannot do on our data and what would unblock it

### E-01. True time-based holdout — BLOCKED, no workaround in the harness
- **Cannot:** run RecBole `TO_RS` / Recommenders `python_chrono_split`. Both
  require a timestamp column; Recommenders raises without one. Our liked file
  has no `added_at` (650 rows, title/artist/Spotify ID only).
- **Cost of the block:** the "most recent 20%" test set is a position proxy of
  unverified ordering (audit 0.1). If the file is actually oldest-first, the
  test set is Kevin's *earliest* likes and the harness measures backward
  generalisation; if it is shuffled, the split is random with a misleading name.
  Either way, "v3 generalises to future taste" is unproven until dates exist.
- **Unblock:** re-fetch `added_at` for all 650 tracks from the Spotify API
  (`GET /me/tracks` returns `added_at` per item; paginate 50/page, 13 pages)
  and store `added_at` beside each Spotify ID. Then P-02 switches the split to
  true `TO_RS` (520/65/65). Cost: one API script, no audio work.
  **Until then the harness must say "position-based", never "time-based" (T-09).**

### E-02. Negative control with statistical power — BLOCKED at n = 13
- **Cannot:** distinguish "v3 avoids Kevin's rejects" from chance. Numbers:
  `P(0 of 13 negatives in top-10 | random) = 0.681`; mean-rank 95% half-width
  **±53.8 ranks** on a 1–343 scale. A model could place all 13 negatives at
  rank ~120 (bad) and the control would still look fine.
- **Unblock (three rungs, in order):**
  1. **n = 102 negatives** makes `P(0 in top-10 | random) <= 0.05` (exact
     requirement 101.2). Mean-rank half-width at n = 100 is ±19.4 ranks.
  2. **How to get there honestly:** each new judged 10-list at a 6/10 keep rate
     yields ~4 real negatives. 5 lists yield ~20 negatives (total ~33) — still
     underpowered. Reaching 102 real negatives takes ~22 lists at that rate:
     months of listening, not a v3 deliverable.
  3. **Silver negatives (interim, labelled as such):** hard pseudo-negatives =
     tracks the model scores in its top 30 that Kevin has *never* liked and that
     share an artist or bucket with a gold negative — reported in a separate
     `silver_negative` block, never mixed into the gold-13 count. They measure
     "does the model avoid the neighbourhood of known rejects", not "does it
     avoid rejects". If silver negatives are used, the harness must print
     `SILVER_NEGATIVES_ARE_NOT_GOLD` on every run.
- **What NOT to do:** do not sample 89 random distractors and call them
  negatives. Unlabelled tracks are not negatives (RecBole's sampler excludes
  positives but never claims sampled items are true negatives); that would
  convert the control into a popularity test.

### E-03. Any claim about a second user, a new listener, or "users like Kevin" — IMPOSSIBLE
- **Cannot:** RecBole and Recommenders both average metrics over users; every
  cross-user statement needs `|U| >= 2`. We have `|U| = 1`. All CIs in this
  spec are over *items and splits*, not users. The v3 verdict is "better for
  Kevin's held-out likes and his next judged lists", nothing broader.
- **Unblock:** nothing in scope. Do not recruit a second user to fix a metric.

### E-04. Detecting a 10-point keep-rate gain (0.60 -> 0.70) — IMPOSSIBLE at feasible n
- **Cannot:** needs **181 judged tracks** one-sample (80% power, alpha 0.05).
  That is 19 ten-track lists. The v3 decision is therefore powered only for
  large gains (>= 0.60 -> 0.80, n = 42). A v3 that is truly 0.70 will most
  likely read as NO-GO (marginal) under this rule — a known false-negative
  rate the spec accepts rather than lowering alpha to manufacture wins.
- **Unblock:** more judged lists (19 lists), or accept the offline legs as the
  decision and treat the live leg as a guardrail only. The spec does not make
  that trade; the parent agent may, explicitly, by striking Leg 4.

### E-05. Coverage / diversity / novelty as decision metrics — THEATER at this scale
- **Cannot:** `catalog_coverage = 10/650 = 0.015` by construction;
  Recommenders `novelty` is degenerate (A10); RecBole `ItemCoverage` needs a
  multi-user, multi-list denominator. A single 10-list's diversity is better
  expressed as the raw count of distinct artists (10 vs 3 is informative;
  0.015 vs 0.017 is not).
- **Unblock:** multiple judged lists over time (diversity/coverage become
  meaningful as *cumulative* quantities over >= 5 lists: distinct artists
  recommended, repeat-artist rate, bucket balance across lists). Until then,
  report per-list distinct-artist count only.

### E-06. Full-catalogue ranking (RecBole `full` mode, exact metrics) — BLOCKED by candidate generation, not by compute
- **Cannot:** rank the Spotify catalogue. The harness ranks a 343-item sampled
  pool; per Krichene & Rendle (A11) the sampled ordering need not match the
  exact ordering. P-10 (full-pool re-rank over the candidate pool file) narrows
  the gap but does not close it: the candidate pool is itself a selection.
- **Unblock:** a candidate pool that is a defined, frozen superset (e.g. every
  track the v2/v3 pipelines ever scored, with features cached), so that
  "full-pool" means a stable denominator rather than a fresh sample. That is a
  data-pipeline task, not a harness task.

### E-07. Streaming history / skip data as graded relevance — BLOCKED, no source
- **Cannot:** build graded relevance (e.g. NDCG with rel = 2 for kept, 1 for
  neutral, 0 for skipped) or a skip-based negative set. No streaming history
  is connected; the only judgements are keep/drop on 10-track lists and the
  binary liked/not-liked file.
- **Unblock:** Spotify streaming history (`GET /me/player/recently-played`,
  or the Extended Streaming History export) or Last.fm scrobbles. With it:
  graded NDCG (`score_type = raw`, Recommenders A9) and a skip-derived silver
  negative set become computable. Without it, binary relevance is the ceiling.

### E-08. Full-audio features in the harness — CONDITIONAL, harness already agnostic
- **Cannot (today):** use full-length audio; Kevin owns no full-length files and
  the friend's API is sign-in pending. Every port in section (b) runs on Deezer
  30 s previews (the current `librosa` features) and the harness consumes only
  `(score, relevance, kind)` rows, so full audio changes feature quality, not
  harness logic. If the API lands, the only harness change is re-running the
  exclusion audit (P-08): preview-missing items may gain features, and
  `P@10_all` vs `P@10` should converge. No metric definition changes.

### E-09. What the v2 harness already gets right (do not break it in v3)
- Same-pool paired comparison of v2 vs v6 (the only valid basis for a delta).
- Binary NDCG with log2 discount, implemented correctly for the 130-positive case.
- Reporting excluded (no-librosa) counts, even if only to stdout.
- The GTZAN/genre sanity check as a separate diagnostic from the ranking metrics.
- Seed fixed at 42 for the distractor draw (matches Recommenders' `SEED = 42`
  convention; keep 42 as the canonical seed and add 43–46 for the sweep, P-04).

---

## Appendix: constants the v3 builder must not re-derive

| Constant | Value | Derivation |
|---|---|---|
| Liked tracks | 650 | `wc -l spotify-liked-songs.txt`, 2026-11-21 |
| Holdout (position-based) | 130 | 20% of 650; `evaluate.py TEST_N` |
| Gold negatives | 13 (9 v5 + 4 v6-H) | `negatives_features.py NEGATIVES`; history file lines 41–60 |
| Distractors | 200, seed 42 | `evaluate.py N_DISTRACTORS`, `RandomState(42)` |
| Pool size | 343 | 130 + 13 + 200 |
| Random P@10 / NDCG@10 | 0.379 | 130/343 |
| Random Recall@10 | 0.029 | (10 × 130/343)/130 |
| Random mean rank | 172.0 | (343+1)/2 |
| Uniform-rank SD | 99.0 | sqrt((343²−1)/12) |
| Neg-control n for α = 0.05 | 102 | ln(0.05)/ln(1 − 10/343) = 101.2 → 102 |
| v6 baseline keep rate | 6/10 = 0.60, Wilson 95% CI [0.313, 0.832] | history file v6 lines 51–60; Wilson z = 1.96 |
| Single-list rejection of p = 0.60 | 9/10 (p = 0.046) | binomial, n = 10 |
| n for 0.60 → 0.80, 80% power | 42 judged tracks | normal approx; exact power 0.882 at critical k = 31 |
| Cumulative GO at n = 42 | k >= 32 (Wilson lower 0.615, p = 0.021) | exact binomial + Wilson |
| Cumulative GO at n = 50 | k >= 37 (Wilson lower 0.604, p = 0.028) | exact binomial + Wilson |
| Bootstrap | B = 10,000, seed 42, stratified by kind | P-04 |
| Distractor-seed sweep | 42, 43, 44, 45, 46 | P-04 |
| Tail ratio | 0.1 | RecBole `TailPercentage` default |
| Popularity-bias flag | top10 mean − pool mean > 15 points | P-05 (pool SD printed beside it) |

## Verification log (what was actually checked for this spec)
- Both repos cloned 2026-11-21 (`git clone --depth 1`, exit 0) and read as files:
  RecBole `evaluator/metrics.py`, `base_metric.py`, `evaluator.py`, `collector.py`,
  `register.py`, `sampler/sampler.py`, `properties/overall.yaml`, `significance_test.py`;
  Recommenders `evaluation/python_evaluation.py` (full, both halves),
  `datasets/python_splitters.py`, `datasets/split_utils.py`, `utils/constants.py`,
  `docs/evaluation.rst`, `examples/03_evaluate/` listing.
- RecBole eval-setting docs confirmed via web search (recbole.io v0.1.2 / v1.0.1 /
  v0.2.0 pages); Krichene & Rendle confirmed via walid.krichene.net PDF + CACM abstract.
- `evaluate.py`, `negatives_features.py`, `gtzan_check.py` (existence),
  `spotify-liked-songs.txt` (650 lines), `spotify-recommended-tracks.txt`
  (60 entries + header) read 2026-11-21.
- All binomial/Wilson/sample-size numbers recomputed in Python 2026-11-21
  (`/tmp/eval_numbers.py`, `/tmp/eval_numbers2.py`); values in the tables above
  are the script outputs, rounded to 3 decimals.
- **Could not verify:** (1) whether `data/negatives_features.json` will contain 9
  or 13 records — the file did not exist when this spec was written (v2 agent
  still building); T-01 exists for exactly this. (2) The commit hashes of the
  cloned repos (depth-1 clone; hashes not recorded). (3) Whether
  `spotify-liked-songs.txt` is newest-first — no dates in the file; the anchor
  check in P-02 is the verification step. (4) The exact Spotify-popularity
  values in the candidate pool (needed to calibrate the 15-point flag in P-05);
  the flag threshold is set as a rule with the pool SD printed beside it, not
  as a pre-computed number.
