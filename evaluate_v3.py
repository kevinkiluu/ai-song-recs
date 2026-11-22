#!/usr/bin/env python3
"""Eval harness v3 (NEW module — evaluate.py is untouched).

Implements research/eval-spec.md §(b) ports P-01..P-11 and the §(d)
pre-registered 4-leg GO verdict, as pure functions over
(score, relevance, kind) rows. No model code, no network, no Spotify /
Deezer calls. All counts are recomputed from the actual inputs on every
call — nothing about the pool is hardcoded.

GROUNDING (verified from the landed v2 files; overrides the spec's
stale-pool assumptions — see audit_real_pool() and test_evaluate_v3.py
header G1..G8):
  * Ranked pool is 311, not 343: 99 held-out likes WITH librosa
    (130 by position, 31 excluded incl. the unresolvable 'None' id at
    position 85), 12 negatives WITH librosa (13 defined; 'Touch' by
    KATSEYE has no librosa), 200 distractors, seed 42.
  * Random baseline is therefore 99/311 = 0.318, random mean rank 156.0.
  * The split is POSITION-based, not time-based: no liked track carries
    added_at (0/650). P-02 labels it so everywhere and refuses any
    time-based split (TIME_SPLIT_REFUSED) until added_at coverage >= 95%.
  * NDCG@10 ideal DCG follows the v2 convention (ideal built from the
    hits inside the top-k window), NOT RecBole's min(|R|, K) ideal:
    the spec's own T-07 hand values (e.g. rel=[1,0,1,...] -> 0.920)
    require the v2 convention, and keeping it preserves comparability
    with evaluation_v2.json. This is stated in METRIC_DEFINITIONS.

Run:  ~/workspace/audio-env/bin/python evaluate_v3.py
      (prints the real-data pool audit + grounded baselines as JSON)
Tests: ~/workspace/audio-env/bin/python test_evaluate_v3.py
"""
import json, math, os, sys
from functools import lru_cache

import numpy as np

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')

SPLIT_LABEL_POSITION = 'position-based holdout (NOT time-based)'
SPLIT_LABEL_TIME = 'time-based holdout (added_at)'
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42
SEED_SWEEP = (42, 43, 44, 45, 46)
POPULARITY_FLAG_GAP = 15
NEGATIVE_HALF_WIDTH_UNDERPOWERED = 25.0
V6_KEEP_RATE_P0 = 0.60
LIVE_N_GO = 42
LIVE_K_GO_42 = 32

METRIC_DEFINITIONS = {
    'P@10': {'formula': 'hits in top 10 / 10',
             'note': 'headline offline metric; direct analogue of the '
                     '10-track keep rate'},
    'NDCG@10': {'discount': 'log2',
                'formula': 'DCG / IDCG, DCG = sum(rel_i / log2(i+2))',
                'ideal': 'IDCG built from ones(min(k, hits inside top-k)) '
                         '— the v2 convention (evaluate.py ndcg_at_k), '
                         'NOT min(k, n_relevant). Kept so v3 NDCG stays '
                         'comparable with evaluation_v2.json; the spec T-07 '
                         'hand values require this convention.',
                'note': 'Recommenders defaults to loge; ours is log2 and '
                        'the two are not interchangeable.'},
    'MAP@10': {'normalisation': 'min(k, n_relevant)',
               'formula': 'sum over hits of precision@rank / min(k, n_relevant)',
               'note': 'Recommenders map_at_k convention. NOT RecBole MAP '
                       '(min(|R-hat|, K)) and NOT Recommenders map '
                       '(normalised by n_relevant).'},
    'MRR@10': {'formula': '1 / rank of first held-out like, else 0',
               'note': 'RecBole default valid_metric.'},
    'Recall@10': {'formula': 'hits in top 10 / n_relevant',
                  'note': 'reported, never a gate: max 10/130 compresses '
                          'real differences into the third decimal.'},
    'Hit@10': {'formula': '1 if any held-out like in top 10 else 0',
               'note': 'reported only; saturated at this pool size.'},
}


# ---------------------------------------------------------------- metrics
def metrics_from_rel(rel, n_relevant, k=10):
    """Hand-checkable metrics from a top-k relevance vector (0/1).

    n_relevant = number of relevant items in the ranked pool (e.g. 99
    held-out likes with librosa on the grounded v2 pool)."""
    rel = [int(x) for x in list(rel)[:k]]
    n_hits = int(sum(rel))
    p10 = n_hits / k
    dcg = float(sum(r / math.log2(i + 2) for i, r in enumerate(rel)))
    ideal_n = min(k, n_hits)
    idcg = float(sum(1.0 / math.log2(i + 2) for i in range(ideal_n)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    mrr = 1.0 / first if first else 0.0
    ap_num = 0.0
    hits_so_far = 0
    for i, r in enumerate(rel):
        if r:
            hits_so_far += 1
            ap_num += hits_so_far / (i + 1)
    denom = min(k, int(n_relevant)) if n_relevant else 0
    map10 = ap_num / denom if denom > 0 else 0.0
    defined = n_hits > 0
    return {
        'P@10': round(p10, 3),
        'NDCG@10': round(ndcg, 3),
        'MAP@10': round(map10, 3),
        'MRR@10': round(mrr, 3),
        'Recall@10': round(n_hits / n_relevant, 3) if n_relevant else 0.0,
        'Hit@10': 1 if n_hits else 0,
        'n_hits': n_hits,
        'defined': defined,
        'warning': (None if defined else
                    'ZERO_HITS: metric is 0.0 with n_hits=0 — a bare 0.0 '
                    'is never reported without this flag (P-09).'),
    }


def rank_items(items, score_key, higher_better=True):
    """Deterministic ranking (P-07).

    Sort key: score desc (or asc when lower is better), spotify_id asc as
    the deterministic tie-breaker. Each item copy gains 'position'
    (1..n in sorted order) and 'rank' (average rank within an exact-score
    tie group, RecBole Collector._average_rank style). Items with a None
    score are dropped and counted in the caller, never ranked."""
    scored = [dict(x) for x in items if x.get(score_key) is not None]
    if higher_better:
        scored.sort(key=lambda x: (-float(x[score_key]),
                                   str(x.get('spotify_id', ''))))
    else:
        scored.sort(key=lambda x: (float(x[score_key]),
                                   str(x.get('spotify_id', ''))))
    i = 0
    n = len(scored)
    while i < n:
        j = i
        while (j + 1 < n
               and float(scored[j + 1][score_key]) == float(scored[i][score_key])):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # positions i+1..j+1, 1-based
        for t in range(i, j + 1):
            scored[t]['position'] = t + 1
            scored[t]['rank'] = avg_rank
        i = j + 1
    return scored


def rank_metrics(items, score_key, higher_better=True, k=10,
                 n_relevant=None):
    """Full metric dict for one model's ranking of one pool (P-01/P-03/
    P-07/P-09). 'items' carry 'relevant' (0/1), 'kind', 'spotify_id' and
    the score column; 'bucket' is used for the H-negative sub-count."""
    ranked = rank_items(items, score_key, higher_better)
    n_no_score = sum(1 for x in items if x.get(score_key) is None)
    if n_relevant is None:
        n_relevant = sum(1 for x in ranked if x.get('relevant'))
    rel = [x.get('relevant', 0) for x in ranked]
    m = metrics_from_rel(rel, n_relevant, k=k)
    topk = ranked[:k]
    neg_top = [x for x in topk if x.get('kind') == 'negative']
    ranks = {}
    for x in ranked:
        ranks.setdefault(x.get('kind'), []).append(x['rank'])
    n_tied = 0
    seen_scores = {}
    for x in ranked:
        seen_scores.setdefault(float(x[score_key]), 0)
        seen_scores[float(x[score_key])] += 1
    n_tied = sum(c for c in seen_scores.values() if c > 1)
    n_heldout = sum(1 for x in ranked if x.get('kind') == 'heldout')
    rb = random_baseline(n_heldout, len(ranked), k=k)
    m.update({
        'n_ranked': len(ranked),
        'n_no_score_excluded': n_no_score,
        'n_tied_scores': n_tied,
        'mean_rank_heldout_like': (round(float(np.mean(ranks['heldout'])), 1)
                                   if 'heldout' in ranks else None),
        'mean_rank_negative': (round(float(np.mean(ranks['negative'])), 1)
                               if 'negative' in ranks else None),
        'mean_score_negative': (round(float(np.mean(
            [float(x[score_key]) for x in ranked
             if x.get('kind') == 'negative'])), 3)
            if 'negative' in ranks else None),
        'n_neg_top10': len(neg_top),
        'n_neg_top10_H': sum(1 for x in neg_top if x.get('bucket') == 'H'),
        'top10_kinds': [x.get('kind') for x in topk],
        'top10_ids': [x.get('spotify_id') for x in topk],
        'distinct_artists_top10': len({str(x.get('artist', '')) for x in topk}),
        'random_P@10': round(rb['random_P@10'], 3),
        'random_NDCG@10': round(rb['random_NDCG@10'], 3),
        'below_random': bool(m['P@10'] <= rb['random_P@10']),
    })
    return m


# ---------------------------------------------------------------- P-01
@lru_cache(maxsize=32)
def _random_ndcg_mc(n_heldout_with_features, n_ranked, reps=2000, seed=42):
    """Monte-Carlo expectation of NDCG@10 under a uniform random ranking,
    using the v2 NDCG convention (ideal from hits inside the top-k).
    Cached: the value depends only on the pool shape."""
    rng = np.random.default_rng(seed)
    base = np.array([1] * n_heldout_with_features
                    + [0] * (n_ranked - n_heldout_with_features))
    vals = []
    for _ in range(reps):
        rel = rng.permutation(base)[:10]
        vals.append(_ndcg_rel(rel))
    return float(np.mean(vals))


def random_baseline(n_heldout_with_features, n_ranked, k=10):
    """Chance floor of the *actual* ranked pool (P-01). Never hardcoded:
    random P@10 = n_heldout / n_ranked.

    DELTA G9 (spec contradiction, resolved in favour of T-07 + the
    landed evaluate.py): the spec's P-01 claims random_NDCG@10 equals
    random_P@10, but that equality holds only under the STANDARD NDCG
    ideal (min(k, n_relevant) ones). The v2/T-07 ideal is built from the
    hits inside the top-k window, under which a random ranking scores
    NDCG@10 ~ 0.63 on the 311 pool — far above 0.318 — because any
    ranking whose top-10 hits sit above its misses scores 1.0. So
    random_NDCG@10 here is the Monte-Carlo expectation under the v2
    convention (seed 42, 2000 reps), NOT random_P@10. Consequence: v2's
    NDCG@10 0.726 sits only ~0.09 above the chance floor under this
    convention, while P@10 0.50 vs 0.318 looks stronger. The NDCG
    margin is the weaker evidence; the GO gate's Leg 1 still uses the
    bootstrap CI on the NDCG *difference*, which is unaffected."""
    p = n_heldout_with_features / n_ranked if n_ranked else 0.0
    sd = math.sqrt((n_ranked ** 2 - 1) / 12.0) if n_ranked else 0.0
    return {
        'random_P@10': p,
        'random_NDCG@10': _random_ndcg_mc(n_heldout_with_features, n_ranked),
        'random_NDCG@10_method': ('Monte-Carlo expectation under the v2 '
                                  'NDCG convention, seed 42, 2000 reps'),
        'random_Recall@10': (k * p) / n_heldout_with_features
                            if n_heldout_with_features else 0.0,
        'random_hits_top10': k * p,
        'random_mean_rank': (n_ranked + 1) / 2.0,
        'uniform_rank_sd': sd,
        'n_heldout_with_features': n_heldout_with_features,
        'n_ranked': n_ranked,
    }


# ---------------------------------------------------------------- P-02
def split_label(split):
    if split == 'position-based':
        return SPLIT_LABEL_POSITION
    if split == 'time-based':
        return SPLIT_LABEL_TIME
    raise ValueError(f'unknown split: {split!r}')


def assert_split_allowed(split, added_at_coverage, total_liked):
    """Hard date guard (P-02, Recommenders python_chrono_split precedent):
    a time-based split without >= 95% added_at coverage exits non-zero
    and writes nothing. Position-based never raises."""
    if split == 'time-based':
        if added_at_coverage < 0.95 * total_liked:
            raise SystemExit(
                f'TIME_SPLIT_REFUSED: added_at coverage = '
                f'{added_at_coverage}/{total_liked} (< 95%, i.e. '
                f'< {math.ceil(0.95 * total_liked)} of {total_liked}). '
                f'No chrono split without timestamps; use the '
                f'position-based holdout and label it as such.')
    elif split != 'position-based':
        raise ValueError(f'unknown split: {split!r}')
    return split_label(split)


# ---------------------------------------------------------------- P-04
def _ndcg_rel(rel):
    rel = np.asarray(rel, dtype=float)
    if len(rel) == 0 or rel.sum() == 0:
        return 0.0
    dcg = float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())
    ideal = np.ones(min(len(rel), int(rel.sum())))
    idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def paired_bootstrap(items, score_a_key, score_b_key,
                     higher_a=True, higher_b=False,
                     B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, k=10):
    """Paired bootstrap CI on the A-minus-B difference (P-04).

    Resampling unit is the ranked pool item (n=1 user makes a t-test over
    users undefined), stratified by kind so each resample keeps the
    99/12/200 counts. B=10,000, seed 42 by default."""
    rel = np.array([1 if x.get('relevant') else 0 for x in items], dtype=float)
    kinds = [x.get('kind') for x in items]
    sa = np.array([float(x[score_a_key]) for x in items], dtype=float)
    sb = np.array([float(x[score_b_key]) for x in items], dtype=float)
    ka = sa if higher_a else -sa
    kb = sb if higher_b else -sb
    kind_values = sorted(set(kinds))
    kind_idx = {kv: np.array([i for i, kk in enumerate(kinds) if kk == kv])
                for kv in kind_values}
    rng = np.random.default_rng(seed)
    d_p, d_n, p_a = [], [], []
    for _ in range(B):
        idx = np.concatenate([
            rng.choice(kind_idx[kv], size=len(kind_idx[kv]), replace=True)
            for kv in kind_values])
        order_a = idx[np.argsort(-ka[idx], kind='stable')]
        order_b = idx[np.argsort(-kb[idx], kind='stable')]
        ra, rb_ = rel[order_a][:k], rel[order_b][:k]
        p_a.append(float(ra.mean()))
        d_p.append(float(ra.mean() - rb_.mean()))
        d_n.append(_ndcg_rel(ra) - _ndcg_rel(rb_))
    d_p, d_n, p_a = map(np.array, (d_p, d_n, p_a))
    lo, hi = 2.5, 97.5
    return {
        'delta_p10': round(float(np.mean(rel[np.argsort(-ka, kind="stable")][:k])
                                 - np.mean(rel[np.argsort(-kb, kind="stable")][:k])), 3),
        'delta_ndcg': round(float(_ndcg_rel(rel[np.argsort(-ka, kind="stable")][:k])
                                  - _ndcg_rel(rel[np.argsort(-kb, kind="stable")][:k])), 3),
        'delta_p10_ci': [round(float(np.percentile(d_p, lo)), 3),
                         round(float(np.percentile(d_p, hi)), 3)],
        'delta_ndcg_ci': [round(float(np.percentile(d_n, lo)), 3),
                          round(float(np.percentile(d_n, hi)), 3)],
        'p10_a': round(float(np.mean(p_a)), 3),
        'p10_a_ci': [round(float(np.percentile(p_a, lo)), 3),
                     round(float(np.percentile(p_a, hi)), 3)],
        'B': B, 'seed': seed, 'k': k,
        'resampling_unit': 'pool item, stratified by kind',
    }


def draw_distractors(candidates, n, seed):
    """Draw n distractors the v2 way: RandomState(seed), no replacement,
    returned in sorted-draw order (P-04 seed sweep)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(candidates), size=min(n, len(candidates)),
                     replace=False)
    return [candidates[i] for i in sorted(idx)]


def seed_sweep_summary(per_seed):
    """Summarise a 5-seed distractor sweep (P-04, spec T-05).

    per_seed: list of dicts with keys p10_a, p10_b, ndcg_a, ndcg_b
    (one per distractor seed, same models, same pool shape). The spec's
    T-05 asserts the per-model P@10 range across seeds (<= 0.20) and
    that sign(delta_NDCG@10) agrees in >= 4 of 5 seeds."""
    pa = [float(s['p10_a']) for s in per_seed]
    pb = [float(s['p10_b']) for s in per_seed]
    dp = [a - b for a, b in zip(pa, pb)]
    dn = [float(s['ndcg_a']) - float(s['ndcg_b']) for s in per_seed]
    pos = sum(1 for x in dn if x > 0)
    neg = sum(1 for x in dn if x < 0)
    return {
        'n_seeds': len(per_seed),
        'seeds': list(SEED_SWEEP[:len(per_seed)]),
        'p10_a_range': round(max(pa) - min(pa), 3) if pa else None,
        'p10_b_range': round(max(pb) - min(pb), 3) if pb else None,
        'p10_range': round(max(max(pa) - min(pa), max(pb) - min(pb)), 3)
                     if pa else None,
        'p10_a_min': round(min(pa), 3) if pa else None,
        'p10_a_median': round(float(np.median(pa)), 3) if pa else None,
        'p10_a_max': round(max(pa), 3) if pa else None,
        'p10_b_min': round(min(pb), 3) if pb else None,
        'p10_b_median': round(float(np.median(pb)), 3) if pb else None,
        'p10_b_max': round(max(pb), 3) if pb else None,
        'delta_p10_median': round(float(np.median(dp)), 3) if dp else None,
        'ndcg_min': round(min(dn), 3) if dn else None,
        'ndcg_median': round(float(np.median(dn)), 3) if dn else None,
        'ndcg_max': round(max(dn), 3) if dn else None,
        'ndcg_sign_agreement': max(pos, neg),
    }


# ---------------------------------------------------------------- P-05
def popularity_audit(items, score_key, higher_better=True, threshold=POPULARITY_FLAG_GAP):
    """AveragePopularity + TailPercentage with a Spotify-popularity proxy
    (P-05). Train-count popularity is a constant at |U|=1, so Spotify
    popularity (0-100) substitutes; if no popularity values exist the
    metric is skipped, never zero-filled."""
    ranked = rank_items(items, score_key, higher_better)
    pops = [float(x['popularity']) for x in ranked
            if x.get('popularity') is not None]
    if not pops:
        return {'popularity_source': 'unavailable', 'skipped': True,
                'popularity_bias': None, 'threshold': threshold,
                'note': 'no popularity values in pool; metric skipped.'}
    top10 = ranked[:10]
    top_pops = [float(x['popularity']) for x in top10
                if x.get('popularity') is not None]
    pool_mean = float(np.mean(pops))
    pool_sd = float(np.std(pops))
    top_mean = float(np.mean(top_pops)) if top_pops else float('nan')
    tail_cut = float(np.percentile(pops, 10))
    tail_share = (sum(1 for p in top_pops if p <= tail_cut) / len(top_pops)
                  if top_pops else 0.0)
    gap = top_mean - pool_mean
    return {
        'popularity_source': 'spotify_popularity_0_100',
        'skipped': False,
        'mean_popularity_top10': round(top_mean, 1),
        'mean_popularity_pool': round(pool_mean, 1),
        'pool_popularity_sd': round(pool_sd, 1),
        'gap': round(gap, 1),
        'threshold': threshold,
        'popularity_bias': bool(gap > threshold),
        'tail_share_top10': round(tail_share, 3),
        'tail_ratio': 0.1,
        'tail_cutoff_popularity': round(tail_cut, 1),
    }


# ---------------------------------------------------------------- P-06
def check_leakage(top10_ids, train_ids, heldout_ids, negative_ids,
                  distractor_ids):
    """Fail-fast leakage guards (P-06). Raises AssertionError on any
    violation; returns the (empty) violation list otherwise."""
    violations = []
    top10 = set(top10_ids)
    if top10 & set(train_ids):
        violations.append(f'top10 ∩ train = {sorted(top10 & set(train_ids))}')
    if set(negative_ids) & set(heldout_ids):
        violations.append('negatives ∩ heldout_likes is non-empty')
    if set(negative_ids) & set(train_ids):
        violations.append('negatives ∩ train_likes is non-empty')
    if set(distractor_ids) & set(heldout_ids):
        violations.append('distractors ∩ heldout_likes is non-empty')
    if set(distractor_ids) & set(negative_ids):
        violations.append('distractors ∩ negatives is non-empty')
    if violations:
        raise AssertionError('LEAKAGE: ' + '; '.join(violations))
    return violations


def artist_overlap_fraction(train_recs, test_recs):
    """Report-only artist-leakage audit (P-06 §4): share of test tracks
    whose artist also appears in train. > 0.50 prints ARTIST_LEAKAGE_WARNING."""
    train_artists = {str(r.get('artist', '')).split(',')[0].strip().lower()
                     for r in train_recs if r.get('artist')}
    test = [r for r in test_recs if r.get('artist')]
    if not test:
        return {'artist_overlap_fraction': None, 'warning': None}
    frac = sum(1 for r in test
               if str(r.get('artist', '')).split(',')[0].strip().lower()
               in train_artists) / len(test)
    return {
        'artist_overlap_fraction': round(frac, 3),
        'warning': ('ARTIST_LEAKAGE_WARNING: P@10 is partly artist '
                    'memorisation; the artist-disjoint split (P-11) is the '
                    'binding sensitivity check.' if frac > 0.50 else None),
    }


# ---------------------------------------------------------------- P-08
def missing_feature_audit(n_heldout_total, excluded_ids, ranked_metrics,
                          n_heldout_ranked):
    """Missing-feature exclusion audit (P-08).

    NOTE (spec delta G6): the spec's literal worst-case imputation —
    excluded held-out likes counted as relevance 0 and ranked last —
    leaves the top 10 unchanged, so P@10_all == P@10 by construction and
    the > 0.10 warning can never fire from that difference alone. The
    implementation follows the spec literally and states the degeneracy
    in the note instead of inventing a different imputation; the honest
    signal here is the excluded count/fraction and the audit list."""
    excluded = list(excluded_ids)
    p10 = ranked_metrics['P@10']
    p10_all = p10  # excluded ranked last cannot enter the top 10
    return {
        'n_heldout_total': n_heldout_total,
        'n_heldout_with_librosa': n_heldout_ranked,
        'n_heldout_excluded': len(excluded),
        'excluded_fraction': (round(len(excluded) / n_heldout_total, 3)
                              if n_heldout_total else 0.0),
        'excluded_spotify_ids': excluded,
        'P@10': p10,
        'P@10_all': p10_all,
        'missing_feature_warning': bool(abs(p10 - p10_all) > 0.10),
        'imputation_note': ('Worst-case imputation is degenerate: excluded '
                            'held-out likes counted as relevance 0 and '
                            'ranked last cannot enter the top 10, so '
                            'P@10_all == P@10 and the top 10 is unchanged. '
                            'Judge the exclusion by n_heldout_excluded and '
                            'excluded_fraction, not by the P@10 gap.'),
    }


# ---------------------------------------------------------------- P-06/T-06 power
def negative_control_stats(n_negatives_ranked, n_ranked, k=10,
                           n_negatives_defined=None):
    """Negative-control power (T-06): chance of seeing zero negatives in
    the top 10, n required for alpha=0.05, and the mean-rank half-width.
    All recomputed from the actual ranked counts."""
    p_top = k / n_ranked if n_ranked else 0.0
    p_zero = (1.0 - p_top) ** n_negatives_ranked
    n_req = math.ceil(math.log(0.05) / math.log(1.0 - p_top)) if 0 < p_top < 1 else None
    sd = math.sqrt((n_ranked ** 2 - 1) / 12.0) if n_ranked else 0.0
    half = 1.96 * sd / math.sqrt(n_negatives_ranked) if n_negatives_ranked else None
    return {
        'n_negatives_ranked': n_negatives_ranked,
        'n_negatives_defined': (n_negatives_defined if n_negatives_defined is not None
                                else n_negatives_ranked),
        'n_ranked': n_ranked,
        'p_zero_neg_in_top10_random': round(p_zero, 3),
        'n_required_for_alpha_0.05': n_req,
        'uniform_rank_sd': round(sd, 1),
        'mean_rank_half_width': round(half, 1) if half is not None else None,
        'power_label': ('UNDERPOWERED' if half is not None and half > 25.0
                        else 'POWERED'),
    }


# ---------------------------------------------------------------- P-10/P-11
def sampled_metric_disagreement(delta_sampled, delta_full):
    """True when the sampled-pool delta and the full-pool delta have
    opposite signs (P-10). A zero delta on either side is not a
    disagreement — it is a tie."""
    return bool((delta_sampled > 0 and delta_full < 0)
                or (delta_sampled < 0 and delta_full > 0))


def artist_disjoint_split(records, target_test_n):
    """Artist-disjoint split (P-11): greedy whole-artist assignment to
    test (largest artists first, name-ascending tie-break) until the
    test size reaches target_test_n. Returns (train_ids, test_ids)."""
    from collections import Counter
    counts = Counter(str(r.get('artist', '')) for r in records)
    artists = sorted(counts, key=lambda a: (-counts[a], a))
    test_artists, test_n = set(), 0
    for a in artists:
        if test_n >= target_test_n:
            break
        test_artists.add(a)
        test_n += counts[a]
    test_ids = {r['spotify_id'] for r in records
                if str(r.get('artist', '')) in test_artists}
    train_ids = {r['spotify_id'] for r in records
                 if str(r.get('artist', '')) not in test_artists}
    return train_ids, test_ids


# ---------------------------------------------------------------- §(d) stats
def wilson_ci(k, n, z=1.96):
    """Wilson score interval (95% by default). Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    m = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def binom_p_ge(k, n, p0=V6_KEEP_RATE_P0):
    """Exact one-sided binomial P(X >= k | n, p0)."""
    return sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
               for i in range(k, n + 1))


# ---------------------------------------------------------------- §(d) verdict
def verdict_4leg(m):
    """Pre-registered 4-leg GO verdict (spec §d). Input dict keys:

      delta_ndcg_ci_low, delta_ndcg, delta_p10, delta_map10,
      p10_v3, p10_v3_ci_low, random_p10,
      n_neg_top10, n_neg_top10_H, mean_rank_negative_v3, random_mean_rank,
      sampled_disagreement (bool),
      live_keeps, live_n (None if no live list judged yet),
      live_per_list_keeps (optional list of keeps per judged 10-list).

    Returns JSON with each leg's pass/fail, the numbers, and the
    verdict-table outcome. The 4-leg rule governs the ship decision and
    is the authority over any per-feature bar."""
    leg1 = bool(
        m['delta_ndcg_ci_low'] > 0.0
        and (m['delta_p10'] >= 0.10
             or (m['delta_p10'] == 0.0 and m.get('delta_map10', 0.0) >= 0.02)))
    leg2 = bool(m['p10_v3'] >= 0.50
                and m['p10_v3_ci_low'] > m['random_p10'])
    leg3 = bool(m['n_neg_top10'] == 0 and m['n_neg_top10_H'] == 0
                and m['mean_rank_negative_v3'] > m['random_mean_rank'])
    agreement = not bool(m.get('sampled_disagreement', False))
    legs = {
        'leg1_offline_vs_v6': {
            'passed': leg1,
            'delta_ndcg': m['delta_ndcg'],
            'delta_ndcg_ci_low': m['delta_ndcg_ci_low'],
            'delta_p10': m['delta_p10'],
            'delta_map10': m.get('delta_map10'),
            'rule': 'bootstrap 95% CI low on delta_NDCG@10 > 0 AND '
                    'delta_P@10 >= +0.10 (or delta_P@10 == 0 with '
                    'delta_MAP@10 >= +0.02)'},
        'leg2_vs_chance': {
            'passed': leg2,
            'p10_v3': m['p10_v3'],
            'p10_v3_ci_low': m['p10_v3_ci_low'],
            'random_p10': m['random_p10'],
            'rule': 'P@10(v3) >= 0.50 AND bootstrap 95% low of P@10(v3) '
                    '> random baseline'},
        'leg3_negative_control': {
            'passed': leg3,
            'n_neg_top10': m['n_neg_top10'],
            'n_neg_top10_H': m['n_neg_top10_H'],
            'mean_rank_negative_v3': m['mean_rank_negative_v3'],
            'random_mean_rank': m['random_mean_rank'],
            'power_label': 'UNDERPOWERED (n=12 ranked negatives; '
                           'half-width ~50.8 ranks — the leg passes on '
                           'point conditions, the label records that the '
                           'control cannot prove much)',
            'rule': '0 gold negatives in top 10 (incl. 0 H-bucket) AND '
                    'mean_rank_negative(v3) > random mean rank'},
        'sampled_metric_agreement': {
            'passed': agreement,
            'sampled_disagreement': bool(m.get('sampled_disagreement', False)),
            'rule': 'sampled-pool and full-pool deltas must not disagree '
                    'in sign (P-10)'},
    }
    live_n = m.get('live_n')
    live_keeps = m.get('live_keeps')
    per_list = m.get('live_per_list_keeps') or []
    consec_le7 = (len(per_list) >= 2 and per_list[-1] <= 7
                  and per_list[-2] <= 7)
    if live_n is None or live_keeps is None:
        leg4 = {'passed': None, 'status': 'pending',
                'rule': f'cumulative keeps k >= 32 of n = 42 (or k >= 37 '
                        f'of 50); Wilson 95% lower bound > 0.60 AND exact '
                        f'binomial p <= 0.05 vs p0 = 0.60'}
        verdict = ('PENDING — live leg not yet judged (n=0/42); GO requires '
                   'Leg 4 at n = 42. A single 9/10 or 10/10 list is not a GO.')
    else:
        lo, _hi = wilson_ci(live_keeps, live_n)
        p_val = binom_p_ge(live_keeps, live_n)
        marginal = bool(p_val <= 0.05 and lo <= 0.60)
        leg4_pass = bool(lo > 0.60 and p_val <= 0.05 and not consec_le7)
        leg4 = {'passed': leg4_pass, 'status': 'judged',
                'keeps': live_keeps, 'n': live_n,
                'keep_rate': round(live_keeps / live_n, 3) if live_n else None,
                'wilson_lower': round(lo, 3),
                'binom_p_vs_0.60': round(p_val, 3),
                'marginal_split_decision': marginal,
                'two_consecutive_lists_le7': consec_le7,
                'rule': 'k >= 32/42 (Wilson lower 0.615, p = 0.021) or '
                        'k >= 37/50; two consecutive lists <= 7/10 trigger '
                        'a NO-GO review'}
        if consec_le7:
            verdict = ('NO-GO — offline signal did not transfer to '
                       'listening (two consecutive lists <= 7/10)')
        elif leg4_pass:
            verdict = 'GO — v3 beats v6'
        elif marginal:
            verdict = ('NO-GO (marginal) — binomial passes, Wilson fails; '
                       'do not ship on a split decision')
        else:
            verdict = ('NO-GO — offline signal did not transfer to listening')
    legs['leg4_live_keep_rate'] = leg4
    offline_pass = leg1 and leg2 and leg3
    if not agreement:
        verdict = ('NO-GO — unresolved measurement disagreement '
                   '(SAMPLED_METRIC_DISAGREEMENT); investigate before '
                   're-deciding')
    elif not offline_pass and live_n is None:
        verdict = 'NO-GO — offline'
    elif not offline_pass and live_n is not None:
        # Legs 1-3 fail: live leg should not have been run; offline rules.
        verdict = 'NO-GO — offline'
    return {'verdict': verdict, 'legs': legs,
            'offline_pass': offline_pass,
            'note': '4-leg rule is pre-registered (spec §d) and governs '
                    'the ship decision over any per-feature bar.'}


# ---------------------------------------------------------------- real-data audit (T-01)
def audit_real_pool():
    """Re-ground the pool against the landed v2 files (P-01/P-06/P-08).

    Replicates evaluate.py's distractor draw (RandomState(42), 200 of the
    librosa-bearing pool) so the audit checks the same draw v2 ranked.
    Returns the counts, the excluded-negative identity, disjointness
    booleans, and the honest split label."""
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    tracks = [t for t in doc['tracks']]
    test = [t for t in tracks if t.get('position') is not None
            and t['position'] < 130]
    test_nodrop = [t for t in test if t['spotify_id'] != 'None']
    test_lb = [t for t in test_nodrop if t.get('librosa')]
    train = [t for t in tracks if t.get('position') is not None
             and t['position'] >= 130 and t['spotify_id'] != 'None']
    neg_recs = json.load(open(os.path.join(DATA, 'negatives_features.json')))
    neg_lb = [r for r in neg_recs if r.get('librosa')]
    pool = json.load(open(os.path.join(DATA, 'pool_features.json')))
    pool_lb = [p for p in pool if p.get('librosa')]
    rng = np.random.RandomState(42)
    idx = rng.choice(len(pool_lb), size=min(200, len(pool_lb)), replace=False)
    distractors = [pool_lb[i] for i in sorted(idx)]
    heldout_ids = {t['spotify_id'] for t in test_lb}
    train_ids = {t['spotify_id'] for t in train}
    neg_ids = {r['spotify_id'] for r in neg_recs}
    neg_lb_ids = {r['spotify_id'] for r in neg_lb}
    distr_ids = {p['spotify_id'] for p in distractors}
    added_at = sum(1 for t in tracks if t.get('added_at'))
    return {
        'test_likes_total': len(test),
        'test_likes_after_none_drop': len(test_nodrop),
        'test_with_librosa': len(test_lb),
        'test_excluded_ids': [t['spotify_id'] for t in test
                              if not t.get('librosa')],
        'negatives_defined': len(neg_recs),
        'negatives_with_librosa': len(neg_lb),
        'negatives_excluded_ids': [r['spotify_id'] for r in neg_recs
                                   if not r.get('librosa')],
        'negatives_excluded_titles': [f"{r['title']} - {r['artist']}"
                                      for r in neg_recs
                                      if not r.get('librosa')],
        'distractors': len(distractors),
        'pool_total': len(pool),
        'n_ranked': len(test_lb) + len(neg_lb) + len(distractors),
        'negatives_disjoint_heldout': bool(not (neg_ids & heldout_ids)),
        'negatives_disjoint_train': bool(not (neg_ids & train_ids)),
        'distractors_disjoint': bool(not (distr_ids & heldout_ids)
                                     and not (distr_ids & neg_lb_ids)),
        'split_label': SPLIT_LABEL_POSITION,
        'added_at_coverage': added_at,
        'total_liked': len(tracks),
        'random_baseline': random_baseline(len(test_lb),
                                           len(test_lb) + len(neg_lb)
                                           + len(distractors)),
        'negative_control': negative_control_stats(len(neg_lb),
                                                   len(test_lb) + len(neg_lb)
                                                   + len(distractors),
                                                   n_negatives_defined=len(neg_recs)),
    }


def main():
    audit = audit_real_pool()
    print(json.dumps(audit, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
