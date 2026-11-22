#!/usr/bin/env python3
"""Evaluation harness v2. Built BEFORE the new recommendations are picked.

Ranking task: rank a pool of (a) held-out liked tracks (test = most recent
20% by list position), (b) the 9 dropped v5 negatives, (c) 200 distractor
tracks sampled from the candidate pool (seed 42). Relevance = 1 iff the
track is a held-out like. Metrics: P@10, NDCG@10, mean rank of held-out
likes, mean score/rank of negatives. Both methods rank the SAME pool.

Methods compared:
  v2  - GMM cluster distance + learned LR log-odds (higher = better fit)
  v6b - v6 baseline, per-bucket profile (lower = closer fit)
  v6g - v6 baseline, global profile (lower = closer fit)
Only items WITH librosa features are ranked, so the comparison is
apples-to-apples; excluded counts are reported.

Sanity check (GTZAN fallback): multinomial logistic regression on the
train features predicts the genre bucket (E/R/H/O) of held-out likes.
If the GTZAN download completed, run the GTZAN check separately
(gtzan_check.py) and report both.

Run: ~/workspace/audio-env/bin/python evaluate.py
Outputs: evaluation_v2.json (metrics + per-item ranks for audit)
"""
import json, os, sys
from collections import Counter

import numpy as np

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
sys.path.insert(0, BASE)
from scoring import V2Model, feat_vector, v6_score  # noqa: E402

TEST_N = 130
N_DISTRACTORS = 200
SEED = 42


def ndcg_at_k(rel, k=10):
    rel = np.asarray(rel[:k], dtype=float)
    dcg = float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())
    ideal = np.ones(min(k, int(rel.sum())))
    idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(rel, k=10):
    return float(np.mean(np.asarray(rel[:k], dtype=float)))


def rank_metrics(items, score_key, higher_better=True):
    """items: list of dicts with 'relevant' (0/1) and score_key.
    Returns P@10, NDCG@10, mean rank of relevant items, mean score of
    negatives, mean rank of negatives."""
    order = sorted(items, key=lambda x: x[score_key],
                   reverse=higher_better)
    rel = [x['relevant'] for x in order]
    ranks = {}
    for i, x in enumerate(order, 1):
        ranks.setdefault(x['kind'], []).append(i)
    negs = [x for x in order if x['kind'] == 'negative']
    return {
        'n_ranked': len(order),
        'P@10': round(precision_at_k(rel), 3),
        'NDCG@10': round(ndcg_at_k(rel), 3),
        'mean_rank_heldout_like': round(float(np.mean(ranks.get('heldout', [np.nan]))), 1),
        'mean_score_negative': round(float(np.mean([x[score_key] for x in negs])), 3) if negs else None,
        'mean_rank_negative': round(float(np.mean(ranks.get('negative', [np.nan]))), 1) if negs else None,
        'top10_kinds': [x['kind'] for x in order[:10]],
    }


def main():
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    tracks = [t for t in doc['tracks'] if t['spotify_id'] != 'None']
    model = V2Model()
    neg_recs = json.load(open(os.path.join(DATA, 'negatives_features.json')))
    pool_path = os.path.join(DATA, 'pool_features.json')
    if not os.path.exists(pool_path):
        print('POOL FILE MISSING: run recommend_v2.py build-pool first')
        sys.exit(1)
    pool = json.load(open(pool_path))

    # artist -> most common genre bucket among liked tracks (for v6 buckets
    # on negatives/distractors whose own bucket is unknown)
    artist_bucket = {}
    for t in tracks:
        a = (t.get('artist') or '').split(',')[0].strip().lower()
        b = t.get('genre_bucket')
        if a and b:
            artist_bucket.setdefault(a, Counter())[b] += 1
    artist_bucket = {a: c.most_common(1)[0][0] for a, c in artist_bucket.items()}

    def bucket_for(rec):
        b = rec.get('genre_bucket')
        if b in ('E', 'R', 'H'):
            return b
        a = (rec.get('artist') or '').split(',')[0].strip().lower()
        return artist_bucket.get(a)

    test = [t for t in tracks if t['position'] < TEST_N]
    test_lb = [t for t in test if t.get('librosa')]
    neg_lb = [r for r in neg_recs if r.get('librosa')]
    print(f'test likes: {len(test)} ({len(test_lb)} with librosa, '
          f'{len(test) - len(test_lb)} excluded: no preview)')
    print(f'negatives: {len(neg_recs)} ({len(neg_lb)} with librosa)')
    print(f'pool tracks: {len(pool)}')

    rng = np.random.RandomState(SEED)
    pool_lb = [p for p in pool if p.get('librosa')]
    idx = rng.choice(len(pool_lb), size=min(N_DISTRACTORS, len(pool_lb)),
                     replace=False)
    distractors = [pool_lb[i] for i in sorted(idx)]
    print(f'distractors sampled: {len(distractors)} (with librosa)')

    items = []
    for t in test_lb:
        items.append({'kind': 'heldout', 'relevant': 1, 'rec': t})
    for r in neg_lb:
        items.append({'kind': 'negative', 'relevant': 0, 'rec': r})
    for p in distractors:
        items.append({'kind': 'distractor', 'relevant': 0, 'rec': p})

    # ---- v2 scores
    X, _ = zip(*[feat_vector(x['rec']) for x in items])
    s = model.score_records([x['rec'] for x in items])
    for x, sc in zip(items, s['final']):
        x['v2'] = round(float(sc), 3)
    # ---- v6 scores
    for x in items:
        sc_b, _ = v6_score(x['rec'].get('librosa'), bucket_for(x['rec']))
        sc_g, _ = v6_score(x['rec'].get('librosa'), None)
        x['v6b'] = sc_b
        x['v6g'] = sc_g
    n_nov6 = sum(1 for x in items if x['v6b'] is None)
    print(f'items without v6 score: {n_nov6}')

    results = {}
    results['v2'] = rank_metrics([x for x in items], 'v2', higher_better=True)
    results['v6_per_bucket'] = rank_metrics([x for x in items if x['v6b'] is not None],
                                            'v6b', higher_better=False)
    results['v6_global'] = rank_metrics([x for x in items if x['v6g'] is not None],
                                        'v6g', higher_better=False)
    print(json.dumps(results, indent=1))

    # ---- genre sanity check: predict E/R/H/O bucket from features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    train = [t for t in tracks if t['position'] >= TEST_N and t.get('genre_bucket')]
    Xtr, _ = zip(*[feat_vector(t) for t in train])
    Xtr = np.vstack(Xtr)
    ytr = np.array([t['genre_bucket'] for t in train])
    test_g = [t for t in test_lb if t.get('genre_bucket')]
    Xte, _ = zip(*[feat_vector(t) for t in test_g])
    Xte = np.vstack(Xte)
    yte = np.array([t['genre_bucket'] for t in test_g])
    med = np.nanmedian(Xtr, axis=0)
    Xtr_i = np.where(np.isnan(Xtr), med, Xtr)
    Xte_i = np.where(np.isnan(Xte), med, Xte)
    sc = StandardScaler().fit(Xtr_i)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(Xtr_i), ytr)
    acc = float(clf.score(sc.transform(Xte_i), yte))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, clf.predict(sc.transform(Xte_i)),
                          labels=['E', 'R', 'H', 'O']).tolist()
    print(f'genre sanity: accuracy on held-out likes = {acc:.3f}')
    print('confusion rows=true E/R/H/O, cols=pred:', cm)
    results['genre_sanity'] = {'accuracy': round(acc, 3),
                               'labels': ['E', 'R', 'H', 'O'],
                               'confusion': cm,
                               'n_train': len(train), 'n_test': len(test_g),
                               'note': 'Fallback sanity check: predicts '
                                       'Kevin genre-bucket labels from audio '
                                       'features. GTZAN check is separate.'}

    results['counts'] = {
        'test_likes': len(test), 'test_with_librosa': len(test_lb),
        'negatives': len(neg_recs), 'negatives_with_librosa': len(neg_lb),
        'distractors': len(distractors),
        'pool_total': len(pool),
        'v6_missing_scores': n_nov6,
    }
    json.dump(results, open(os.path.join(BASE, 'evaluation_v2.json'), 'w'), indent=1)
    # per-item audit trail
    audit = [{'kind': x['kind'], 'title': x['rec'].get('title'),
              'artist': x['rec'].get('artist'),
              'spotify_id': x['rec'].get('spotify_id'),
              'deezer_id': x['rec'].get('deezer_id'),
              'v2': x['v2'], 'v6b': x['v6b'], 'v6g': x['v6g']}
             for x in items]
    json.dump(audit, open(os.path.join(DATA, 'evaluation_audit.json'), 'w'), indent=1)
    print('wrote evaluation_v2.json + data/evaluation_audit.json')


if __name__ == '__main__':
    main()
