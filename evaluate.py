#!/usr/bin/env python3
"""Compare a fold-local v2 fit with the historical frozen v6 profiles.

Assign recordings to training and evaluation by connected provider IDs, ISRC
and available preview digests. Hold out liked positions 0..129 and one of
three reject folds. Sample up to 200 eligible unrated catalog recordings
with seed 42. Only the observed shared identities can be checked.

Run: python evaluate.py
Outputs: data/evaluation_v2_refit.json and data/evaluation_audit.json
"""
import json, os, sys
from pathlib import Path

import numpy as np

BASE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(BASE, 'data')
sys.path.insert(0, BASE)
from scoring import V2Model, feat_vector, v6_score  # noqa: E402

TEST_N = 130
N_DISTRACTORS = 200
SEED = 42


def ndcg_at_k(rel, k=10, *, n_relevant=None):
    """Binary NDCG. A truncated vector must supply the total relevant count."""
    if k <= 0:
        raise ValueError('k must be positive')
    values = np.asarray(rel, dtype=float)
    total = int(values.sum()) if n_relevant is None else n_relevant
    top = values[:k]
    discounts = 1 / np.log2(np.arange(2, max(len(top), min(k, total)) + 2))
    ideal = discounts[:min(k, total)].sum()
    return float(top @ discounts[:len(top)] / ideal) if ideal else 0.0


def legacy_topk_hit_ndcg(rel, k=10):
    """Historical v2/v3 metric, conditional on the hits already in the list."""
    return ndcg_at_k(rel[:k], k)


def precision_at_k(rel, k=10):
    if k <= 0:
        raise ValueError('k must be positive')
    return float(sum(rel[:k])) / k


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
        'legacy_topk_hit_ndcg': round(legacy_topk_hit_ndcg(rel), 3),
        'mean_rank_heldout_like': round(float(np.mean(ranks['heldout'])), 1) if ranks.get('heldout') else None,
        'mean_score_negative': round(float(np.mean([x[score_key] for x in negs])), 3) if negs else None,
        'mean_rank_negative': round(float(np.mean(ranks.get('negative', [np.nan]))), 1) if negs else None,
        'top10_kinds': [x['kind'] for x in order[:10]],
    }


def genre_sanity(train, test, median):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler

    train = [row for row in train if row.get('genre_bucket')]
    test = [row for row in test if row.get('genre_bucket')]
    labels = ['E', 'R', 'H', 'O']
    ytr = [row['genre_bucket'] for row in train]
    yte = [row['genre_bucket'] for row in test]
    result = dict(labels=labels, n_train=len(train), n_test=len(test))
    if len(set(ytr)) < 2 or not test:
        return dict(result, accuracy=None, confusion=None,
                    note='Requires two labeled training genres and labeled test rows.')

    def matrix(rows):
        values = np.vstack([feat_vector(row)[0] for row in rows])
        return np.where(np.isnan(values), median, values)

    xtr = matrix(train)
    scale = StandardScaler().fit(xtr)
    model = LogisticRegression(max_iter=2000).fit(scale.transform(xtr), ytr)
    predicted = model.predict(scale.transform(matrix(test)))
    return dict(result, accuracy=round(float(np.mean(predicted == yte)), 3),
                confusion=confusion_matrix(yte, predicted, labels=labels).tolist(),
                note='Predicts genre-bucket labels from training-fold audio features.')


def write_replay(base, stem, payload):
    """Write recalculated historical metrics to the checkout's replay directory."""
    destination = Path(base) / 'data' / f'{stem}_replay.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(payload, indent=1, allow_nan=False) + '\n', encoding='utf-8')


def main():
    from evaluate_v4 import build_manifest, evaluation_pool, sample_pool
    from integrate_v4 import _enrich_records
    from embeat_port import artist_bucket_map, normalize_artist
    from model_v2 import fit_records

    base, data = Path(BASE), Path(DATA)
    tracks = json.loads((base / 'features_v2.json').read_text())['tracks']
    negatives = json.loads((data / 'negatives_features.json').read_text())
    catalog = json.loads((data / 'pool_features.json').read_text())
    matches_path = data / 'deezer_matches.json'
    matches = json.loads(matches_path.read_text()) if matches_path.exists() else {}
    previews = base / 'previews'
    tracks, negatives, catalog = (
        _enrich_records(records, matches, previews if previews.is_dir() else None)
        for records in (tracks, negatives, catalog))
    manifest = build_manifest(tracks, negatives, catalog, cutoff=TEST_N)
    train = {'like': [], 'reject': []}
    for row in manifest:
        if row['partition'] == 'train' and np.isfinite(feat_vector(row['record'])[0]).any():
            train[row['label']].append(row['record'])
    config, artifacts = fit_records(train['like'], train['reject'])
    model = V2Model(artifacts, config)
    buckets = artist_bucket_map(train['like'])
    kinds = {'like': 'heldout', 'reject': 'negative', 'unlabeled': 'distractor'}
    sample = sample_pool(evaluation_pool(manifest), N_DISTRACTORS, SEED)
    items = [{'kind': kinds[row['label']], 'relevant': int(row['label'] == 'like'),
              'rec': row['record'], 'recording_id': row['recording_id']} for row in sample]
    test = [r for r in tracks if type(r.get('position')) is int and 0 <= r['position'] < TEST_N]
    test_lb = [x['rec'] for x in items if x['kind'] == 'heldout']
    neg_lb = [x for x in items if x['kind'] == 'negative']
    distractors = [x for x in items if x['kind'] == 'distractor']

    # ---- v2 scores
    s = model.score_records([x['rec'] for x in items])
    for x, sc in zip(items, s['final']):
        x['v2'] = float(sc)
    # ---- v6 scores
    for x in items:
        artist = normalize_artist(x['rec'].get('artist'))
        sc_b, _ = v6_score(x['rec'].get('librosa'), buckets.get(artist))
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

    results['training_counts'] = {label: len(rows) for label, rows in train.items()}
    results['genre_sanity'] = genre_sanity(train['like'], test_lb, model.median)

    results['counts'] = {
        'test_likes': len(test), 'test_with_librosa': len(test_lb),
        'negatives': len(negatives), 'test_rejects_ranked': len(neg_lb),
        'distractors': len(distractors),
        'pool_total': len(catalog),
        'v6_missing_scores': n_nov6,
    }
    results['scope'] = ('Position-based recording-disjoint refit. Frozen v6 profile '
                        'training provenance is unavailable.')
    (data / 'evaluation_v2_refit.json').write_text(json.dumps(results, indent=1) + '\n')
    # per-item audit trail
    audit = [{'kind': x['kind'], 'recording_id': x['recording_id'], 'title': x['rec'].get('title'),
              'artist': x['rec'].get('artist'),
              'spotify_id': x['rec'].get('spotify_id'),
              'deezer_id': x['rec'].get('deezer_id'),
              'v2': x['v2'], 'v6b': x['v6b'], 'v6g': x['v6g']}
             for x in items]
    (data / 'evaluation_audit.json').write_text(json.dumps(audit, indent=1) + '\n')
    print('wrote data/evaluation_v2_refit.json + data/evaluation_audit.json')


if __name__ == '__main__':
    main()
