"""Measure recording leakage and its effect on a fixed candidate panel across three folds."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_v4 import build_manifest, evaluation_pool, identity_tokens
from integrate_v4 import _enrich_records, _read_records, fit_and_score_pool
from scoring import feat_vector


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('v2_refit', 'embeat_full', 'embeat_minus_ch1')


def identity_coverage(records):
    counts = Counter(token.partition(':')[0] for record in records for token in identity_tokens(record))
    return {'rows': len(records),
            **{field: counts[field] for field in ('spotify', 'deezer', 'isrc', 'sha256')}}


def compact_metrics(pool, training):
    with threadpool_limits(limits=1):
        result, _ = fit_and_score_pool(pool, training)
    return {arm: {'p_at_10': result['metrics'][arm]['p_at_k'],
                  'ndcg_at_10': result['metrics'][arm]['ndcg_at_k'],
                  'rejects_at_10': result['metrics'][arm]['negative_top_k']} for arm in ARMS}


def measure_fold(likes, rejects, catalog, *, start, end, reject_fold):
    """The before fit restores only older training aliases of eligible held-out likes."""
    missing = [role for role, rows in [('rejects', rejects), ('catalog', catalog)] if rows is None]
    manifest = build_manifest(likes, rejects or [], catalog or [], holdout_start=start,
                              cutoff=end, reject_fold=reject_fold)
    pool = evaluation_pool(manifest)
    test_components = {r['recording_id'] for r in pool}
    heldout = {r['recording_id'] for r in pool if r['label'] == 'like'}
    old_likes, recent_likes, train = [], [], {'like': [], 'reject': []}
    reasons = Counter()
    active_train = set()
    components_by_token = {}
    training_like_rows = set()
    for row in manifest:
        record = row['record']
        group = row['recording_id']
        for token in identity_tokens(record):
            if components_by_token.setdefault(token, group) != group:
                raise ValueError('same provider identity assigned to different components')
        if row['source'] == 'liked_snapshot' and type(record.get('position')) is int:
            if record['position'] >= end:
                old_likes.append(row)
            elif start <= record['position']:
                recent_likes.append(row)
        if row['partition'] == 'train':
            if np.isfinite(feat_vector(record)[0]).any():
                train[row['label']].append(record)
                if row['label'] == 'like':
                    training_like_rows.add(row['row_id'])
                if group in active_train:
                    raise ValueError('duplicate training recording')
                active_train.add(group)
            else:
                reasons['missing training features'] += 1
        elif row['partition'] == 'excluded':
            reasons[row['reason']] += 1

    overlap = active_train & test_components
    duplicates = len(pool) - len(test_components)
    if overlap or duplicates:
        raise ValueError('recording components overlap or repeat in evaluation')
    naive_overlap = ({r['recording_id'] for r in old_likes}
                     & {r['recording_id'] for r in recent_likes})
    older_twins = [r for r in old_likes if r['recording_id'] in naive_overlap]
    twins_to_fit = [r for r in older_twins if r['recording_id'] in heldout
                    and np.isfinite(feat_vector(r['record'])[0]).any()]
    eligibility = [(r['recording_id'], r['label']) for r in pool]
    summary = {
        'position_range': [start, end], 'reject_fold': reject_fold, 'missing_inputs': missing,
        'position_split': {'older_like_rows': len(old_likes), 'holdout_like_rows': len(recent_likes),
                           'overlapping_liked_components': len(naive_overlap),
                           'older_alias_rows': len(older_twins)},
        'recording_split': {'train_likes_with_features': len(train['like']),
                            'train_rejects_with_features': len(train['reject']),
                            'eval_rows': len(pool), 'eligible_likes': len(heldout),
                            'identity_coverage': {
                                'training': identity_coverage(train['like'] + train['reject']),
                                'evaluation': identity_coverage([r['record'] for r in pool])},
                            'exclusions_by_reason': dict(sorted(reasons.items())),
                            'cross_partition_components': len(overlap),
                            'duplicate_eval_components': duplicates,
                            'panel_sha256': hashlib.sha256(json.dumps(eligibility).encode()).hexdigest()},
        'paired_fit': None,
    }
    if missing:
        return summary
    if len(train['like']) < 3 or not train['reject'] or not heldout or not catalog:
        summary['fit_blocker'] = 'Need a catalog, three usable train likes, a train reject and a held-out like'
        return summary
    after = compact_metrics(pool, train)
    if twins_to_fit:
        # Preserve the original position ordering so the sole intervention is
        # the membership of these training rows. The panel/labels stay fixed.
        before_rows = training_like_rows | {row['row_id'] for row in twins_to_fit}
        before = compact_metrics(pool, {'like': [r['record'] for r in manifest if r['row_id'] in before_rows],
                                        'reject': train['reject']})
    else:
        before = after
    summary['paired_fit'] = {'reintroduced_training_rows': len(twins_to_fit),
                             'before': before, 'after': after,
                             'after_minus_before': {arm: {key: after[arm][key] - before[arm][key]
                                                          for key in after[arm]} for arm in ARMS}}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--likes', type=Path, default=ROOT / 'features_v2.json')
    parser.add_argument('--rejects', type=Path, default=ROOT / 'data/negatives_features.json')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'data/pool_features.json')
    parser.add_argument('--matches', type=Path, help='Frozen Spotify-to-Deezer match cache')
    parser.add_argument('--previews', type=Path, help='Authorized original preview cache')
    args = parser.parse_args(argv)
    if args.previews and not args.previews.is_dir():
        parser.error('specified preview directory is unavailable')
    matches = {}
    digests = {}
    if args.matches:
        data = args.matches.read_bytes()
        digests['matches'] = hashlib.sha256(data).hexdigest()
        matches = json.loads(data)
    coverage = {}
    sources = {}
    for role in ('likes', 'rejects', 'catalog'):
        path = getattr(args, role)
        if role != 'likes' and not path.is_file():
            sources[role] = None
            continue
        data = path.read_bytes()
        digests[role] = hashlib.sha256(data).hexdigest()
        raw = _read_records(data, role, 'tracks' if role == 'likes' else None)
        records = _enrich_records(raw, matches, args.previews)
        coverage[role] = {'raw': identity_coverage(raw), 'enriched': identity_coverage(records)}
        sources[role] = records
    with redirect_stdout(sys.stderr):
        folds = [measure_fold(**sources, start=130*i, end=130*(i+1), reject_fold=i)
                 for i in range(3)]
    print(json.dumps({'input_sha256': digests, 'identity_coverage': coverage, 'folds': folds},
                     indent=2, allow_nan=False))
    return 0 if all(f['paired_fit'] is not None for f in folds) else 2


if __name__ == '__main__':
    raise SystemExit(main())
