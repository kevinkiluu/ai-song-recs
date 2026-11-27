"""Recalculate historical retrieval metrics from item records using the standard library."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from math import comb, isfinite, log2, sqrt
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[1]
K = 10
LOWER_SCORES = {'v6b', 'v6g', 'bliss_fixed', 'bliss_adaptive'}
# Median BPM, BPM interquartile range, mean RMS, mean brightness, from scoring.py.
FROZEN_V6 = {
    'global': (123.0, 152.0 - 103.0, 0.320, 2766),
    'E': (123.0, 136.0 - 112.3, 0.320, 2668),
    'R': (123.0, 161.5 - 103.4, 0.322, 2655),
    'H': (136.0, 161.5 - 117.5, 0.341, 2897),
}


def ideal_dcg(hits):
    return sum(1 / log2(rank + 1) for rank in range(1, hits + 1))


def ranking(rows, method):
    # Historical baseline score ties use the original row order.
    ordered = sorted(rows, key=lambda r: r['scores'][method], reverse=method not in LOWER_SCORES)
    hits = [rank for rank, row in enumerate(ordered[:K], 1) if row['relevant']]
    dcg = sum(1 / log2(rank + 1) for rank in hits)
    positives = sum(row['relevant'] for row in rows)
    return {
        'hit_ranks': hits,
        'precision': len(hits) / K,
        'ndcg_standard': dcg / ideal_dcg(min(K, positives)) if positives else 0,
        'ndcg_legacy': dcg / ideal_dcg(len(hits)) if hits else 0,
        'rejects_top10': [r['title'] for r in ordered[:K] if r['kind'] == 'negative'],
    }


def chance(rows):
    total = len(rows)
    positives = sum(row['relevant'] for row in rows)
    # Conditional on h hits, each of the K positions has hit probability h/K.
    expected_legacy = sum(
        (comb(positives, h) * comb(total - positives, K - h) / comb(total, K))
        * (h / K) * ideal_dcg(K) / ideal_dcg(h)
        for h in range(1, min(K, positives) + 1)
    )
    return {'pool': total, 'like_rows': positives, 'p10': positives / total,
            'ndcg_standard': positives / total * ideal_dcg(K) / ideal_dcg(min(K, positives)),
            'ndcg_legacy': expected_legacy}


def v6_distance(row, bucket):
    bpm, width, energy, brightness = FROZEN_V6.get(bucket, FROZEN_V6['global'])
    audio = row['librosa']
    return round(abs(audio['bpm'] - bpm) / width + abs(audio['rms_mean'] - energy) / 0.10
                 + abs(audio['brightness_mean'] - brightness) / 500
                 - 0.5 * ('minor' in (audio.get('key') or '')), 2)


def rebuild_v6(likes, scored):
    buckets = defaultdict(Counter)
    valid = [r for r in likes if r['spotify_id'] and r['spotify_id'] != 'None']
    for row in valid:
        if row['genre_bucket']:
            buckets[row['artist'].split(',')[0].strip().lower()][row['genre_bucket']] += 1
    heldout = sorted((r for r in valid if r['position'] < 130 and r.get('librosa')),
                     key=lambda r: r['position'])
    stored = {r['spotify_id']: r for r in scored if r['kind'] == 'heldout'}
    rebuilt = []
    matched = 0
    for row in heldout:
        bucket = row['genre_bucket']
        if bucket not in {'E', 'R', 'H'}:
            modes = buckets[row['artist'].split(',')[0].strip().lower()].most_common(1)
            bucket = modes[0][0] if modes else None
        scores = {'v6b': v6_distance(row, bucket), 'v6g': v6_distance(row, None)}
        if row['spotify_id'] in stored:
            for method, value in scores.items():
                if value != stored[row['spotify_id']]['scores'][method]:
                    raise ValueError(f'v6 disagreement for {method} on {row["spotify_id"]}')
            matched += 1
        rebuilt.append(dict(title=row['title'], scores=scores, relevant=1, kind='heldout'))
    if matched != len(stored):
        raise ValueError('historical held-out rows do not match the liked-feature snapshot')
    return rebuilt + [r for r in scored if r['kind'] != 'heldout'], matched


def coverage(likes, scored):
    heldout = [r for r in likes if r['position'] < 130]
    train = [r for r in likes if r['position'] >= 130]
    train_by_deezer = defaultdict(list)
    for row in train:
        if row['deezer_id']:
            train_by_deezer[row['deezer_id']].append(row['position'])
    twins = [dict(title=r['title'], deezer_id=r['deezer_id'], holdout=r['position'],
                  train_positions=train_by_deezer[r['deezer_id']])
             for r in heldout if r['deezer_id'] in train_by_deezer]
    pool_by_deezer = defaultdict(list)
    for row in scored:
        pool_by_deezer[row['deezer_id']].append(row['spotify_id'])
    model_tree = ast.parse((ROOT / 'model_v2.py').read_text())
    negative_ids = next(ast.literal_eval(s.value) for s in model_tree.body
                        if isinstance(s, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == 'NEGATIVE_IDS' for t in s.targets))
    observed_negatives = {r['spotify_id'] for r in scored if r['kind'] == 'negative'}
    if not observed_negatives <= set(negative_ids):
        raise ValueError('scored negatives missing from configured historical reject set')
    heldout_ids = {r['spotify_id'] for r in heldout if r.get('librosa')
                   and r['deezer_id'] not in train_by_deezer}
    if heldout_ids != {r['spotify_id'] for r in scored if r['kind'] == 'heldout'}:
        raise ValueError('feature holdout after twin removal differs from scored holdout')

    # In the pinned snapshot valid Spotify IDs are unique. Deezer IDs provide the only aliases.
    valid = [r for r in likes if r['spotify_id'] not in (None, 'None', '')]
    if len({r['spotify_id'] for r in valid}) != len(valid):
        raise ValueError('the liked-only Deezer grouping assumes distinct valid Spotify IDs')
    def recording(row):
        return ('deezer', row['deezer_id']) if row['deezer_id'] else ('spotify', row['spotify_id'])
    holdout_groups = {recording(r) for r in valid if r['position'] < 130}
    preview_fields = ('bpm', 'brightness_mean', 'rms_mean')
    eligible_holdout = {recording(r) for r in valid if r['position'] < 130 and all(
        type((r.get('librosa') or {}).get(k)) in (int, float) and isfinite(r['librosa'][k])
        for k in preview_fields)}
    train_only_groups = {recording(r) for r in valid if r['position'] >= 130} - holdout_groups
    return {
        'liked_only_v4_test_representatives': len(eligible_holdout),
        'liked_only_v4_train_representatives': len(train_only_groups),
        'liked_positions': len(likes),
        'valid_spotify_ids': len({r['spotify_id'] for r in likes if r['spotify_id'] not in (None, 'None', '')}),
        'added_at_present': sum(bool(r.get('added_at')) for r in likes),
        'deezer_ids_present': sum(bool(r.get('deezer_id')) for r in likes),
        'librosa_present': sum(bool(r.get('librosa')) for r in likes),
        'reccobeats_present': sum(bool(r.get('reccobeats')) for r in likes),
        'both_feature_families': sum(bool(r.get('librosa') and r.get('reccobeats')) for r in likes),
        'holdout_positions': len(heldout),
        'holdout_librosa': sum(bool(r.get('librosa')) for r in heldout),
        'cross_split_deezer_twins': twins,
        'duplicate_deezer_ids_in_likes': sum(v > 1 for v in Counter(
            r['deezer_id'] for r in likes if r['deezer_id']).values()),
        'duplicate_deezer_ids_in_ranked_rows': {
            k: v for k, v in pool_by_deezer.items() if len(v) > 1
        },
        'historical_negative_ids': len(negative_ids),
        'historical_negative_ids_ranked': len(observed_negatives),
        'historical_negative_ids_absent': sorted(set(negative_ids) - observed_negatives),
    }


def wilson(keeps, n):
    if n < 1 or not 0 <= keeps <= n:
        raise ValueError('counts require 0 <= keeps <= n and n > 0')
    z = NormalDist().inv_cdf(0.975)
    p = keeps / n
    factor = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / factor
    delta = z / factor * sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return [max(0, center - delta), min(1, center + delta)]


def trial_replay(path):
    """Use recording labels and per-arm scores in a local v4/v5 scored export."""
    raw = path.read_bytes()
    doc = json.loads(raw)
    split = doc['split']
    if (doc['version'] != 4 or doc['settings']['k'] != K or doc['blockers']
            or split not in {'fresh judgments', 'position-based'}):
        raise ValueError('expected a complete, scored v4-format trial at k=10')
    rows = doc['pools']['full']['rows']
    manifest = {r['row_id']: r for r in doc['manifest']}
    groups = defaultdict(list)
    for r in manifest.values():
        groups[r['recording_id']].append(r)
    expected_sources = {'likes': 'liked_snapshot', 'rejects': 'historical_judgment',
                        'catalog': 'frozen_catalog', 'fresh': 'fresh_judgment'}
    source_counts = {source: doc['source_counts'][key] for key, source in expected_sources.items()}
    if (any(type(n) is not int or n < 0 for n in source_counts.values())
            or Counter(r['source'] for r in manifest.values()) != Counter(source_counts)
            or any(r['partition'] not in {'train', 'test', 'excluded'} for r in manifest.values())):
        raise ValueError('manifest is incomplete or has an unrecognized partition')
    tested = {r['row_id'] for r in manifest.values() if r['partition'] == 'test'}
    if (not rows or len(manifest) != len(doc['manifest']) or len(tested) != len(rows)
            or tested != {r['row_id'] for r in rows}
            or len({r['recording_id'] for r in rows}) != len(rows)):
        raise ValueError('full scored pool and unique manifest test rows differ')
    methods = ('v2_refit', 'v6_frozen', 'embeat_full', 'embeat_minus_ch1')
    for r in rows:
        item = manifest[r['row_id']]
        group = groups[r['recording_id']]
        if (any(item[k] != r[k] for k in ('recording_id', 'label', 'label_source'))
                or r['label'] not in {'like', 'reject', 'unlabeled'}
                or any(g['partition'] == 'train' for g in group)
                or set(r['scores']) != set(methods) or set(r['ranks']) != set(methods)
                or any(type(v) not in (int, float) or not isfinite(v)
                       for v in [*r['scores'].values(), r['embeat_tie_score']])):
            raise ValueError('mismatched test identity, label, or arm scores')
        if split == 'fresh judgments':
            if item['source'] != 'frozen_catalog' or any(
                    g['source'] in {'liked_snapshot', 'historical_judgment'} for g in group):
                raise ValueError('fresh scored pool contains a historical or noncatalog recording')
            fresh = [g for g in group if g['source'] == 'fresh_judgment']
            if r['label'] == 'unlabeled':
                if fresh or item.get('judgment_row_id'):
                    raise ValueError('unlabeled recording has a declared judgment')
            else:
                linked = manifest[item['judgment_row_id']]
                if (linked not in fresh or linked['partition'] != 'excluded'
                        or not r['label_source'] or linked['label_source'] != r['label_source']
                        or any(g['label'] != r['label'] for g in fresh)):
                    raise ValueError('eligible label lacks a consistent linked fresh judgment')

    counts = Counter(r['label'] for r in rows)
    metrics = {}
    for method in methods:
        def order(r):
            score = r['scores'][method]
            tie = -r['embeat_tie_score'] if method.startswith('embeat') else 0
            return (score if method == 'v6_frozen' else -score, tie, r['recording_id'])
        ordered = sorted(rows, key=order)
        if any(type(r['ranks'][method]) is not int or r['ranks'][method] != i
               for i, r in enumerate(ordered, 1)):
            raise ValueError(f'stored {method} ranks disagree with recomputed scores and tie order')
        likes = [i for i, r in enumerate(ordered[:K], 1) if r['label'] == 'like']
        ideal = ideal_dcg(min(K, counts['like']))
        metrics[method] = {
            'hit_ranks': likes, 'precision': len(likes) / K,
            'ndcg_standard': sum(1 / log2(i + 1) for i in likes) / ideal if ideal else 0.0,
            'rejects_top10': sum(r['label'] == 'reject' for r in ordered[:K]),
        }
    result = {'sha256': hashlib.sha256(raw).hexdigest(), 'split': split, 'rows': len(rows),
              'declared_label_counts': counts, 'rankings': metrics}
    if split == 'fresh judgments':
        result['additional_likes_needed'] = max(0, K - counts['like'])
        result['additional_rejects_needed'] = max(0, 2 - counts['reject'])
    return result


def spotify_snapshot(path):
    """Summarize visible track rows, preserving ambiguity about original offers."""
    groups = json.loads(path.read_text())
    membership = {}
    collections = {}
    token_sets = {}
    for name, rows in groups.items():
        tokens = set()
        keeps = 0
        for row in rows:
            token = row['token']
            status = row['library_status']
            saved = status == 'SAVED'
            if not token or token in tokens or status not in {'SAVED', 'NOT_SAVED'}:
                raise ValueError(f'invalid or duplicate membership row in {name}')
            if row['in_liked_collection'] is not saved or membership.get(token, saved) != saved:
                raise ValueError(f'conflicting saved-track evidence in {name}')
            tokens.add(token)
            membership[token] = saved
            keeps += saved
        collections[name] = {'visible': len(rows), 'saved': keeps,
                             'wilson': wilson(keeps, len(rows)) if rows else None}
        token_sets[name] = tokens
    overlaps = {f'{first}/{second}': len(token_sets[first] & token_sets[second])
                for first in groups for second in groups if first < second}
    return {'collections': collections, 'distinct_tracks': len(membership), 'overlaps': overlaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keeps', nargs=2, type=int, action='append', default=[],
                        metavar=('KEPT', 'OFFERS'), help='additional Wilson interval from supplied counts')
    parser.add_argument('--spotify-snapshot', type=Path, help='private JSON of per-collection token, '
                        'library_status, and in_liked_collection rows')
    parser.add_argument('--trial', type=Path, help='private scored v4/v5 JSON for a full-pool replay')
    args = parser.parse_args()
    paths = {name: ROOT / name for name in ('features_v2.json', 'v3_scores_306.json', 'model_v2.py')}
    likes = json.loads(paths['features_v2.json'].read_bytes())['tracks']
    rows = json.loads(paths['v3_scores_306.json'].read_bytes())['items']
    methods = tuple(rows[0]['scores'])
    for r in rows:
        if int(r['kind'] == 'heldout') != r['relevant'] or set(r['scores']) != set(methods):
            raise ValueError('inconsistent item label or score columns')
        if not all(isfinite(v) for v in r['scores'].values()):
            raise ValueError('missing or nonfinite score')
    original_pool, shared = rebuild_v6(likes, rows)
    result = {
        'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        'coverage': coverage(likes, rows),
        'sampled_counts': Counter(r['kind'] for r in rows),
        'historical_rankings': {method: ranking(rows, method) for method in methods},
        'v6_original_pool': {method: ranking(original_pool, method) for method in ('v6b', 'v6g')},
        'v6_like_rows_with_exactly_reproduced_scores': shared,
        'chance': [chance(rows), chance(original_pool)],
        'wilson_intervals_given_counts': {
            f'{h}/{n}': wilson(h, n) for h, n in [(6, 10), (31, 42), (32, 42),
                                                    (36, 50), (37, 50), *args.keeps]
        },
    }
    if args.trial:
        try:
            result['supplied_trial'] = trial_replay(args.trial)
        except (ValueError, KeyError, TypeError, OSError) as error:
            parser.error(f'invalid trial input: {error}')
    if args.spotify_snapshot:
        result['spotify_visible_membership'] = spotify_snapshot(args.spotify_snapshot)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
