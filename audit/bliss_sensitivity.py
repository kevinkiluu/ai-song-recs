"""Offline Bliss pool sensitivity. Run from the checkout with python -m audit.bliss_sensitivity."""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

import bliss_features as bliss
from evaluate_v3 import SEED_SWEEP, draw_distractors, sampled_metric_disagreement
from evaluate_v4 import build_manifest, identity_tokens, ranking_metrics
from model_v2 import NEGATIVE_IDS
from scoring import V2Model, v6_score

METHODS = ('v6b', 'bliss_fixed', 'bliss_adaptive', 'bliss_v2_blend')
SCORE_COLUMNS = ('v2', *METHODS)
LABEL = {'heldout': 'like', 'negative': 'reject', 'distractor': 'unlabeled'}
RAW_INPUTS = ('features_v2.json', 'model_v2.json', 'model_v2.npz', 'data/bliss23.json',
              'data/negatives_features.json', 'data/pool_features.json')
# Input bytes at the audited revision 03483da. The other four original raw
# files have no independently available digest in this checkout.
PINNED_SHA256 = {
    'v3_scores_306.json': '65b4ee2c90fe9b451c7d714f17e35d9b4a091a7c8d00fd318d76413cbff16092',
    'features_v2.json': 'd8db0af163807d9c40fa8fcc2903ac70db059cd1492efa639e92e7d7a4488ed1',
    'model_v2.json': '29ad3e9bc2baf1acf874d2a972e841e65f8764f2696b060d78639724d1aec23e',
}
FIT_REJECTS = {f'spotify:{sid}' for sid in NEGATIVE_IDS}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def measured(rows, *, historical_ties=False):
    """Each sorted list uses the same fixed scores and a panel-wide IDCG."""
    result = {}
    for method in METHODS:
        def order(row):
            score = row['scores'][method]
            return (-score if method == 'bliss_v2_blend' else score,
                    '' if historical_ties else row['recording_id'])
        ordered = sorted(rows, key=order)
        metrics = ranking_metrics(ordered, k=10)
        result[method] = dict(p_at_10=metrics['p_at_k'], ndcg_at_10=metrics['ndcg_at_k'],
                              top10=[r['recording_id'] for r in ordered[:10]])
    return result


def changes(new, old):
    return dict(entrants=[rid for rid in new if rid not in old],
                displaced=[rid for rid in old if rid not in new])


def sign(value):
    return (value > 0) - (value < 0)


def compare(rows):
    metrics = measured(rows)
    baseline = metrics['v6b']
    differences = {}
    for method in METHODS[1:]:
        arm = metrics[method]
        delta_p = round(arm['p_at_10'] - baseline['p_at_10'], 10)
        delta_ndcg = arm['ndcg_at_10'] - baseline['ndcg_at_10']
        differences[method] = dict(p_at_10=delta_p, ndcg_at_10=delta_ndcg,
                                   p_sign=sign(delta_p), ndcg_sign=sign(delta_ndcg),
                                   **changes(arm['top10'], baseline['top10']))
    return dict(members=[r['recording_id'] for r in rows],
                counts=Counter(r['label'] for r in rows), metrics=metrics, vs_v6b=differences)


def sensitivity(rows, size, *, scope):
    if size < 0:
        raise ValueError('sample size must be nonnegative')
    if not rows or not any(r['label'] == 'like' for r in rows):
        raise ValueError('need at least one eligible held-out like')
    for row in rows:
        if not all(type(v) in (int, float) and math.isfinite(v) for v in row['scores'].values()):
            raise ValueError(f'invalid score for {row["recording_id"]}')
    reference = compare(rows)
    judged = [r for r in rows if r['label'] != 'unlabeled']
    distractors = [r for r in rows if r['label'] == 'unlabeled']
    samples = []
    reference_members = frozenset(reference['members'])
    memberships = set()
    changed_seeds = []
    opposite = {method: {m: [] for m in ('p_at_10', 'ndcg_at_10')} for method in METHODS[1:]}
    for seed in SEED_SWEEP:
        sample = compare(judged + draw_distractors(distractors, size, seed))
        sample['seed'] = seed
        members = frozenset(sample['members'])
        memberships.add(members)
        sample['membership_matches_reference'] = members == reference_members
        if members != reference_members:
            changed_seeds.append(seed)
        sample['top10_changes_from_reference'] = {
            method: changes(sample['metrics'][method]['top10'], reference['metrics'][method]['top10'])
            for method in METHODS}
        for method in opposite:
            for metric, seeds in opposite[method].items():
                if sampled_metric_disagreement(sample['vs_v6b'][method][metric],
                                               reference['vs_v6b'][method][metric]):
                    seeds.append(seed)
        samples.append(sample)
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode()
    draw_membership = {
        'status': 'tested' if changed_seeds else 'not_testable',
        'distinct_sample_memberships': len(memberships),
        'distinct_sample_memberships_other_than_reference': len(memberships - {reference_members}),
        'seeds_with_membership_different_from_reference': changed_seeds,
        'eligible_distractors': len(distractors),
    }
    return dict(scope=scope, sample_distractors=size, seeds=list(SEED_SWEEP),
                draw_membership=draw_membership,
                tie_rule='ascending recording_id', sample_rule='RandomState without replacement in frozen catalog order',
                score_panel=rows, score_panel_sha256=sha256(encoded), reference=reference, samples=samples,
                opposite_sign_seeds_vs_reference=opposite if changed_seeds else None,
                full_pool_reversal=opposite if changed_seeds and scope == 'full_eligible_supplied_pool' else None)


def historical_fit(linked):
    return dict(inferential=False, configured_reject_ids_in_panel=sorted(linked),
                note='The original frozen v2 fit used the historical judged rejects. '
                     'Ranked rejects shared with its configured fit set are shown as recording IDs.')


def archived(path, size=None):
    raw = path.read_bytes()
    source = json.loads(raw)['items']
    by_recording = {}
    historical = []
    linked = set()
    for row in source:
        tokens = identity_tokens(row)
        recording = next((t for t in tokens if t.startswith('deezer:')), None)
        if not recording or row['relevant'] != int(row['kind'] == 'heldout'):
            raise ValueError('invalid archive identity or relevance')
        entry = dict(recording_id=recording, label=LABEL[row['kind']],
                     scores={key: row['scores'][key] for key in SCORE_COLUMNS})
        historical.append(entry)
        if FIT_REJECTS.intersection(tokens):
            linked.add(recording)
        if by_recording.setdefault(recording, entry) != entry:
            raise ValueError(f'conflicting archive labels or scores for {recording}')
    rows = list(by_recording.values())
    if size is None:
        size = sum(r['label'] == 'unlabeled' for r in rows) // 2
    result = sensitivity(rows, size, scope='archived_sample_only')
    digest = sha256(raw)
    result.update(input_sha256={path.name: digest}, historical_fit=historical_fit(linked),
                  matches_pinned_input={'v3_scores_306.json': digest == PINNED_SHA256['v3_scores_306.json']},
                  archived_rows=dict(count=len(source), tie_rule='archived source order',
                                     metrics=measured(historical, historical_ties=True)),
                  eligibility=dict(removed_alias_rows=len(source) - len(rows)))
    return result


def preview_vectors(cache):
    vectors = {}
    for row in cache.values():
        if row.get('source') != 'preview' or row.get('bliss23') is None:
            continue
        tokens = identity_tokens({'deezer_id': row.get('deezer_id')})
        if not tokens:
            continue
        vector = np.asarray(row['bliss23'], dtype=float)
        if vector.shape != (bliss.BLISS_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f'invalid Bliss vector for {tokens[0]}')
        if tokens[0] in vectors and not np.array_equal(vectors[tokens[0]], vector):
            raise ValueError(f'conflicting preview vectors for {tokens[0]}')
        vectors[tokens[0]] = vector
    return vectors


def raw_panel(likes, rejects, catalog, vectors, model):
    """Use the v3 weights and v6 bucket convention on one common eligible panel."""
    liked_source_count = len(likes)
    likes = [r for r in likes if r.get('spotify_id') not in (None, '', 'None')]
    buckets = {}

    def artist(row):
        return (row.get('artist') or '').split(',')[0].strip().lower()

    for row in likes:
        if artist(row) and row.get('genre_bucket'):
            buckets.setdefault(artist(row), Counter())[row['genre_bucket']] += 1
    modal = {name: b.most_common(1)[0][0] for name, b in buckets.items()}

    def vector_key(row):
        return next((t for t in identity_tokens(row) if t.startswith('deezer:')), None)

    # Keep identity links for unavailable rows, so another alias cannot bring a training
    # recording into the pool. Allow an alias with shared coverage to become representative.
    def prepare(rows):
        return [dict(r, librosa=r.get('librosa') if vector_key(r) in vectors else None) for r in rows]

    manifest = build_manifest(prepare(likes), prepare(rejects), prepare(catalog), reject_folds=1)
    # The frozen v2 model includes historical train twins even when the newer
    # recording-level manifest assigns that component to test.
    train_ids = {r['recording_id'] for r in manifest if r['source'] == 'liked_snapshot'
                 and type(r['record'].get('position')) is int and r['record']['position'] >= 130}
    seed_ids = {r['recording_id'] for r in manifest if r['partition'] == 'train'
                and r['source'] == 'liked_snapshot'}
    seeds = {}
    pool = []
    excluded = Counter()
    for row in manifest:
        rid, rec = row['recording_id'], row['record']
        key = vector_key(rec)
        if row['source'] == 'liked_snapshot' and rid in seed_ids and key in vectors:
            seeds.setdefault(rid, vectors[key])
        if row['partition'] == 'test' and rid not in train_ids:
            pool.append(row)
        elif row['partition'] == 'test':
            excluded['overlap with historical liked training rows'] += 1
        elif row['reason'] == 'missing shared ranking features' and key not in vectors:
            excluded['missing preview Bliss vector'] += 1
        else:
            excluded[row['reason'] or 'training row'] += 1
    if len(seeds) < 2 or not pool:
        raise ValueError('need at least two distinct train-only Bliss seeds and a nonempty eligible pool')
    matrix = np.array(list(seeds.values()))
    adaptive = bliss.compute_adaptive_weights(matrix)
    predictions = model.score_records([r['record'] for r in pool])['final']
    panel = []
    for row, prediction in zip(pool, predictions, strict=True):
        record = row['record']
        vec = vectors[vector_key(record)]
        bucket = record.get('genre_bucket')
        if bucket not in ('E', 'R', 'H'):
            bucket = modal.get(artist(record))
        panel.append(dict(recording_id=row['recording_id'], label=row['label'],
                          scores=dict(v2=round(float(prediction), 3),
                                      v6b=v6_score(record['librosa'], bucket)[0],
                                      bliss_fixed=bliss.bliss_score_to_liked(vec, matrix, bliss.VERSION2_WEIGHTS, k=10),
                                      bliss_adaptive=bliss.bliss_score_to_liked(vec, matrix, adaptive, k=10))))

    def z(values):
        values = np.array(values)
        std = values.std()
        if not std or not np.isfinite(std):
            raise ValueError('equal-weight blend requires two finite, varying score columns')
        return (values - values.mean()) / std

    blend = (z([r['scores']['v2'] for r in panel]) + z([-r['scores']['bliss_fixed'] for r in panel])) / 2
    for row, value in zip(panel, blend):
        row['scores']['bliss_v2_blend'] = float(value)
    known_fit = {r['recording_id'] for r in manifest if FIT_REJECTS.intersection(r['identities'])}
    linked = known_fit.intersection(row['recording_id'] for row in pool)
    catalog_status = 'eligible'
    if not catalog:
        catalog_status = 'empty_input'
    elif not any(row['label'] == 'unlabeled' for row in pool):
        catalog_status = 'no_eligible_distractors'
    return panel, linked, dict(bliss_seed_recordings=len(seeds), excluded=dict(excluded),
                               catalog_status=catalog_status,
                       input_counts=dict(likes=liked_source_count, valid_spotify_likes=len(likes),
                                         rejects=len(rejects), catalog=len(catalog)))


def full(root, size=200):
    missing = [name for name in RAW_INPUTS if not (root / name).is_file()]
    if missing:
        raise ValueError('missing raw inputs: ' + ', '.join(missing))
    raw = {name: (root / name).read_bytes() for name in RAW_INPUTS}
    with np.load(io.BytesIO(raw['model_v2.npz']), allow_pickle=False) as weights:
        model = V2Model(dict(weights), json.loads(raw['model_v2.json']))
    panel, linked, eligibility = raw_panel(json.loads(raw['features_v2.json'])['tracks'],
                                   json.loads(raw['data/negatives_features.json']),
                                   json.loads(raw['data/pool_features.json']),
                                   preview_vectors(json.loads(raw['data/bliss23.json'])), model)
    result = sensitivity(panel, size, scope='full_eligible_supplied_pool')
    digests = {name: sha256(b) for name, b in raw.items()}
    result.update(eligibility=eligibility, input_sha256=digests, historical_fit=historical_fit(linked),
                  matches_pinned_input={k: digests[k] == PINNED_SHA256[k] for k in digests if k in PINNED_SHA256})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--archive', type=Path, help='historical v3 score archive')
    mode.add_argument('--raw-root', type=Path, help='original catalog, preview features and fitted model directory')
    parser.add_argument('--sample-size', type=int, help='distractors per draw (full 200, archive half)')
    parser.add_argument('--output', type=Path, help='new JSON output path (default stdout)')
    args = parser.parse_args()
    try:
        if args.raw_root is not None:
            result = full(args.raw_root, 200 if args.sample_size is None else args.sample_size)
        else:
            result = archived(args.archive, args.sample_size)
        output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output is None:
            print(output)
        else:
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(output + '\n')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
