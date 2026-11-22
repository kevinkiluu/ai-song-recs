#!/usr/bin/env python3
"""Run the offline v4 comparison against a frozen candidate snapshot."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import embeat_port as embeat
from evaluate_v4 import (SEEDS, build_manifest, evaluation_pool, live_decision,
                         ranking_metrics, sample_pool)
from model_v2 import fit_records
from scoring import V2Model, feat_vector, v6_score

BASE_SHA = '13604951419ad199cf8639f2e811f5b99fdad3d9'
METHODS = ('v2_refit', 'v6_frozen', 'embeat_full', 'embeat_minus_ch1')
TRIAL = 'embeat_minus_ch1'
K = 10


def _score_pool(pool, model, seed, buckets):
    records = [dict(row['record'], _evaluation_recording_id=row['recording_id'])
               for row in pool]
    # Snapshot taste buckets come from Kevin's playlists. Give held-out
    # tracks and unlabeled catalog rows the same train-derived bucket lookup.
    for record in records:
        record.pop('genre_bucket', None)
    index = embeat.build_index(records, model, artist_buckets=buckets)
    channels = {
        'similar': embeat.recall_similar(index, seed),
        'popular': embeat.recall_popular(index, seed),
        'same_artist': embeat.recall_same_artist(index, seed),
        'related_artist': embeat.recall_related_artist(index, seed),
    }
    full = {row['key']: row['score_raw'] for row in embeat._merge_scores(channels)}
    without = {name: rows for name, rows in channels.items() if name != 'similar'}
    ablated = {row['key']: row['score_raw'] for row in embeat._merge_scores(without)}
    cosine = index['embeddings'] @ embeat.embed_records(model, [seed])[0]
    rows = []
    for row, record, relatedness in zip(pool, index['records'], cosine):
        key = embeat.cand_key(record)
        v6 = v6_score(record['librosa'], buckets.get(embeat.normalize_artist(record.get('artist'))))[0]
        scores = dict(v2_refit=record['v2_raw'], v6_frozen=v6,
                      embeat_full=full.get(key, 0.0),
                      embeat_minus_ch1=ablated.get(key, 0.0))
        if not all(math.isfinite(score) for score in scores.values()):
            raise ValueError(f'nonfinite score for {row["recording_id"]}')
        rows.append(dict(recording_id=row['recording_id'], row_id=row['row_id'],
                         label=row['label'], label_source=row['label_source'],
                         scores=scores, embeat_tie_score=float(relatedness), ranks={}))
    ranked = _rank_pool(rows)
    ranked['channel_coverage'] = {name: len(result) for name, result in channels.items()}
    return ranked


def _rank_pool(rows):
    rows = [dict(row, ranks={}) for row in rows]
    metrics = {}
    for method in METHODS:
        def ranking_key(row):
            score = row['scores'][method]
            tie = -row['embeat_tie_score'] if method.startswith('embeat_') else 0.0
            return (score if method == 'v6_frozen' else -score, tie, row['recording_id'])

        ordered = sorted(rows, key=ranking_key)
        for rank, row in enumerate(ordered, 1):
            row['ranks'][method] = rank
        metrics[method] = ranking_metrics(ordered, k=K)
    counts = Counter(row['label'] for row in rows)
    return dict(rows=rows, metrics=metrics, counts=dict(counts),
                chance_p_at_k=counts['like'] * min(K, len(rows)) / (K * len(rows)),
                chance_mean_rank=(len(rows) + 1) / 2)


def _deltas(pool):
    candidate = pool['metrics'][TRIAL]
    return {method: {metric: candidate[metric] - pool['metrics'][method][metric]
                     for metric in ('p_at_k', 'ndcg_at_k', 'map_at_k')}
            for method in METHODS if method != TRIAL}


def offline_decision(pools, *, fresh):
    # A corrected frozen-pool comparison can qualify for a blind round.
    # Promotion still requires the existing prospective live gate.
    if not pools or not fresh:
        return 'INSUFFICIENT_EVIDENCE'
    full = pools['full']
    if full['counts'].get('like', 0) < K or full['counts'].get('reject', 0) < 2:
        return 'INSUFFICIENT_EVIDENCE'
    candidate = full['metrics'][TRIAL]
    ablation = full['deltas']['embeat_full']
    if ablation['ndcg_at_k'] <= 0 or ablation['p_at_k'] < 0:
        return 'NO_GO_OFFLINE'
    for baseline in ('v2_refit', 'v6_frozen'):
        delta = full['deltas'][baseline]
        if delta['ndcg_at_k'] <= 0 or (
                delta['p_at_k'] < 0.1 - 1e-12 and
                not (delta['p_at_k'] == 0 and delta['map_at_k'] >= .02 - 1e-12)):
            return 'NO_GO_OFFLINE'
        if any(delta['p_at_k'] * pool['deltas'][baseline]['p_at_k'] < 0
               or delta['ndcg_at_k'] * pool['deltas'][baseline]['ndcg_at_k'] < 0
               for name, pool in pools.items() if name != 'full'):
            return 'NO_GO_OFFLINE'
    if (candidate['p_at_k'] < .5 or candidate['p_at_k'] <= full['chance_p_at_k']
            or candidate['negative_top_k']
            or candidate['mean_negative_rank'] <= full['chance_mean_rank']):
        return 'NO_GO_OFFLINE'
    return 'ELIGIBLE_FOR_BLIND'


def run_experiment(likes, rejects, catalog, *, fresh=None, holdout_start=0,
                   cutoff=130, reject_fold=0, sample_size=200, live=None):
    if sample_size < 0:
        raise ValueError('sample size must be nonnegative')
    manifest = build_manifest(likes, rejects, catalog, cutoff=cutoff,
                              holdout_start=holdout_start, reject_fold=reject_fold,
                              fresh=fresh)
    pool = evaluation_pool(manifest)
    train = {'like': [], 'reject': []}
    for row in manifest:
        if row['partition'] == 'train':
            record = row['record']
            if not np.isfinite(feat_vector(record)[0]).any():
                row.update(partition='excluded', reason='missing training features')
            else:
                train[row['label']].append(record)
    audit_rows = []
    for row in manifest:
        record = row['record']
        metadata = {name: record.get(name) for name in
                    ('spotify_id', 'deezer_id', 'isrc', 'preview_sha256',
                     'position', 'added_at', 'title', 'artist', 'audio_source')}
        audit_rows.append({key: value for key, value in row.items() if key != 'record'}
                           | metadata)

    pools, blocked = {}, []
    if len(train['like']) < 3 or not train['reject']:
        blocked.append('Need at least three training likes and one disjoint training reject')
    if not any(row['label'] == 'like' for row in pool):
        blocked.append('No eligible known likes in the evaluation pool')
    seed_id = None
    if not blocked:
        with threadpool_limits(limits=1):
            config, artifacts = fit_records(train['like'], train['reject'])
            model = V2Model(artifacts, config)
            buckets = embeat.artist_bucket_map(train['like'])
            # Recompute the medoid using only the fold's training likes.
            embeddings = embeat.embed_records(model, train['like'])
            centroid = embeddings.mean(axis=0)
            seed_index = int(np.argmax(embeddings @ centroid))
            seed = train['like'][seed_index]
            seed_id = next(row['recording_id'] for row in manifest
                           if row['partition'] == 'train' and row['record'] is seed)
            pools['full'] = _score_pool(pool, model, seed, buckets)
            for draw in SEEDS:
                sample = sample_pool(pools['full']['rows'], sample_size, draw)
                pools[f'sample_{draw}'] = _rank_pool(sample)
        for value in pools.values():
            value['deltas'] = _deltas(value)

    offline = offline_decision(pools, fresh=fresh is not None)
    trial_id = (hashlib.sha256(json.dumps(pools['full']['rows'], sort_keys=True,
                                          allow_nan=False).encode()).hexdigest()
                if pools else None)
    live_rounds = [] if live is None else live
    live_counts = summarize_live(live_rounds, trial_id, pools)
    checkpoint = {key: live_counts[key] for key in ('keeps', 'n', 'per_list_keeps')}
    return dict(
        version=4, base_sha=BASE_SHA, split='fresh judgments' if fresh is not None else 'position-based',
        settings=dict(holdout_start=holdout_start, cutoff=cutoff, reject_fold=reject_fold,
                      sample_size=sample_size, seeds=list(SEEDS), k=K),
        source_counts=dict(likes=len(likes), rejects=len(rejects), catalog=len(catalog),
                           fresh=len(fresh or [])),
        scope='Matched ranking on the frozen catalog. Retrieval quality is unverified.',
        training_counts={kind: len(records) for kind, records in train.items()},
        seed_recording_id=seed_id, manifest=audit_rows, pools=pools, blockers=blocked,
        trial_id=trial_id, offline_status=offline, live=live_counts, live_rounds=live_rounds,
        status=live_decision(offline, **checkpoint),
    )


def summarize_live(rounds, trial_id, pools):
    """Count only unique challenger ratings linked to a private round key."""
    if not isinstance(rounds, list):
        raise ValueError('live input must be a list of rated rounds')
    rows = {row['recording_id']: row for row in pools.get('full', {}).get('rows', ())}
    seen_recordings, seen_spotify, challenger = set(), set(), []
    for entry in rounds:
        if trial_id is None or entry.get('trial_id') != trial_id:
            raise ValueError('live round belongs to a different trial')
        deck, ratings = entry['reveal'], entry['ratings']
        tokens = {row['token'] for row in deck}
        if (len(deck) != len(tokens) or set(ratings) != tokens
                or not 2 <= len(deck) <= K or any(type(value) is not bool for value in ratings.values())
                or Counter(row['method'] for row in deck) != {TRIAL: len(deck)//2,
                                                              'v2_refit': len(deck)//2}
                or len(deck) % 2):
            raise ValueError('live round needs balanced arms and one boolean rating per token')
        for pick in deck:
            recording, spotify = pick['recording_id'], pick['spotify_id']
            row = rows.get(recording)
            if recording in seen_recordings or spotify in seen_spotify:
                raise ValueError('live rating repeats a recording')
            if (not isinstance(spotify, str) or not re.fullmatch(r'[A-Za-z0-9]{22}', spotify)
                    or not row or row['label'] != 'unlabeled' or row['scores'] != pick['scores']
                    or row['ranks'] != pick['ranks']):
                raise ValueError('live rating does not match the frozen trial rows')
            seen_recordings.add(recording)
            seen_spotify.add(spotify)
            if pick['method'] == TRIAL:
                challenger.append(ratings[pick['token']])
    total = len(challenger)
    if total > 50:
        raise ValueError('live protocol ends at 50 challenger judgments')
    n = 42 if 42 <= total < 50 else total
    return dict(keeps=sum(challenger[:n]), n=n, total_challenger_judged=total,
                per_list_keeps=[sum(challenger[i:i+10]) for i in range(0, n - 9, 10)])



def prepare_blind(result, verified, excluded_ids, excluded_keys):
    """Create a balanced, shuffled deck and an operator-only answer key."""
    if result['offline_status'] != 'ELIGIBLE_FOR_BLIND':
        raise ValueError('offline evidence has not qualified for a blind round')
    rows = result['pools']['full']['rows']
    manifest = {row['row_id']: row for row in result['manifest']}
    methods = [TRIAL, 'v2_refit']
    rng = secrets.SystemRandom()
    rng.shuffle(methods)
    selected = {method: [] for method in methods}
    ranking = {method: iter(sorted(rows, key=lambda row: row['ranks'][method]))
               for method in methods}
    previous = [row for entry in result.get('live_rounds', []) for row in entry['reveal']]
    seen_recordings = {row['recording_id'] for row in previous}
    seen_ids = set(excluded_ids) | {row['spotify_id'] for row in previous}
    seen_keys = set(excluded_keys)
    for _ in range(K // 2):
        for method in methods:
            for row in ranking[method]:
                sid = verified.get(row['recording_id'])
                data = manifest[row['row_id']]
                key = (embeat.clean(data['title']), embeat.normalize_artist(data['artist']))
                if (row['label'] != 'unlabeled' or not isinstance(sid, str)
                        or not re.fullmatch(r'[A-Za-z0-9]{22}', sid)
                        or sid in seen_ids or row['recording_id'] in seen_recordings
                        or key in seen_keys or not all(key)):
                    continue
                seen_ids.add(sid)
                seen_recordings.add(row['recording_id'])
                seen_keys.add(key)
                selected[method].append((row, data, sid))
                break

    # Trim both arms to the same number if the verified catalog is sparse.
    remaining = 50 - result.get('live', {}).get('total_challenger_judged', 0)
    n = min(remaining, *map(len, selected.values()))
    draft = [(method, row, meta, sid) for method, items in selected.items()
             for row, meta, sid in items[:n]]
    rng.shuffle(draft)
    blind, reveal = [], []
    for index, (method, row, meta, sid) in enumerate(draft, 1):
        token = f'{index:02d}'
        blind.append(dict(token=token, title=meta['title'], artist=meta['artist'], spotify_id=sid))
        reveal.append(dict(token=token, method=method, recording_id=row['recording_id'], spotify_id=sid,
                           scores=row['scores'], ranks=row['ranks']))
    return blind, reveal


def _read_track_export(path):
    rows = {}
    for position, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        fields = line.split('\t')
        if len(fields) >= 3 and not line.startswith('#'):
            title, artist, uri = fields[:3]
            rows[position] = (title, artist, uri.removeprefix('spotify:track:'))
    return rows


def _write_blind(args, result):
    if result['status'] != 'PENDING_LIVE':
        return
    required = ('liked_export', 'history', 'verified_spotify')
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        result['blind_list'] = {'status': 'missing shortlist inputs', 'missing': missing}
        return
    current_likes = args.current_likes or args.liked_export
    exclusions = [row for path in (current_likes, args.history)
                  for row in _read_track_export(path).values()]
    ids = {sid for _, _, sid in exclusions}
    keys = {(embeat.clean(title), embeat.normalize_artist(artist))
            for title, artist, _ in exclusions}
    verified = json.loads(args.verified_spotify.read_text(encoding='utf-8'))
    blind, reveal = prepare_blind(result, verified, ids, keys)
    if blind:
        with args.blind_output.open('x', encoding='utf-8') as stream:
            json.dump(blind, stream, indent=2)
            stream.write('\n')
        result['blind_list'] = {'status': 'local deck created', 'count': len(blind),
                                'trial_id': result['trial_id'], 'reveal_after_ratings': reveal}
    else:
        result['blind_list'] = {'status': 'no balanced verified shortlist'}



def _read_records(path, key=None):
    doc = json.loads(path.read_text(encoding='utf-8'))
    if key is not None:
        doc = doc[key]
    if not isinstance(doc, list) or any(not isinstance(row, dict) for row in doc):
        raise ValueError(f'{path} must contain a list of record objects')
    return doc


def _enrich_records(records, matches, previews):
    hashes = {}
    enriched = []
    for record in records:
        record = dict(record)
        match = matches.get(record.get('spotify_id'), {})
        # Compare the recorded match before borrowing metadata from it.
        if (record.get('deezer_id') and
                str(match.get('deezer_id')) == str(record['deezer_id']) and
                not record.get('isrc') and match.get('isrc')):
            record['isrc'] = match['isrc']
        deezer_id = str(record.get('deezer_id'))
        if previews is not None and deezer_id.isascii() and deezer_id.isdecimal():
            if deezer_id not in hashes:
                path = previews / f'{deezer_id}.mp3'
                hashes[deezer_id] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                     if path.is_file() and path.stat().st_size else None)
            if hashes[deezer_id]:
                if record.get('preview_sha256') not in (None, hashes[deezer_id]):
                    raise ValueError(f'preview hash mismatch for Deezer ID {deezer_id}')
                record['preview_sha256'] = hashes[deezer_id]
        enriched.append(record)
    return enriched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--likes', type=Path, default=Path('features_v2.json'))
    parser.add_argument('--rejects', type=Path, default=Path('data/negatives_features.json'))
    parser.add_argument('--catalog', type=Path, default=Path('data/pool_features.json'))
    parser.add_argument('--fresh', type=Path, help='Untouched judgments with existing features')
    parser.add_argument('--previews', type=Path, help='Optional authorized Deezer preview cache')
    parser.add_argument('--matches', type=Path, help='Optional frozen Deezer match cache')
    parser.add_argument('--holdout-start', type=int, default=0)
    parser.add_argument('--cutoff', type=int, default=130)
    parser.add_argument('--reject-fold', type=int, choices=range(3), default=0)
    parser.add_argument('--sample-size', type=int, default=200)
    parser.add_argument('--live', type=Path, help='Local rated rounds with private answer keys')
    parser.add_argument('--liked-export', type=Path, help='Original liked-track TSV')
    parser.add_argument('--current-likes', type=Path, help='Current liked-track TSV for blind-list exclusions')
    parser.add_argument('--history', type=Path, help='Prior recommendations TSV')
    parser.add_argument('--verified-spotify', type=Path, help='Recording ID to verified Spotify ID JSON')
    parser.add_argument('--blind-output', type=Path, help='Optional separate listener-only deck')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {name: getattr(args, name) for name in
             ('likes', 'rejects', 'catalog', 'fresh', 'matches', 'live',
              'liked_export', 'current_likes', 'history', 'verified_spotify')
             if getattr(args, name) is not None}
    input_paths = {path.resolve() for path in paths.values()}
    if args.output.resolve() in input_paths:
        parser.error('output must differ from all source files')
    if args.blind_output and args.blind_output.resolve() in input_paths | {args.output.resolve()}:
        parser.error('blind output must differ from the audit and all source files')
    missing = {name: str(path) for name, path in paths.items() if not path.is_file()}
    if args.previews is not None and not args.previews.is_dir():
        missing['previews'] = str(args.previews)
    if missing:
        result = dict(version=4, base_sha=BASE_SHA, status='INSUFFICIENT_EVIDENCE',
                      offline_status='INSUFFICIENT_EVIDENCE', missing_inputs=missing, pools={})
    else:
        matches = json.loads(args.matches.read_text(encoding='utf-8')) if args.matches else {}
        inputs = {name: _enrich_records(_read_records(path, 'tracks' if name == 'likes' else None),
                                        matches, args.previews)
                  for name, path in paths.items() if name in ('likes', 'rejects', 'catalog', 'fresh')}
        if args.liked_export:
            original = _read_track_export(args.liked_export)
            if set(original) != {row.get('position') for row in inputs['likes']} or any(
                    original[row['position']][2] != str(row.get('spotify_id'))
                    for row in inputs['likes']):
                parser.error('original liked export does not match the frozen feature snapshot')
        live = json.loads(args.live.read_text(encoding='utf-8')) if args.live else None
        result = run_experiment(**inputs, holdout_start=args.holdout_start, cutoff=args.cutoff,
                                reject_fold=args.reject_fold, sample_size=args.sample_size, live=live)
        result['source_files'] = {name: {'path': str(path),
                                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                  for name, path in paths.items()}
        if args.blind_output:
            _write_blind(args, result)
    with args.output.open('w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(result['status'], file=sys.stderr)
    return 2 if missing or result.get('blockers') else 0


if __name__ == '__main__':
    raise SystemExit(main())
