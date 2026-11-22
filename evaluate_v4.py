#!/usr/bin/env python3
"""Recording-level evaluation for the v4 Embeat channel 1 experiment."""

import math
import random
import re
from collections import defaultdict
from statistics import mean

SEEDS = (42, 43, 44, 45, 46)
PREVIEW_FIELDS = ('bpm', 'rms_mean', 'brightness_mean')


def identity_tokens(record):
    tokens = []
    for field, namespace in (('spotify_id', 'spotify'), ('deezer_id', 'deezer'),
                             ('isrc', 'isrc'), ('preview_sha256', 'sha256')):
        value = record.get(field)
        value = '' if value is None else str(value).strip()
        if value.lower() in ('', 'none', 'null', 'nan', 'unknown', '0'):
            continue
        if namespace == 'spotify':
            value = value.removeprefix('spotify:track:')
        elif namespace == 'deezer':
            if not value.isascii() or not value.isdecimal():
                continue
            value = value.lstrip('0')
        elif namespace == 'isrc':
            value = value.replace('-', '').upper()
        elif namespace == 'sha256':
            value = value.lower()
            if not re.fullmatch(r'[0-9a-f]{64}', value):
                raise ValueError('preview_sha256 must be a 64-character hex digest')
        if value.lower() not in ('', 'none', 'null', 'nan', 'unknown', '0'):
            tokens.append(f'{namespace}:{value}')
    return tokens


def has_preview_features(record):
    features = record.get('librosa') or {}
    return all(isinstance(features.get(key), (int, float))
               and math.isfinite(features[key]) for key in PREVIEW_FIELDS)


def build_manifest(likes, rejects, catalog, *, cutoff=130, holdout_start=0,
                   reject_fold=0, reject_folds=3, fresh=None):
    """Assign exactly one representative to train or test per recording."""
    if not 0 <= holdout_start < cutoff or not 0 <= reject_fold < reject_folds:
        raise ValueError('invalid holdout positions or reject fold')
    rows, parents = [], {}

    def root(token):
        while parents[token] != token:
            parents[token] = parents[parents[token]]
            token = parents[token]
        return token

    for source, default_label, records in (
        ('liked_snapshot', 'like', likes), ('historical_judgment', 'reject', rejects),
        ('frozen_catalog', 'unlabeled', catalog), ('fresh_judgment', None, fresh or ()),
    ):
        for index, record in enumerate(records):
            label = default_label or record.get('label')
            if label not in ('like', 'reject', 'unlabeled') or (
                    source == 'fresh_judgment' and
                    (label == 'unlabeled' or not record.get('label_source'))):
                raise ValueError('fresh judgments require a like/reject label and label_source')
            keys = identity_tokens(record)
            for key in keys:
                parents.setdefault(key, key)
                a, b = sorted((root(keys[0]), root(key)))
                parents[b] = a
            rows.append(dict(row_id=f'{source}:{index}', source=source, label=label,
                             label_source=record.get('label_source') or source,
                             record=record, identities=keys, partition='excluded',
                             reason='unverified recording identity' if not keys else 'recording alias'))

    groups = defaultdict(list)
    for row in rows:
        keys = row['identities']
        row['recording_id'] = root(keys[0]) if keys else row['row_id']
        if keys:
            groups[row['recording_id']].append(row)

    reject_groups = sorted(key for key, group in groups.items()
                           if {r['label'] for r in group} <= {'reject', 'unlabeled'}
                           and any(r['source'] == 'historical_judgment' for r in group))
    reject_test = set(reject_groups[reject_fold::reject_folds])

    for key, group in sorted(groups.items()):
        labeled = [r for r in group if r['label'] != 'unlabeled']
        historic = [r for r in labeled if r['source'] != 'fresh_judgment']
        fresh_rows = [r for r in labeled if r['source'] == 'fresh_judgment']
        if len({r['label'] for r in labeled}) > 1:
            for row in group:
                row['reason'] = 'conflicting labels'
            continue
        if fresh_rows and historic:
            for row in fresh_rows:
                row['reason'] = 'previously labeled recording'
            labeled = historic
        elif fresh_rows:
            candidates = [row for row in group if row['source'] == 'frozen_catalog'
                          and has_preview_features(row['record'])]
            if candidates:
                chosen = min(candidates, key=lambda row: (tuple(row['identities']), row['row_id']))
                judgment = min(fresh_rows, key=lambda row: row['row_id'])
                chosen.update(partition='test', reason=None, label=judgment['label'],
                              label_source=judgment['label_source'], judgment_row_id=judgment['row_id'])
            for row in fresh_rows:
                row['reason'] = ('judgment joined to frozen catalog' if candidates
                                 else 'recording absent from eligible frozen catalog')
            continue

        partition = 'test'
        choices = labeled or group
        if historic and historic[0]['label'] == 'like':
            if any(type(r['record'].get('position')) is not int
                   or r['record']['position'] < 0 for r in historic):
                for row in historic:
                    row['reason'] = 'invalid source position'
                continue
            choices = [r for r in historic
                       if holdout_start <= r['record']['position'] < cutoff] if fresh is None else []
            if choices:
                for row in historic:
                    if row not in choices:
                        row['reason'] = 'held-out recording alias'
            else:
                partition = 'train'
                choices = historic if fresh is not None else [
                    row for row in historic if row['record']['position'] >= cutoff]
                for row in historic:
                    if row not in choices:
                        row['reason'] = 'newer than the position holdout'
                if not choices:
                    continue
        elif historic:
            partition = 'test' if fresh is None and key in reject_test else 'train'

        if partition == 'test':
            eligible = [r for r in choices if has_preview_features(r['record'])]
            if not eligible:
                for row in choices:
                    row['reason'] = 'missing shared ranking features'
                continue
            choices = eligible
        representative = min(choices, key=lambda r: (
            r['record'].get('position', math.inf) if r['source'] == 'liked_snapshot' else math.inf,
            tuple(r['identities']), r['row_id'],
        ))
        representative.update(partition=partition, reason=None)

    return rows


def evaluation_pool(manifest):
    return sorted((r for r in manifest if r['partition'] == 'test'),
                  key=lambda row: row['recording_id'])


def sample_pool(pool, n, seed):
    """Keep judged rows and sample distractors without replacement."""
    if n < 0:
        raise ValueError('sample size must be nonnegative')
    judged, distractors = [], []
    for row in sorted(pool, key=lambda r: r['recording_id']):
        (distractors if row['label'] == 'unlabeled' else judged).append(row)
    selected = random.Random(seed).sample(distractors, min(n, len(distractors)))
    return sorted(judged + selected, key=lambda row: row['recording_id'])


def ranking_metrics(ordered, k=10):
    if k <= 0:
        raise ValueError('k must be positive')
    positives = sum(row['label'] == 'like' for row in ordered)
    hit_ranks, negative_ranks = [], []
    for rank, row in enumerate(ordered, 1):
        if row['label'] == 'like' and rank <= k:
            hit_ranks.append(rank)
        if row['label'] == 'reject':
            negative_ranks.append(rank)
    dcg = sum(1 / math.log2(rank + 1) for rank in hit_ranks)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, positives) + 1))
    average_precision = sum(hits / rank for hits, rank in enumerate(hit_ranks, 1))
    return {
        'p_at_k': len(hit_ranks) / k,
        'ndcg_at_k': dcg / ideal if ideal else 0.0,
        'map_at_k': average_precision / min(k, positives) if positives else 0.0,
        'negative_ranks': negative_ranks,
        'mean_negative_rank': mean(negative_ranks) if negative_ranks else None,
        'negative_top_k': sum(rank <= k for rank in negative_ranks),
    }


def live_decision(offline_status, keeps=0, n=0, per_list_keeps=()):
    """Evaluate the documented v3 live checkpoints with the minimum enforced."""
    if (type(keeps) is not int or type(n) is not int or not 0 <= keeps <= n <= 50
            or any(type(k) is not int or not 0 <= k <= 10 for k in per_list_keeps)
            or len(per_list_keeps) * 10 > n
            or not 0 <= keeps - sum(per_list_keeps) <= n - len(per_list_keeps) * 10):
        raise ValueError('invalid live keep counts')
    if offline_status != 'ELIGIBLE_FOR_BLIND':
        return offline_status
    if any(max(a, b) <= 7 for a, b in zip(per_list_keeps, per_list_keeps[1:])):
        return 'NO_GO_LIVE'
    if n not in (42, 50):
        return 'PENDING_LIVE'
    # At these fixed sample sizes both the Wilson and one-sided binomial
    # criteria against 0.60 are met at exactly these integer boundaries.
    return 'GO' if keeps >= {42: 32, 50: 37}[n] else 'NO_GO_LIVE'
