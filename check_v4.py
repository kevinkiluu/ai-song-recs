#!/usr/bin/env python3
"""Independently verify a local v4 audit from its manifest and scored rows."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
import sys

from sklearn.metrics import ndcg_score
from scipy.stats import binomtest

METHODS = {'v2_refit', 'v6_frozen', 'embeat_full', 'embeat_minus_ch1'}
CHALLENGER = 'embeat_minus_ch1'


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _equal(expected, actual, name):
    if isinstance(expected, float):
        _require(isinstance(actual, (int, float))
                 and math.isclose(expected, actual, abs_tol=1e-12), name)
    else:
        _require(expected == actual, name)


def _identities(row):
    identities = set()
    placeholders = {'', 'none', 'null', 'nan', 'unknown', '0'}
    for name, key in (('spotify', 'spotify_id'), ('deezer', 'deezer_id'),
                      ('isrc', 'isrc'), ('sha256', 'preview_sha256')):
        text = '' if row[key] is None else str(row[key]).strip()
        if text.lower() in placeholders:
            continue
        if name == 'spotify':
            text = text[len('spotify:track:'):] if text.startswith('spotify:track:') else text
        if name == 'deezer':
            text = text.lstrip('0') if re.fullmatch(r'[0-9]+', text) else ''
        if name == 'isrc':
            text = ''.join(text.split('-')).upper()
        if name == 'sha256':
            text = text.lower()
            _require(re.fullmatch(r'[0-9a-f]{64}', text), 'invalid preview identity')
        if text.lower() not in placeholders:
            identities.add(f'{name}:{text}')
    return identities



def _verify_manifest(manifest):
    by_row, groups, by_token = {}, defaultdict(list), defaultdict(set)
    for row in manifest:
        row_id = row['row_id']
        _require(row_id not in by_row, 'duplicate manifest row')
        by_row[row_id] = row
        group = row['recording_id']
        groups[group].append(row)
        tokens = _identities(row)
        _equal(tokens, set(row['identities']), 'provider identity mismatch')
        for token in tokens:
            by_token[token].add(group)
        partition = row['partition']
        _require(partition in ('train', 'test', 'excluded'), 'unrecognized partition')
        _require((partition == 'excluded') == bool(row['reason']), 'missing exclusion reason')
        _require(bool(row['identities']) or partition == 'excluded', 'unverified recording active')
        if partition == 'train':
            _require(row['source'] in ('liked_snapshot', 'historical_judgment'),
                     'unlabeled or sealed judgment used for training')
    _require(all(len(ids) == 1 for ids in by_token.values()), 'identity spans recordings')
    for group, rows in groups.items():
        _require(sum(row['partition'] != 'excluded' for row in rows) <= 1,
                 'recording has multiple active memberships')
        all_tokens = [token for row in rows for token in row['identities']]
        _require(group == (min(all_tokens) if all_tokens else rows[0]['row_id']),
                 'recording does not use the canonical identity')
        if all_tokens:
            connected = {group}
            remaining = [set(row['identities']) for row in rows]
            while remaining:
                linked = [tokens for tokens in remaining if tokens & connected]
                _require(bool(linked), 'recording merges unconnected identities')
                remaining = [tokens for tokens in remaining if not tokens & connected]
                connected.update(*linked)
        original_labels = {row['label'] for row in rows if row['source'] != 'frozen_catalog'}
        _require(len(original_labels) <= 1 or all(r['partition'] == 'excluded' for r in rows),
                 'conflicting recording labels not quarantined')
    return {r['recording_id']: r for r in manifest if r['partition'] == 'test'}


def _verify_pool(pool, expected, k):
    rows = pool['rows']
    counts = Counter(row['label'] for row in rows)
    _require(len(rows) == len(expected) and {r['recording_id'] for r in rows} == set(expected),
             'pool recording membership mismatch')
    for row in rows:
        source = expected[row['recording_id']]
        for name in ('row_id', 'label', 'label_source'):
            _equal(source[name], row[name], f'pool {name} mismatch')
        _require(set(row['scores']) == set(row['ranks']) == METHODS, 'missing rank or score')
        _require(all(math.isfinite(value) for value in row['scores'].values())
                 and math.isfinite(row['embeat_tie_score']), 'nonfinite score')
    _equal(dict(counts), pool['counts'], 'pool label counts')
    chance = (counts['like'] / len(rows)) * (min(len(rows), k) / k)
    _equal(chance, pool['chance_p_at_k'], 'chance P@k')
    _equal((len(rows) + 1) / 2, pool['chance_mean_rank'], 'chance mean rank')

    _require(set(pool['metrics']) == METHODS, 'pool method coverage')
    for method in METHODS:
        keys = [(r['scores'][method] * (1 if method == 'v6_frozen' else -1),
                 -r['embeat_tie_score'] if method.startswith('embeat_') else 0,
                 r['recording_id'], r) for r in rows]
        ranked = [row for _, _, _, row in sorted(keys)]
        _equal(list(range(1, len(ranked) + 1)), [r['ranks'][method] for r in ranked],
               f'{method} rank order')
        relevance = [int(row['label'] == 'like') for row in ranked]
        scores = list(range(len(rows), 0, -1))
        ndcg = (float(ndcg_score([relevance], [scores], k=k, ignore_ties=True))
                if len(rows) > 1 else float(relevance[0]))
        bad = [n for n, r in enumerate(ranked, 1) if r['label'] == 'reject']
        denominator = min(counts['like'], k)
        ap = (sum(sum(relevance[:i]) / i for i, r in enumerate(relevance[:k], 1) if r)
              / denominator if denominator else 0.0)
        metrics = dict(p_at_k=sum(relevance[:k]) / k, ndcg_at_k=ndcg, map_at_k=ap,
                       negative_ranks=bad, mean_negative_rank=mean(bad) if bad else None,
                       negative_top_k=sum(n <= k for n in bad))
        for name, value in metrics.items():
            _equal(value, pool['metrics'][method][name], f'{method} {name}')
        if method != CHALLENGER:
            for name in ('p_at_k', 'ndcg_at_k', 'map_at_k'):
                delta = pool['metrics'][CHALLENGER][name] - metrics[name]
                _equal(delta, pool['deltas'][method][name], f'{method} delta {name}')


def _offline_status(report, k):
    if report['split'] != 'fresh judgments' or not report['pools']:
        return 'INSUFFICIENT_EVIDENCE'
    full = report['pools']['full']
    if full['counts'].get('like', 0) < k or full['counts'].get('reject', 0) < 2:
        return 'INSUFFICIENT_EVIDENCE'
    candidate = full['metrics'][CHALLENGER]
    passing = (candidate['p_at_k'] >= .5 and candidate['p_at_k'] > full['chance_p_at_k']
               and not candidate['negative_top_k']
               and candidate['mean_negative_rank'] > full['chance_mean_rank'])
    ablation = full['deltas']['embeat_full']
    passing &= ablation['ndcg_at_k'] > 0 and ablation['p_at_k'] >= 0
    for method in ('v2_refit', 'v6_frozen'):
        effect = full['deltas'][method]
        passing &= effect['ndcg_at_k'] > 0 and (
            effect['p_at_k'] >= 0.1 - 1e-12 or
            (effect['p_at_k'] == 0 and effect['map_at_k'] >= .02 - 1e-12))
        for key, sample in report['pools'].items():
            if key != 'full':
                passing &= effect['p_at_k'] * sample['deltas'][method]['p_at_k'] >= 0
    return 'ELIGIBLE_FOR_BLIND' if passing else 'NO_GO_OFFLINE'


def _live_status(offline, live):
    n, keeps = live['n'], live['keeps']
    per_list = live['per_list_keeps']
    _require(type(n) is type(keeps) is int and 0 <= keeps <= n <= 50
             and all(type(x) is int and 0 <= x <= 10 for x in per_list)
             and 0 <= keeps - sum(per_list) <= n - len(per_list) * 10,
             'invalid live judgment counts')
    if offline != 'ELIGIBLE_FOR_BLIND':
        return offline
    if any(a <= 7 and b <= 7 for a, b in zip(per_list, per_list[1:])):
        return 'NO_GO_LIVE'
    if n not in (42, 50):
        return 'PENDING_LIVE'
    p = keeps / n
    z = 1.96
    bound = (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
    test = binomtest(keeps, n, p=.60, alternative='greater')
    return 'GO' if bound > .60 and test.pvalue <= .05 else 'NO_GO_LIVE'


def _verify_live_summary(report):
    rows = {row['recording_id']: row for row in report['pools'].get('full', {}).get('rows', [])}
    trial = (hashlib.sha256(json.dumps(report['pools']['full']['rows'],
                                        sort_keys=True, allow_nan=False).encode()).hexdigest()
             if rows else None)
    _equal(trial, report['trial_id'], 'trial identity mismatch')
    seen_ids, seen_recordings, replay = set(), set(), []
    for round_ in report['live_rounds']:
        _require(trial is not None and round_['trial_id'] == trial, 'wrong live trial')
        ratings, reveal = round_['ratings'], round_['reveal']
        _require(2 <= len(reveal) <= 10 and set(ratings) == {r['token'] for r in reveal}
                 and len(ratings) == len(reveal), 'live round token coverage')
        _require(Counter(entry['method'] for entry in reveal) ==
                 {CHALLENGER: len(reveal)/2, 'v2_refit': len(reveal)/2}, 'unbalanced live arms')
        for entry in reveal:
            rec, sid = entry['recording_id'], entry['spotify_id']
            _require(rec in rows and rows[rec]['label'] == 'unlabeled',
                     'live pick not in trial catalog')
            _require(rec not in seen_recordings and sid not in seen_ids
                     and re.fullmatch(r'[A-Za-z0-9]{22}', sid), 'live recording repeats or lacks an ID')
            for name in ('scores', 'ranks'):
                _equal(rows[rec][name], entry[name], f'live reveal {name}')
            seen_recordings.add(rec)
            seen_ids.add(sid)
            keep = ratings[entry['token']]
            _require(type(keep) is bool, 'live rating is not boolean')
            if entry['method'] == CHALLENGER:
                replay.append(keep)
    count = len(replay)
    _require(count <= 50, 'too many live ratings')
    checkpoint = 42 if count in range(42, 50) else count
    scored = replay[:checkpoint]
    expected = dict(n=checkpoint, keeps=sum(scored), total_challenger_judged=count,
                    per_list_keeps=[sum(scored[start:start+10])
                                    for start in range(0, len(scored)//10 * 10, 10)])
    _equal(expected, report['live'], 'live summary disagrees with individual ratings')



def verify(report):
    """Recompute from the audit. This module never imports the v4 runner."""
    _require(report['version'] == 4, 'expected v4 audit')
    if report.get('missing_inputs'):
        _require(not report['pools'] and report['status'] == report['offline_status'] ==
                 'INSUFFICIENT_EVIDENCE', 'unavailable-input verdict')
        return
    expected = _verify_manifest(report['manifest'])
    train = [r for r in report['manifest'] if r['partition'] == 'train']
    for kind in ('like', 'reject'):
        _equal(sum(r['label'] == kind for r in train), report['training_counts'][kind],
               f'{kind} training count')
    settings = report['settings']
    k = settings['k']
    _require(k == 10 and settings['seeds'] == [42, 43, 44, 45, 46], 'unexpected protocol')
    if report['split'] == 'fresh judgments':
        inputs = {row['row_id']: row for row in report['manifest']}
        for row in expected.values():
            _require(row['source'] == 'frozen_catalog', 'fresh pool uses unfrozen features')
            if row['label'] != 'unlabeled':
                judgment = inputs.get(row.get('judgment_row_id'), {})
                _require(judgment.get('source') == 'fresh_judgment' and all(
                    judgment.get(field) == row[field]
                    for field in ('recording_id', 'label', 'label_source')),
                    'fresh evidence has no matched fresh label')
    elif report['split'] == 'position-based':
        _require(all(row['position'] >= settings['cutoff'] for row in train
                     if row['label'] == 'like'), 'training like is newer than position holdout')
        _require(all(settings['holdout_start'] <= row['position'] < settings['cutoff']
                     for row in expected.values() if row['label'] == 'like'),
                 'judged like is outside position holdout')
    else:
        raise ValueError('unknown split')
    pools = report['pools']
    if pools:
        _require(set(pools) == {'full', *(f'sample_{seed}' for seed in settings['seeds'])},
                 'missing sensitivity pool')
        _require(any(r['recording_id'] == report['seed_recording_id'] and r['label'] == 'like'
                     for r in train), 'seed was not a training like')
        _verify_pool(pools['full'], expected, k)
        unknown = sorted(i for i, row in expected.items() if row['label'] == 'unlabeled')
        judged = {i for i, row in expected.items() if row['label'] != 'unlabeled'}
        for seed in settings['seeds']:
            drawn = random.Random(seed).sample(unknown, min(settings['sample_size'], len(unknown)))
            sample = {i: expected[i] for i in judged | set(drawn)}
            actual_sample = pools[f'sample_{seed}']
            _verify_pool(actual_sample, sample, k)
            full_rows = {row['recording_id']: row for row in pools['full']['rows']}
            for row in actual_sample['rows']:
                original = full_rows[row['recording_id']]
                _equal(original['scores'], row['scores'], 'sample score drift')
                _equal(original['embeat_tie_score'], row['embeat_tie_score'], 'sample tie drift')
    else:
        _require(bool(report['blockers']), 'empty run has no blocker')
    _verify_live_summary(report)
    offline = _offline_status(report, k)
    _equal(offline, report['offline_status'], 'offline verdict')
    _equal(_live_status(offline, report['live']), report['status'], 'live verdict')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python check_v4.py path/to/evaluation-v4.json')
    verify(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
    print('Verified v4 audit')
