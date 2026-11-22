#!/usr/bin/env python3
"""Independently verify a local v4 audit from its manifest and scored rows."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
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


def _verify_manifest(manifest):
    by_row, groups, by_token = {}, defaultdict(list), defaultdict(set)
    for row in manifest:
        row_id = row['row_id']
        _require(row_id not in by_row, 'duplicate manifest row')
        by_row[row_id] = row
        group = row['recording_id']
        groups[group].append(row)
        for token in row['identities']:
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
    if len(per_list) >= 2 and all(k <= 7 for k in per_list[-2:]):
        return 'NO_GO_LIVE'
    if n not in (42, 50):
        return 'PENDING_LIVE'
    p = keeps / n
    z = 1.96
    bound = (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
    test = binomtest(keeps, n, p=.60, alternative='greater')
    return 'GO' if bound > .60 and test.pvalue <= .05 else 'NO_GO_LIVE'


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
        _require(all(row['source'] == 'fresh_judgment'
                     for row in expected.values() if row['label'] != 'unlabeled'),
                 'fresh evidence includes historical labels')
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
            _verify_pool(pools[f'sample_{seed}'], sample, k)
    else:
        _require(bool(report['blockers']), 'empty run has no blocker')
    offline = _offline_status(report, k)
    _equal(offline, report['offline_status'], 'offline verdict')
    _equal(_live_status(offline, report['live']), report['status'], 'live verdict')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python check_v4.py path/to/evaluation-v4.json')
    verify(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
    print('Verified v4 audit')
