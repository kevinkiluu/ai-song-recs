#!/usr/bin/env python3
"""Acceptance tests for pipeline v2 (obra TDD: checks first, watch fail,
then the work that passes them).

Each check is one falsifiable claim about the finished result, read from
the real files on disk (never from a summary). Run:
  ~/workspace/audio-env/bin/python test_pipeline_v2.py
Exit 0 = every check green. Exit 1 = at least one check red (listed).

RED state (Nov 21, 2026): checks 4-7 fail because evaluation_v2.json,
recommendations_v2.json and evaluation_report_v2.md do not exist yet.
GREEN requires the evaluation harness run, the 10 recommendations
generated, and the report written.
"""
import json, os, sys

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
LIKED_FILE = os.path.expanduser('~/workspace/spotify-liked-songs.txt')
HIST_FILE = os.path.expanduser('~/workspace/spotify-recommended-tracks.txt')

failures = []
passes = []


def check(name, cond, detail=''):
    if cond:
        passes.append(name)
        print(f'PASS  {name}')
    else:
        failures.append(name)
        print(f'FAIL  {name}  {detail}')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def id_set(path):
    ids = set()
    for line in open(path):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3 and not line.startswith('#'):
            ids.add(p[2].replace('spotify:track:', ''))
    return ids


# --- 1. features file
try:
    doc = load_json(os.path.join(BASE, 'features_v2.json'))
    tr = doc['tracks']
    check('features: 650 tracks', len(tr) == 650, f'got {len(tr)}')
    check('features: librosa >= 400', sum(1 for t in tr if t.get('librosa')) >= 400)
    check('features: reccobeats >= 400', sum(1 for t in tr if t.get('reccobeats')) >= 400)
    check('features: test set librosa >= 90',
          sum(1 for t in tr if t['position'] < 130 and t.get('librosa')) >= 90)
    check('features: notes record preview-only + no full-length audio',
          any('no full-length audio files' in n for n in doc.get('notes', [])))
    check('features: genre_bucket present on >= 600 tracks',
          sum(1 for t in tr if t.get('genre_bucket')) >= 600,
          f"got {sum(1 for t in tr if t.get('genre_bucket'))} (a /tmp wipe "
          f"once zeroed these; see load_buckets)")
except FileNotFoundError:
    check('features: file exists', False, 'features_v2.json missing')

# --- 2. model
try:
    m = load_json(os.path.join(BASE, 'model_v2.json'))
    check('model: gmm_k in 2..8', 2 <= m['gmm_k'] <= 8, f"k={m.get('gmm_k')}")
    check('model: 13 negatives in LR', m.get('lr_n_neg') == 13,
          f"lr_n_neg={m.get('lr_n_neg')}")
    check('model: npz exists', os.path.exists(os.path.join(BASE, 'model_v2.npz')))
except FileNotFoundError:
    check('model: file exists', False, 'model_v2.json missing')

# --- 3. negatives
try:
    negs = load_json(os.path.join(DATA, 'negatives_features.json'))
    check('negatives: 13 records', len(negs) == 13, f'got {len(negs)}')
    want = {'6aJn7Cst74cj4lNIiPRgav', '6HacgXCExkzS552ILfJTXu',
            '1GfBLbAhZUWdseuDqhocmn', '5v7oj9PosKdE39x9uwDs6C',
            '4bkPAARkDov1lV05emEF7a', '2jv1ovpDpTJEPIlth5Wn07',
            '7zDu3l7DDlfSlt20YMKkWo', '2y57H6nA3vic5t8awwaS69',
            '2slLXFcITuhf0oSp2wVJLv', '30efCAopZCFcoi43sSToNr',
            '0FmrNFYyLLRtIS7XWJPLLk', '32gSPav2xqFylAeCjJCxWi',
            '1vvBUZseJ7fwYPX1NedOLd'}
    got = {r['spotify_id'] for r in negs}
    check('negatives: all 13 IDs present', want == got,
          f'missing={want - got} extra={got - want}')
    check('negatives: >= 12 with librosa',
          sum(1 for r in negs if r.get('librosa')) >= 12)
except FileNotFoundError:
    check('negatives: file exists', False, 'negatives_features.json missing')

# --- 4. evaluation (RED until evaluate.py runs)
try:
    ev = load_json(os.path.join(BASE, 'evaluation_v2.json'))
    for method in ('v2', 'v6_per_bucket', 'v6_global'):
        check(f'eval: {method} P@10 in [0,1]',
              0.0 <= ev[method]['P@10'] <= 1.0, f"{ev[method].get('P@10')}")
        check(f'eval: {method} NDCG@10 in [0,1]',
              0.0 <= ev[method]['NDCG@10'] <= 1.0, f"{ev[method].get('NDCG@10')}")
        check(f'eval: {method} mean negative score is a number',
              isinstance(ev[method].get('mean_score_negative'), (int, float)))
    check('eval: test_with_librosa >= 90',
          ev['counts']['test_with_librosa'] >= 90,
          f"{ev['counts'].get('test_with_librosa')}")
    check('eval: negatives_with_librosa >= 10',
          ev['counts']['negatives_with_librosa'] >= 10)
    check('eval: genre sanity accuracy >= 0.3',
          ev['genre_sanity']['accuracy'] >= 0.3,
          f"{ev['genre_sanity'].get('accuracy')}")
except FileNotFoundError:
    check('eval: file exists', False, 'evaluation_v2.json missing')

# --- 5. GTZAN sanity
try:
    gz = load_json(os.path.join(DATA, 'gtzan_check.json'))
    check('gtzan: status done', gz.get('status') == 'done',
          f"status={gz.get('status')}")
    check('gtzan: accuracy >= 0.3 (chance 0.10)',
          gz.get('accuracy', 0) >= 0.3, f"acc={gz.get('accuracy')}")
except FileNotFoundError:
    check('gtzan: file exists', False, 'gtzan_check.json missing')

# --- 6. recommendations (RED until recommend_v2.py recommend runs)
try:
    recs = load_json(os.path.join(BASE, 'recommendations_v2.json'))
    check('recs: exactly 10', len(recs) == 10, f'got {len(recs)}')
    liked = id_set(LIKED_FILE)
    hist = id_set(HIST_FILE)
    sids = [r['spotify_id'] for r in recs]
    check('recs: all have spotify_id', all(sids))
    check('recs: no duplicates', len(set(sids)) == len(sids))
    check('recs: none in Liked Songs', not (set(sids) & liked),
          f"overlap={set(sids) & liked}")
    check('recs: none in recommendation history', not (set(sids) & hist),
          f"overlap={set(sids) & hist}")
    check('recs: all have v2 + v6 scores',
          all(isinstance(r.get('v2_score'), (int, float))
              and isinstance(r.get('v6_score_per_bucket'), (int, float))
              for r in recs))
except FileNotFoundError:
    check('recs: file exists', False, 'recommendations_v2.json missing')

# --- 7. report
try:
    rep = open(os.path.join(BASE, 'evaluation_report_v2.md')).read()
    check('report: mentions preview-only limit', 'preview' in rep.lower())
    check('report: mentions no full-length audio', 'full-length' in rep.lower())
    check('report: mentions 13 negatives', '13' in rep)
    check('report: mentions H-bucket pattern', 'H bucket' in rep or 'H-bucket' in rep)
    check('report: has metrics', 'P@10' in rep and 'NDCG' in rep)
except FileNotFoundError:
    check('report: file exists', False, 'evaluation_report_v2.md missing')

print(f'\n{len(passes)} passed, {len(failures)} failed')
if failures:
    print('FAILING:', ', '.join(failures))
    sys.exit(1)
print('ALL ACCEPTANCE CHECKS GREEN')
