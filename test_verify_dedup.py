#!/usr/bin/env python3
"""Tests for the recording-key dedup in recommend_v2.verify_candidates.

Regression: during the v9 generation (Nov 22, 2026) two tracks Kevin had
already dropped slipped back in under different Spotify IDs, because the
verify loop deduped by Spotify track ID only:
  - "af1 (sped up)" by Lilbubblegum (v7 drop) re-appeared as "af1"
  - "A Milli" by Lil Wayne (v8 drop) re-appeared under a new ID
The script already computes recording keys (cleaned title + first artist)
as liked_keys / hist_keys in load_id_sets(), but the verify loop never
used them. A candidate must be rejected when its recording key matches a
liked or previously recommended recording, even when its Spotify ID is new.

Run:
  ~/workspace/audio-env/bin/python test_verify_dedup.py
Exit 0 = every check green. Exit 1 = at least one check red (listed).
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recommend_v2 as R

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')

failures = []
passes = []


def check(name, cond, detail=''):
    if cond:
        passes.append(name)
        print(f'PASS  {name}')
    else:
        failures.append(name)
        print(f'FAIL  {name}  {detail}')


def key(title, artist):
    return (R.clean(title), R.clean(artist.split(',')[0]))


# --- fabricated ID sets: deterministic, no network, no real files ---
LIKED_IDS = {'LIKEDID0001'}
LIKED_KEYS = {key('A Milli (Album Version Explicit)', 'Lil Wayne')}
HIST_IDS = {'HISTID0001'}
HIST_KEYS = {key('af1 (sped up)', 'Lilbubblegum')}

# 1. history key match with a fresh Spotify ID must be rejected
check('key: history match with fresh ID is rejected',
      R.is_already_known('FRESHID0001', 'af1 (sped up)', 'Lilbubblegum',
                         LIKED_IDS, LIKED_KEYS, HIST_IDS, HIST_KEYS))

# 2. liked key match with a fresh Spotify ID must be rejected
check('key: liked match with fresh ID is rejected',
      R.is_already_known('FRESHID0002', 'A Milli (Album Version Explicit)',
                         'Lil Wayne',
                         LIKED_IDS, LIKED_KEYS, HIST_IDS, HIST_KEYS))

# 3. Spotify ID match is still rejected (pre-existing behaviour)
check('id: history ID match is rejected',
      R.is_already_known('HISTID0001', 'Some Other Title', 'Someone Else',
                         LIKED_IDS, LIKED_KEYS, HIST_IDS, HIST_KEYS))
check('id: liked ID match is rejected',
      R.is_already_known('LIKEDID0001', 'Some Other Title', 'Someone Else',
                         LIKED_IDS, LIKED_KEYS, HIST_IDS, HIST_KEYS))

# 4. fresh ID and fresh key is kept
check('clean: fresh ID and fresh key is kept',
      not R.is_already_known('FRESHID0003', 'Brand New Song', 'New Artist',
                             LIKED_IDS, LIKED_KEYS, HIST_IDS, HIST_KEYS))

# 5. only the first artist counts in the key
check('key: first artist only',
      R.is_already_known('FRESHID0004', 'af1 (sped up)',
                         'Lilbubblegum, Guest Vocalist',
                         LIKED_IDS, LIKED_KEYS, HIST_IDS, HIST_KEYS))
check('key: second artist alone does not match',
      not R.is_already_known('FRESHID0005', 'af1 (sped up)',
                             'Guest Vocalist, Lilbubblegum',
                             LIKED_IDS, LIKED_KEYS, HIST_IDS, HIST_KEYS))

# 6. verify_candidates end to end with a stubbed search (no network)
def stub_search(title, artist):
    return {'af1 (sped up)': 'FRESHID0001',
            'A Milli (Album Version Explicit)': 'FRESHID0002',
            'Brand New Song': 'FRESHID0003'}.get(title)

cands = [
    {'title': 'af1 (sped up)', 'artist': 'Lilbubblegum', 'deezer_id': 1},
    {'title': 'A Milli (Album Version Explicit)', 'artist': 'Lil Wayne',
     'deezer_id': 2},
    {'title': 'Brand New Song', 'artist': 'New Artist', 'deezer_id': 3},
]
verified = R.verify_candidates(cands, stub_search,
                               LIKED_IDS, LIKED_KEYS, HIST_IDS, HIST_KEYS)
check('verify: only the clean candidate survives',
      [c['title'] for c in verified] == ['Brand New Song'],
      f"got {[c['title'] for c in verified]}")
check('verify: surviving candidate carries its spotify_id',
      verified and verified[0].get('spotify_id') == 'FRESHID0003')

# 7. real data: the exact v9 incident. The pool records for the two
#    re-recommended tracks carry keys that are already in history.
try:
    liked_ids, liked_keys, hist_ids, hist_keys = R.load_id_sets()
    pool = json.load(open(os.path.join(DATA, 'pool_features.json')))
    by_key = {}
    for p in pool:
        by_key[key(p.get('title', ''), p.get('artist', ''))] = p
    af1 = by_key.get(key('af1 (sped up)', 'Lilbubblegum'))
    milli = by_key.get(key('A Milli (Album Version Explicit)', 'Lil Wayne'))
    check('real: af1 pool record exists', af1 is not None)
    check('real: af1 key is in history keys',
          af1 is not None and key(af1['title'], af1['artist']) in hist_keys)
    check('real: A Milli pool record exists', milli is not None)
    check('real: A Milli key is in history keys',
          milli is not None and key(milli['title'], milli['artist']) in hist_keys)
    if af1 is not None:
        check('real: af1 pool record rejected by key with a fresh ID',
              R.is_already_known('FRESHID9999', af1['title'], af1['artist'],
                                 liked_ids, liked_keys, hist_ids, hist_keys))
    if milli is not None:
        check('real: A Milli pool record rejected by key with a fresh ID',
              R.is_already_known('FRESHID9998', milli['title'], milli['artist'],
                                 liked_ids, liked_keys, hist_ids, hist_keys))
except FileNotFoundError as e:
    check('real: files exist', False, str(e))

print(f'\n{len(passes)} passed, {len(failures)} failed')
if failures:
    print('FAILING:', ', '.join(failures))
    sys.exit(1)
print('ALL VERIFY-DEDUP CHECKS GREEN')
