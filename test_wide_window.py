#!/usr/bin/env python3
"""Tests for the widened verify window (top 100) and the empty-list guard
in recommend_v2.recommend().

Regression: the v10 generation (Nov 22, 2026) found the MMR top 30 fully
consumed by v7/v8/v9 -- the verify loop dropped all 30 as already
liked/history, leaving 0 clean picks, and recommend() then crashed with
ValueError: not enough values to unpack at the rescore step (zip(*) over
the empty verified list). Two changes:
  1. MMR-select and verify the top 100 instead of the top 30
     (same ranking, wider window).
  2. When the verify loop leaves nothing, recommend() logs
     "no clean candidates" and returns [] instead of raising.

Tests use the real pool and the real v2 model, with Spotify search
stubbed out (no network). Each call writes to its own temporary directory.

Run:
  ~/workspace/audio-env/bin/python test_wide_window.py
Exit 0 = every check green. Exit 1 = at least one check red (listed).
"""
import contextlib, io, os, sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recommend_v2 as R

BASE = os.path.expanduser('~/workspace/audio-pipeline')

failures = []
passes = []


def check(name, cond, detail=''):
    if cond:
        passes.append(name)
        print(f'PASS  {name}')
    else:
        failures.append(name)
        print(f'FAIL  {name}  {detail}')


def isolated_recommend(**stubs):
    with TemporaryDirectory() as tmp, patch.multiple(R, BASE=tmp, **stubs):
        os.symlink(os.path.join(BASE, 'features_v2.json'), os.path.join(tmp, 'features_v2.json'))
        return R.recommend()


def fill_reccobeats(cands):
    """Give stub-verified candidates a non-empty reccobeats dict so the
    real recommend() skips its live ReccoBeats fill (no network)."""
    for c in cands:
        if not c.get('reccobeats'):
            c['reccobeats'] = {'danceability': 0.5}
        c['spotify_id'] = c.get('spotify_id') or f"TEST{c['deezer_id']}"
    return cands


# --- 1. window width: recommend() verifies 100 candidates, not 30 ---
recorded = {}


def stub_verify_wide(cands, *args, **kwargs):
    recorded['n'] = len(cands)
    return fill_reccobeats([dict(c) for c in cands[:10]])


try:
    result = isolated_recommend(verify_candidates=stub_verify_wide)
    check('window: verify saw 100 candidates',
          recorded.get('n') == 100, f"got {recorded.get('n')}")
    check('window: 10 final picks from the widened window',
          isinstance(result, list) and len(result) == 10,
          f"got {len(result) if isinstance(result, list) else type(result)}")
except Exception as e:
    check('window: recommend() ran without raising', False, f'{type(e).__name__}: {e}')

# --- 2. empty: no clean candidates is reported, not a crash ---
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), patch.object(R.time, 'sleep'):
        result = isolated_recommend(spotify_search_id=lambda title, artist: None)
    out = buf.getvalue()
    check('empty: recommend() returns [] instead of raising',
          result == [], f'got {result!r}'[:200])
    check('empty: "no clean candidates" is reported',
          'no clean candidates' in out)
except ValueError as e:
    check('empty: recommend() returns [] instead of raising', False,
          f'ValueError: {e}')
except Exception as e:
    check('empty: recommend() returns [] instead of raising', False,
          f'{type(e).__name__}: {e}')

# --- 3. short list: fewer than 10 verified still returns without crashing ---


def stub_verify_five(cands, *args, **kwargs):
    return fill_reccobeats([dict(c) for c in cands[:5]])


try:
    result = isolated_recommend(verify_candidates=stub_verify_five)
    check('short: 5 verified yields 5 final picks, no crash',
          isinstance(result, list) and len(result) == 5,
          f"got {len(result) if isinstance(result, list) else type(result)}")
except Exception as e:
    check('short: 5 verified yields 5 final picks, no crash', False,
          f'{type(e).__name__}: {e}')

print(f'\n{len(passes)} passed, {len(failures)} failed')
if failures:
    print('FAILING:', ', '.join(failures))
    sys.exit(1)
print('ALL WIDE-WINDOW CHECKS GREEN')
