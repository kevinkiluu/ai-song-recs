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
stubbed out (no network). recommend() writes recommendations_v2.json on
the success path, so each call is wrapped in a snapshot/restore of that
file; the v9 list it holds is never left modified by the tests.

Run:
  ~/workspace/audio-env/bin/python test_wide_window.py
Exit 0 = every check green. Exit 1 = at least one check red (listed).
"""
import contextlib, io, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recommend_v2 as R

BASE = os.path.expanduser('~/workspace/audio-pipeline')
OUT_FILE = os.path.join(BASE, 'recommendations_v2.json')

failures = []
passes = []


def check(name, cond, detail=''):
    if cond:
        passes.append(name)
        print(f'PASS  {name}')
    else:
        failures.append(name)
        print(f'FAIL  {name}  {detail}')


def snapshot_out():
    with open(OUT_FILE, 'rb') as f:
        return f.read()


def restore_out(data):
    with open(OUT_FILE, 'wb') as f:
        f.write(data)


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
orig_verify = R.verify_candidates


def stub_verify_wide(cands, *args, **kwargs):
    recorded['n'] = len(cands)
    return fill_reccobeats([dict(c) for c in cands[:10]])


R.verify_candidates = stub_verify_wide
before = snapshot_out()
try:
    result = R.recommend()
    check('window: verify saw 100 candidates',
          recorded.get('n') == 100, f"got {recorded.get('n')}")
    check('window: 10 final picks from the widened window',
          isinstance(result, list) and len(result) == 10,
          f"got {len(result) if isinstance(result, list) else type(result)}")
except Exception as e:
    check('window: recommend() ran without raising', False, f'{type(e).__name__}: {e}')
finally:
    R.verify_candidates = orig_verify
    restore_out(before)

# --- 2. empty: no clean candidates is reported, not a crash ---
orig_search = R.spotify_search_id
orig_sleep = R.time.sleep
R.spotify_search_id = lambda title, artist: None
R.time.sleep = lambda s: None
before = snapshot_out()
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        result = R.recommend()
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
finally:
    R.spotify_search_id = orig_search
    R.time.sleep = orig_sleep
    restore_out(before)

# --- 3. short list: fewer than 10 verified still returns without crashing ---
orig_verify = R.verify_candidates


def stub_verify_five(cands, *args, **kwargs):
    return fill_reccobeats([dict(c) for c in cands[:5]])


R.verify_candidates = stub_verify_five
before = snapshot_out()
try:
    result = R.recommend()
    check('short: 5 verified yields 5 final picks, no crash',
          isinstance(result, list) and len(result) == 5,
          f"got {len(result) if isinstance(result, list) else type(result)}")
except Exception as e:
    check('short: 5 verified yields 5 final picks, no crash', False,
          f'{type(e).__name__}: {e}')
finally:
    R.verify_candidates = orig_verify
    restore_out(before)

print(f'\n{len(passes)} passed, {len(failures)} failed')
if failures:
    print('FAILING:', ', '.join(failures))
    sys.exit(1)
print('ALL WIDE-WINDOW CHECKS GREEN')
