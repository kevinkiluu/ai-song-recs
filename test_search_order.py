#!/usr/bin/env python3
"""Tests for the search-hit selection in recommend_v2.pick_search_hit.

Regression: during the v9 generation (Nov 22, 2026) the verify step's
spotify_search_id returned the first track URI in document order instead
of the exact title match. For "drift" by Lilbubblegum it returned "af1"
(spotify:track:0pFz5fQBkspNYZGKBgVfdl, the 2nd hit) instead of "drift"
(spotify:track:69zSQXthMS87H2AvfnMWvd, the 1st hit and the exact title
match), putting the wrong track in the first v9 playlist.

Rule: prefer an exact title match (cleaned, case-insensitive) over
document order; among exact matches only one whose first artist agrees
is returned. Drop the candidate when no hit agrees on both fields.

The fixtures below mirror the real Spotify search response shape
(sections -> items with title / subtitle / spotify_uri), with the exact
match deliberately NOT first in document order. No network is used.

Run:
  ~/workspace/audio-env/bin/python test_search_order.py
Exit 0 = every check green. Exit 1 = at least one check red (listed).
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recommend_v2 as R

failures = []
passes = []


def check(name, cond, detail=''):
    if cond:
        passes.append(name)
        print(f'PASS  {name}')
    else:
        failures.append(name)
        print(f'FAIL  {name}  {detail}')


def item(title, artist, sid):
    return {'title': title, 'subtitle': artist,
            'spotify_uri': f'spotify:track:{sid}'}


def resp(items):
    return {'sections': [{'items': items}]}


# Real v9 case, reordered so the exact match is NOT first: af1 first,
# drift second, plus a near-miss ("Drift Barbie") that must not win.
DRIFT_RESP = resp([
    item('af1', 'lilbubblegum', '0pFz5fQBkspNYZGKBgVfdl'),
    item('drift', 'lilbubblegum', '69zSQXthMS87H2AvfnMWvd'),
    item('Drift Barbie', 'blueberry, PiNKII', '2jwideUDj7G4SsyuYkrJWn'),
    item('af1 (sped up)', 'lilbubblegum', '1HmUGju9HonKgT157b0DLB'),
])

check('drift: exact match wins over document order',
      R.pick_search_hit('drift', 'Lilbubblegum', DRIFT_RESP)
      == '69zSQXthMS87H2AvfnMWvd',
      f"got {R.pick_search_hit('drift', 'Lilbubblegum', DRIFT_RESP)}")
check('drift: does not return af1',
      R.pick_search_hit('drift', 'Lilbubblegum', DRIFT_RESP)
      != '0pFz5fQBkspNYZGKBgVfdl')

# Title comparison is case-insensitive and normalises the same way as
# the dedup keys (clean): a differently cased query still finds it.
check('drift: case-insensitive title match',
      R.pick_search_hit('Drift', 'Lilbubblegum', DRIFT_RESP)
      == '69zSQXthMS87H2AvfnMWvd')

# Among exact title matches, first-artist agreement wins.
TWO_MILLI = resp([
    item('A Milli', 'Other Artist', 'OTHERID0001'),
    item('A Milli', 'Lil Wayne', '1LDwUN1jMnTK7vCZBFoTYe'),
])
check('tie-break: artist agreement wins among exact title matches',
      R.pick_search_hit('A Milli', 'Lil Wayne', TWO_MILLI)
      == '1LDwUN1jMnTK7vCZBFoTYe')

# Without an exact title match, the candidate has no verified recording.
NO_MATCH = resp([
    item('Some Other Song', 'Someone', 'FIRSTID0001'),
    item('Another Song', 'Someone Else', 'SECONDID0002'),
])
check('no matching title: unrelated hit is rejected',
      R.pick_search_hit('drift', 'Lilbubblegum', NO_MATCH) is None)

# Exact title matches exist but none agrees on artist: return None
# rather than a different artist's same-title track. Mirrors the live
# "drift (feat. Lames)" response, which returns only other artists'
# "Drift" recordings.
WRONG_ARTIST = resp([
    item('Drift', 'James Leeluck', '6tlFACf9nEplYqUAr4ppBG'),
    item('Drift', 'Emil Rottmayer', '3N4QVGaqqta8L7HqwuOG0p'),
])
check('no artist agreement: exact matches by other artists return None',
      R.pick_search_hit('drift (feat. Lames)', 'Lilbubblegum', WRONG_ARTIST)
      is None)

# No hits at all: None.
check('empty: no hits returns None',
      R.pick_search_hit('drift', 'Lilbubblegum', resp([])) is None)
check('empty: no sections returns None',
      R.pick_search_hit('drift', 'Lilbubblegum', {}) is None)

print(f'\n{len(passes)} passed, {len(failures)} failed')
if failures:
    print('FAILING:', ', '.join(failures))
    sys.exit(1)
print('ALL SEARCH-ORDER CHECKS GREEN')
