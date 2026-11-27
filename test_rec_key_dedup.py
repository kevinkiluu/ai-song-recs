#!/usr/bin/env python3
"""Tests for folding remix variants into the dedup recording key.

Regression: the v2 generator re-offered 'Lemonade (feat. NAV)' in both v11
and v12 even though Kevin dropped 'Lemonade (Remix)' in v8. clean() strips
'(feat ...)' but keeps '(Remix)' as the word 'remix', so the two versions
of the same song got different keys ('lemonade' vs 'lemonade remix') and
the verify step let the original through. Each round needed a one-off
scratch swap script to fix it.

Fix: rec_key() folds a trailing 'remix' onto the base title, so a remix
and its original share one key. Other parentheticals are kept, so two
different songs that share a title stay distinct: 'Strategy (from the
Netflix film KPop Demon Hunters)' (in Liked Songs) is not the same song
as 'Strategy (feat. Megan Thee Stallion)' (offered in v12), and the
latter must still be offered.

Run:
  ~/workspace/audio-env/bin/python test_rec_key_dedup.py
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


rec_key = getattr(R, 'rec_key', None)
check('rec_key exists', callable(rec_key))

if callable(rec_key):
    check('rec_key folds remix onto base title',
          rec_key('Lemonade (Remix)', 'Internet Money')
          == rec_key('Lemonade (feat. NAV)', 'Internet Money')
          == ('lemonade', 'internet money'),
          f"got {rec_key('Lemonade (Remix)', 'Internet Money')} vs "
          f"{rec_key('Lemonade (feat. NAV)', 'Internet Money')}")
    check('rec_key keeps film parenthetical distinct',
          rec_key('Strategy (from the Netflix film KPop Demon Hunters)', 'TWICE')
          != rec_key('Strategy (feat. Megan Thee Stallion)', 'TWICE'),
          'film version and album version must not collide')
    check('rec_key strips feat markers',
          rec_key('Lemonade (feat. NAV)', 'Internet Money')
          == ('lemonade', 'internet money'))
    check('rec_key keeps plain titles unchanged',
          rec_key('VISIONS', 'eaJ') == ('visions', 'eaj'))

# is_already_known against the real sets from load_id_sets(), the same
# sets recommend() uses. Hand-building the sets would hide the bug: the
# old code built history keys with clean(), so a hand-built folded set
# would pass even before the fix.
liked_ids, liked_keys, hist_ids, hist_keys = R.load_id_sets()
check('original dropped when history holds the remix',
      R.is_already_known('1p0rEzrK7YtdRZVtiyV7RN', 'Lemonade (feat. NAV)',
                         'Internet Money', liked_ids, liked_keys,
                         hist_ids, hist_keys),
      'v11/v12 collision: Lemonade original must be dropped')
check('remix dropped when history holds the original',
      R.is_already_known('3gU7Pr1I4plN6QSxF6jFh3', 'Lemonade (Remix)',
                         'Internet Money', liked_ids, liked_keys,
                         hist_ids, hist_keys),
      'symmetric case: remix must be dropped too')
check('different same-title song stays offered',
      not R.is_already_known('5zQUndaoBtXJ10SsApqtvw',
                             'Strategy (feat. Megan Thee Stallion)',
                             'TWICE', liked_ids, liked_keys,
                             hist_ids, hist_keys),
      'KPop Demon Hunters Strategy must not block the album Strategy')
check('unrelated track stays offered',
      not R.is_already_known('0GA850AWIBdMW3Cfd4ptds', 'Reflex',
                             'SAINt JHN', liked_ids, liked_keys,
                             hist_ids, hist_keys))

# Live-data checks against the real files on disk.
check('load_id_sets folds the v8 Lemonade Remix history row',
      ('lemonade', 'internet money') in hist_keys,
      'history holds Lemonade (Remix); key must fold to lemonade')
check('load_id_sets keeps the film Strategy liked row distinct',
      ('strategy from the netflix film kpop demon hunters', 'twice')
      in liked_keys,
      'liked holds the KPop Demon Hunters Strategy; key must keep it')

# Cross-module consistency: every consumer that compares against
# load_id_sets() keys must build keys the same way, or a folded set key
# and an unfolded candidate key miss each other (that is what broke
# embeat T-14 when only recommend_v2 was fixed).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import troi_patch as TP

check('troi_patch norm_key folds remix like rec_key',
      TP.norm_key('Lemonade (Remix)', 'Internet Money')
      == rec_key('Lemonade (feat. NAV)', 'Internet Money')
      == ('lemonade', 'internet money'))
check('troi_patch norm_key keeps film parenthetical distinct',
      TP.norm_key('Strategy (from the Netflix film KPop Demon Hunters)',
                  'TWICE')
      != rec_key('Strategy (feat. Megan Thee Stallion)', 'TWICE'))

print(f'\n{len(passes)} passed, {len(failures)} failed')
if failures:
    print('FAILING:', ', '.join(failures))
    sys.exit(1)
print('ALL ACCEPTANCE CHECKS GREEN')
