#!/usr/bin/env python3
"""TDD acceptance tests for the Troi liked-only patch (spec section c).

Written FIRST, before troi_patch.py exists. Each test must FAIL on the
empty implementation (ImportError), then pass after implementation.
Run: ~/workspace/audio-env/bin/python -m pytest test_troi_patch.py -v
Root-level file (v2 convention), NOT tests/ -- deviation from spec (b.1)
recorded in the build report.
Seed 42 everywhere randomness is involved.
"""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.expanduser('~/workspace/audio-pipeline'))

import troi_patch as tp  # noqa: E402  (RED: ModuleNotFoundError before implementation)

BASE = os.path.expanduser('~/workspace/audio-pipeline')
LIKED_FILE = os.path.expanduser('~/workspace/spotify-liked-songs.txt')
HIST_FILE = os.path.expanduser('~/workspace/spotify-recommended-tracks.txt')


# ---------------------------------------------------------------- helpers
def rec(title, artist, sid=None, ranking=None, mode=None, stream=None,
        score=None):
    return tp.Recording(title=title, artist=artist, spotify_id=sid,
                        ranking=ranking, mode=mode, stream=stream,
                        score=score)


def load_liked():
    out = []
    for line in open(LIKED_FILE):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3:
            out.append((p[0], p[1], p[2].replace('spotify:track:', '')))
    return out


def load_history():
    out = []
    for line in open(HIST_FILE):
        if line.startswith('#'):
            continue
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3:
            out.append((p[0], p[1], p[2].replace('spotify:track:', '')))
    return out


NEGATIVE_IDS = [
    '6aJn7Cst74cj4lNIiPRgav', '6HacgXCExkzS552ILfJTXu',
    '1GfBLbAhZUWdseuDqhocmn', '5v7oj9PosKdE39x9uwDs6C',
    '4bkPAARkDov1lV05emEF7a', '2jv1ovpDpTJEPIlth5Wn07',
    '7zDu3l7DDlfSlt20YMKkWo', '2y57H6nA3vic5t8awwaS69',
    '2slLXFcITuhf0oSp2wVJLv', '30efCAopZCFcoi43sSToNr',
    '0FmrNFYyLLRtIS7XWJPLLk', '32gSPav2xqFylAeCjJCxWi',
    '1vvBUZseJ7fwYPX1NedOLd',
]


# ---------------------------------------------------------------- T-01
def test_patch_chain_assembles_in_order():
    """T-01: patch stages run in declared order; generate() -> Playlist(10)."""
    items = [rec(f'Song {i}', f'Artist {i}', sid=f'id{i:02d}', ranking=i + 1,
                 score=float(10 - i)) for i in range(30)]
    s1 = tp.StreamElement(items, name='S1')
    f1 = tp.DifferenceElement(liked_ids=set(), liked_keys=set(),
                              history_ids=set(), history_keys=set())
    f3 = tp.DuplicateFilterElement()
    f5 = tp.ArtistCreditLimiterElement(count=2)
    z = tp.ZipperElement(weights=[1])
    red = tp.PlaylistRedundancyReducerElement(length=10)
    patch = tp.Patch([s1, f1, f3, f5, z, red])
    assert patch.stages == [s1, f1, f3, f5, z, red]
    out = patch.generate()
    assert isinstance(out, tp.Playlist)
    assert len(out.recordings) == 10


# ---------------------------------------------------------------- T-02
def test_duplicate_filters_remove_both_kinds():
    """T-02: same Spotify ID twice + same (title, artist) under two IDs."""
    cands = [
        rec('Song A', 'Artist X', sid='aaa'),
        rec('Song A', 'Artist X', sid='aaa'),          # dup by ID
        rec('Song B', 'Artist Y', sid='bbb'),
        rec('Song B', 'Artist Y', sid='ccc'),          # dup by title+artist
        rec('Song C', 'Artist Z', sid='ddd'),
    ]
    out = tp.DuplicateFilterElement().read(cands)
    assert [r.spotify_id for r in out] == ['aaa', 'bbb', 'ddd']
    # first-seen order preserved
    assert [r.title for r in out] == ['Song A', 'Song B', 'Song C']


# ---------------------------------------------------------------- T-03
def test_difference_removes_all_liked():
    """T-03: 50 real liked tracks + 50 non-liked -> 0 liked IDs survive F1."""
    liked = load_liked()
    assert len(liked) == 650
    cands = [rec(t, a, sid=s) for t, a, s in liked[:50]]
    cands += [rec(f'Fresh {i}', f'New Artist {i}', sid=f'fresh{i:02d}')
              for i in range(50)]
    liked_ids = {s for _, _, s in liked}
    liked_keys = {(tp.clean(t), tp.clean(a.split(',')[0]))
                  for t, a, _ in liked}
    out = tp.DifferenceElement(liked_ids, liked_keys, set(), set()).read(cands)
    assert len(out) == 50
    assert not ({r.spotify_id for r in out} & liked_ids)


# ---------------------------------------------------------------- T-04
def test_difference_removes_all_history():
    """T-04: all 60 history entries + 40 fresh -> 0 history IDs survive F1."""
    hist = load_history()
    assert len(hist) == 60
    cands = [rec(t, a, sid=s) for t, a, s in hist]
    cands += [rec(f'Fresh {i}', f'New Artist {i}', sid=f'fresh{i:02d}')
              for i in range(40)]
    hist_ids = {s for _, _, s in hist}
    hist_keys = {(tp.clean(t), tp.clean(a.split(',')[0]))
                 for t, a, _ in hist}
    out = tp.DifferenceElement(set(), set(), hist_ids, hist_keys).read(cands)
    assert len(out) == 40
    assert not ({r.spotify_id for r in out} & hist_ids)


# ---------------------------------------------------------------- T-05
def test_hated_filter_removes_all_13_negatives():
    """T-05: all 13 negatives by ID + 'prove it' under a fake new ID."""
    assert len(NEGATIVE_IDS) == 13
    negs = json.load(open(os.path.join(BASE, 'data', 'negatives_features.json')))
    by_id = {r['spotify_id']: r for r in negs}
    cands = []
    for sid in NEGATIVE_IDS:
        r = by_id[sid]
        cands.append(rec(r['title'], r['artist'], sid=sid))
    # re-release of 'prove it' under a fake new ID, identical title/artist
    cands.append(rec('prove it', '21 Savage, Summer Walker', sid='FAKE0001'))
    cands.append(rec('Innocent Song', 'Somebody Else', sid='ok0001'))
    out = tp.HatedRecordingsFilterElement(NEGATIVE_IDS).read(cands)
    assert [r.spotify_id for r in out] == ['ok0001']


# ---------------------------------------------------------------- T-06
def test_artist_cap_is_2_drop_lowest_ranked():
    """T-06: one artist, 4 candidates ranked 1-4 by audio fit -> keep 1,2."""
    cands = [rec(f'Track {i}', 'Same Artist', sid=f's{i}', ranking=i,
                 score=float(10 - i)) for i in [1, 2, 3, 4]]
    cands += [rec('Other', 'Other Artist', sid='o1', ranking=1, score=9.0)]
    out = tp.ArtistCreditLimiterElement(count=2).read(cands)
    kept = sorted(r.ranking for r in out if r.artist == 'Same Artist')
    assert kept == [1, 2]
    assert len(out) == 3


# ---------------------------------------------------------------- T-07
def test_zipper_weights_3_2_1():
    """T-07: three synthetic streams, weights 3:2:1 -> exact interleave."""
    a = [rec(f'A{i}', f'Artist A{i}', sid=f'a{i}') for i in range(6)]
    b = [rec(f'B{i}', f'Artist B{i}', sid=f'b{i}') for i in range(6)]
    c = [rec(f'C{i}', f'Artist C{i}', sid=f'c{i}') for i in range(6)]
    z = tp.ZipperElement(weights=[3, 2, 1])
    z.set_sources([tp.StreamElement(a), tp.StreamElement(b),
                   tp.StreamElement(c)])
    out = z.read()
    assert [r.title for r in out[:12]] == [
        'A0', 'A1', 'A2', 'B0', 'B1', 'C0',
        'A3', 'A4', 'A5', 'B2', 'B3', 'C1']


# ---------------------------------------------------------------- T-08
def test_latest_listened_filter_noop_without_timestamps():
    """T-08: liked-only run -> output == input; log line present."""
    cands = [rec(f'Song {i}', f'Artist {i}', sid=f'x{i}') for i in range(5)]
    filt = tp.LatestListenedAtFilterElement(min_number_of_days=14)
    log = {}
    out = filt.read(cands, log=log)
    assert [r.spotify_id for r in out] == [r.spotify_id for r in cands]
    assert log.get('latest_listened_at_filter') == (
        'skipped, reason=no timestamps (n=0 with data)')


# ---------------------------------------------------------------- T-09
def test_mode_tertiles():
    """T-09: ordered 90 items -> easy ranks 1-30, medium 31-60, hard 61-90."""
    items = [rec(f'Track {i:02d}', f'Artist {i:02d}', sid=f't{i:02d}',
                 ranking=i + 1) for i in range(90)]
    easy = tp.apply_mode(items, 'easy')
    med = tp.apply_mode(items, 'medium')
    hard = tp.apply_mode(items, 'hard')
    assert [r.ranking for r in easy] == list(range(1, 31))
    assert [r.ranking for r in med] == list(range(31, 61))
    assert [r.ranking for r in hard] == list(range(61, 91))


# ---------------------------------------------------------------- T-10 / T-11
def _synthetic_streams():
    def mk(prefix, n, mode):
        return [rec(f'{prefix} Song {i}', f'{prefix} Artist {i}',
                    sid=f'{prefix}{i:02d}', ranking=i + 1, mode=mode,
                    stream=prefix, score=float(10 - i)) for i in range(n)]
    return {
        'audio-neighbour:easy': mk('ANE', 10, 'easy'),
        'audio-neighbour:medium': mk('ANM', 10, 'medium'),
        'audio-neighbour:hard': mk('ANH', 10, 'hard'),
        'similar-artist:easy': mk('SAE', 10, 'easy'),
        'same-artist:easy': mk('SME', 10, 'easy'),
    }


def test_exploration_has_3_hard_in_slots_8_to_10():
    """T-10: Exploration slots 8-10 are hard-mode; slots 1-7 have 0 hard."""
    out = tp.generate_playlists(_synthetic_streams(), seed=42)
    expl = out['exploration'].recordings
    assert len(expl) == 10
    assert all(r.mode == 'hard' for r in expl[7:10])
    assert all(r.mode != 'hard' for r in expl[:7])


def test_jams_and_exploration_disjoint():
    """T-11: one run, both outputs length 10, disjoint IDs."""
    out = tp.generate_playlists(_synthetic_streams(), seed=42)
    jams = out['jams'].recordings
    expl = out['exploration'].recordings
    assert len(jams) == 10 and len(expl) == 10
    assert not ({r.spotify_id for r in jams} &
                {r.spotify_id for r in expl})


# ---------------------------------------------------------------- T-12
def test_determinism_seed_42():
    """T-12: generate() twice, same inputs, seed 42 -> byte-identical."""
    streams = _synthetic_streams()
    o1 = tp.generate_playlists(streams, seed=42)
    o2 = tp.generate_playlists(streams, seed=42)
    for key in ('jams', 'exploration'):
        ids1 = [r.spotify_id for r in o1[key].recordings]
        ids2 = [r.spotify_id for r in o2[key].recordings]
        assert ids1 == ids2


# ---------------------------------------------------------------- T-13
@pytest.mark.xfail(reason='history gate closed: no streaming export; '
                         'pure skip-proxy definition only, not wired into '
                         'generate()', strict=False)
def test_skip_proxy_definition():
    """T-13 (history-gated): X played 3x at 12s, 0 completions -> negative;
    Y played 1x at 200s -> not negative."""
    plays = [
        {'title': 'X', 'artist': 'X Artist', 'ms_played': 12000},
        {'title': 'X', 'artist': 'X Artist', 'ms_played': 12000},
        {'title': 'X', 'artist': 'X Artist', 'ms_played': 12000},
        {'title': 'Y', 'artist': 'Y Artist', 'ms_played': 200000},
    ]
    negs = tp.skip_proxy_negatives(plays)
    assert ('x', 'x artist') in negs
    assert ('y', 'y artist') not in negs


# ---------------------------------------------------------------- T-14
def test_resource_budget():
    """T-14: full default run over the 300-candidate cap: <600s, <4.5GB."""
    import resource
    t0 = time.time()
    out, log, stats = tp.run_default_patch()
    wall = time.time() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    assert wall < 600, f'wall {wall:.1f}s exceeds 600s budget'
    assert peak_gb < 4.5, f'peak RSS {peak_gb:.2f}GB exceeds 4.5GB budget'
    assert len(out['jams'].recordings) == 10
    assert len(out['exploration'].recordings) == 10
    stats['wall_s'] = round(wall, 1)
    stats['peak_rss_gb'] = round(peak_gb, 2)


# ---------------------------------------------------------------- T-15
def test_prompt_parser():
    """T-15: Troi docs example parses; omitted weight defaults to 1."""
    terms = tp.parse_prompt(
        'artist:(pretty lights):3:easy tag:(trip hop):2 '
        'artist:(morcheeba)::nosim')
    assert terms == [
        ('artist', ['pretty lights'], 3, 'easy'),
        ('tag', ['trip hop'], 2, None),
        ('artist', ['morcheeba'], 1, 'nosim'),
    ]
    terms2 = tp.parse_prompt(tp.DEFAULT_PROMPT)
    assert terms2[0] == ('audio-neighbour', ['liked-seeds'], 3, 'easy')
    assert terms2[1] == ('similar-artist', ['top20'], 2, 'easy')
    assert terms2[2] == ('same-artist', ['top20'], 1, 'easy')


# ---------------------------------------------------------------- T-16
def test_no_leakage_into_eval_pool():
    """T-16: held-out liked IDs (positions <130) in neither the seed-artist
    input nor the S3 neighbour fit."""
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    heldout_ids = {t['spotify_id'] for t in doc['tracks']
                   if t['position'] < 130 and t['spotify_id'] != 'None'}
    assert len(heldout_ids) == 129  # 130 positions, 1 unresolvable 'None' id
    _, seed_ids = tp.build_seed_artists()
    assert not (heldout_ids & seed_ids)
    _, _, fit_ids = tp.fit_audio_neighbours()
    assert not (heldout_ids & fit_ids)


# ---------------------------------------------------------------- extra guard
def test_eval_pool_reproduces_v2_baselines():
    """Eval-pool guard: rebuilding the identical 311-item pool with
    evaluate.py's functions must reproduce evaluation_v2.json exactly
    (v2 P@10 0.50, NDCG@10 0.726; v6-per-bucket 0.50/0.666;
    v6-global 0.60/0.664)."""
    res = tp.evaluate_patch(write=False)
    assert res['pool']['n_ranked'] == 311
    assert res['pool']['test_with_librosa'] == 99
    assert res['pool']['negatives_with_librosa'] == 12
    assert res['pool']['distractors'] == 200
    assert res['methods']['v2']['P@10'] == 0.5
    assert res['methods']['v2']['legacy_topk_hit_ndcg'] == 0.726
    assert res['methods']['v6_per_bucket']['P@10'] == 0.5
    assert res['methods']['v6_per_bucket']['legacy_topk_hit_ndcg'] == 0.666
    assert res['methods']['v6_global']['P@10'] == 0.6
    assert res['methods']['v6_global']['legacy_topk_hit_ndcg'] == 0.664


# ---------------------------------------------------------------- decontamination + 4-leg
def test_eval_decontamination_removes_duplicate_audio():
    """5 held-out tracks share a Deezer preview ID with a train track
    (same audio, different Spotify ID: Warriors, CRAZY, Red Rum,
    [Gen]inside, Partyy). Decontaminated pool is 306 items; Troi
    P@10 0.80 / NDCG@10 0.82 (raw 0.90/0.99 was inflated); v2 stays
    0.50/0.726 (its 5 dups ranked 49-218, so v2 was not inflated)."""
    res = tp.evaluate_patch(write=False)
    dec = res['decontaminated']
    assert dec['pool']['n_dup_audio_removed'] == 5
    assert dec['pool']['n_ranked'] == 306
    assert dec['methods']['troi']['P@10'] == 0.8
    assert dec['methods']['troi']['legacy_topk_hit_ndcg'] == 0.82
    assert dec['methods']['troi']['NDCG@10'] == 0.714
    assert dec['methods']['v2']['P@10'] == 0.5
    assert dec['methods']['v2']['legacy_topk_hit_ndcg'] == 0.726
    assert sorted(t['title'] for t in dec['removed_items']) == sorted([
        'Warriors', 'CRAZY', 'Red Rum', '[Gen]inside', 'Partyy'])


def test_four_leg_verdict_is_no_go_offline():
    """Eval-spec section (d): all four legs must pass. On decontaminated
    numbers Leg 1 fails (bootstrap CI for delta_NDCG includes 0) and
    Leg 3 fails (mean negative rank 146.7 < random 153.5); Leg 2 passes;
    Leg 4 is unmeasured. Verdict is NO-GO on offline evidence."""
    res = tp.evaluate_patch(write=False)
    legs = res['four_leg']
    assert legs['leg1_offline_beats_v6']['passed'] is False
    assert legs['leg2_beats_chance']['passed'] is True
    assert legs['leg3_negative_control']['passed'] is False
    assert legs['leg4_live_keep_rate']['passed'] is None
    assert res['verdict'].startswith('NO-GO')


def test_paired_bootstrap_ci():
    """P-04: B=10,000 stratified paired bootstrap, seed 42. Independently
    computed CI: delta_NDCG@10 [-0.154, 0.785], delta_P@10 [-0.100, 0.800],
    P@10(troi) [0.400, 1.000]. The delta_NDCG CI includes 0."""
    res = tp.evaluate_patch(write=False)
    pb = res['paired_bootstrap']
    assert pb['B'] == 10000 and pb['seed'] == 42
    assert pb['delta_NDCG@10']['ci_95'][0] == pytest.approx(-0.154, abs=0.01)
    assert pb['delta_NDCG@10']['ci_95'][1] == pytest.approx(0.785, abs=0.01)
    assert pb['delta_P@10']['ci_95'][0] == pytest.approx(-0.100, abs=0.01)
    assert pb['P@10_troi']['ci_95'][0] == pytest.approx(0.400, abs=0.01)


def test_seed_sweep_five_distractor_draws():
    """P-04 repeated measurement: distractor seeds 42-46, min/median/max
    of delta_NDCG@10 and delta_P@10 (Troi minus v6-per-bucket)."""
    res = tp.evaluate_patch(write=False)
    sw = res['seed_sweep']
    assert sw['seeds'] == [42, 43, 44, 45, 46]
    for key in ('delta_P@10', 'delta_NDCG@10'):
        assert sw[key]['min'] <= sw[key]['median'] <= sw[key]['max']
