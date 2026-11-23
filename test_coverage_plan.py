import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from audit import coverage as c


def sid(i):
    return f'{i:022}'


ISRC = 'USAB12345678'
AUDIO = {'bpm': 100, 'rms_mean': .3, 'brightness_mean': 1200}


def panel(i, code=ISRC):
    fields = {k: .5 for k in ['acousticness', 'danceability', 'energy', 'instrumentalness',
                              'liveness', 'speechiness', 'valence']}
    return dict(fields, id='provider-id', href=f'https://open.spotify.com/track/{sid(i)}',
                isrc=code, tempo=120, key=3, mode=1, loudness=-6)


def track(i, did=None, rb=None, lb=None):
    return dict(spotify_id=sid(i), position=i, deezer_id=did, reccobeats=rb, librosa=lb)


class CoveragePlanTests(unittest.TestCase):
    def test_panel_accepts_alternate_edition_only_with_independent_isrc(self):
        rows = [track(0), track(1), track(2), track(3), track(4), track(5)]
        panels = [panel(6), panel(2, 'USAB19876543'), dict(panel(3, 'USAA11234567'), tempo=None),
                  dict(panel(4), href='https://wrong.test/track/' + sid(4)),
                  panel(7, 'USXX12345678'), dict(panel(8, 'USXX12345678'), key=7)]
        written, counts = c.recover_panels(rows, {sid(i): ISRC for i in [0, 2]}
                                            | {sid(3): 'USAA11234567', sid(4): 'USZZ11234567', sid(5): 'USXX12345678'},
                                            lambda keys: panels, limit=6)
        self.assertEqual(set(written), {sid(0)})
        self.assertEqual(written[sid(0)]['isrc'], ISRC)
        self.assertEqual(counts['missing_independent_isrc'], 1)
        self.assertEqual(counts['conflicting_isrc'], 1)
        self.assertEqual(counts['ambiguous_panel'], 1)
        self.assertEqual(counts['incomplete_panel'], 1)

    def test_spotify_track_export_rejects_conflicting_crosswalks(self):
        rows = [{'id': sid(0), 'external_ids': {'isrc': ISRC.lower()}},
                {'id': 'None', 'external_ids': {'isrc': ISRC}}]
        self.assertEqual(c.spotify_isrcs(rows), {sid(0): ISRC})
        with self.assertRaises(ValueError):
            c.spotify_isrcs(rows + [{'id': sid(0), 'external_ids': {'isrc': 'USAB19876543'}}])

    def test_preview_recovery_checks_exact_id_and_separates_stages(self):
        rows = [track(i, i + 100) for i in range(6)] + [track(6, 100), track(7)]
        metadata = {100: {'id': 100, 'preview': 'one'}, 101: {'id': 101},
                    102: {'id': 998, 'preview': 'other'}, 103: {'id': 103, 'preview': 'bad'},
                    104: {'id': 104, 'preview': 'garbled'}}
        def download(url):
            if url == 'bad':
                raise OSError()
            return url.encode()
        def analyze(path):
            if Path(path).read_bytes() != b'one':
                raise ValueError()
            return AUDIO
        patches, stages = c.recover_previews(rows, lambda k: metadata[int(k)], download,
                                            analyze, limit=6)
        self.assertEqual(set(patches), {'100'})
        self.assertEqual(patches['100']['librosa'], AUDIO)
        self.assertEqual(stages, dict(redecoded=1, matched_no_preview=1, identity_mismatch=1,
                                      transfer_failure=1, decode_failure=1, metadata_unavailable=1))

    def test_failure_reasons_keep_unknowns_and_respect_frozen_match(self):
        rows = [track(i, i if i >= 2 else None) for i in range(6)]
        snapshot = {'tracks': rows, 'unmatched': [{'spotify_id': sid(0), 'reason': 'no match'},
                                                  {'spotify_id': sid(1), 'reason': 'no preview'}]}
        matches = {sid(2): {'deezer_id': 2, 'download_error': 'failed'},
                   sid(3): {'deezer_id': 99, 'download_error': 'failed'}}
        deep = {'4': {'error': 'analysis failed'}, '5': {'error': 'preview file missing'}}
        self.assertEqual(c.failure_stages(snapshot, matches, deep), dict(no_match=1,
                         matched_no_preview=1, transfer_failure=1, decode_failure=1,
                         unknown=2))

    def test_coverage_uses_fixed_positions_and_tracks_original_neither(self):
        rows = [track(132), track(1, 100), track(0, 101, lb=AUDIO), track(131, rb=panel(131))]
        old = copy.deepcopy(rows)
        rb = {sid(132): panel(132, 'USAA11234567'), sid(0): panel(0)}
        lb = {'100': {'librosa': AUDIO, 'preview_sha256': '1'*64}}
        new = c.apply_features(rows, rb, lb)
        report = c.cohorts(rows, new)
        self.assertEqual(rows, old)
        self.assertEqual(report['newest']['baseline']['rows'], 2)
        self.assertEqual(report['older']['after']['reccobeats'], 2)
        self.assertEqual(report['all']['baseline']['neither'], 2)
        self.assertEqual(report['all']['original_neither_gained_any_source'], 2)
        self.assertEqual(report['all']['new_reccobeats_recordings'], 2)

    def test_paired_rank_freezes_model_and_shared_rows(self):
        rows = [track(0, 10, lb=AUDIO), track(1, 11), track(2, 12), track(3, 13)]
        for i in range(4, 12):
            rows.append(track(i, i + 20, lb=dict(AUDIO, bpm=110+i, rms_mean=i/30)))
        rejects = [track(i, 100+i, lb=dict(AUDIO, bpm=i*5)) for i in range(20, 25)]
        catalog = [track(i, 100+i, lb=dict(AUDIO, bpm=220-i)) for i in range(30, 45)]
        new = c.apply_features(rows, {}, {'11': {'librosa': AUDIO, 'preview_sha256': 'a'*64}})
        result = c.paired_ranking(rows, new, rejects, catalog, cutoff=4)
        self.assertEqual(result['newly_evaluable'], 1)
        self.assertEqual(result['shared_pool'], 18)
        self.assertEqual(result['delta'], {'p_at_10': 0.0, 'ndcg_at_10': 0.0})
        self.assertEqual(result['baseline'], result['after'])

    def test_cache_binds_source_and_refuses_to_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'derived.json'
            c.save_cache(path, dict(version=1, features_sha256='a'*64, panels={}, previews={}))
            with self.assertRaises(FileExistsError):
                c.save_cache(path, {'version': 2})
            with self.assertRaises(ValueError):
                c.load_cache(path, 'b'*64, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(c.load_cache(path, 'a'*64, hashlib.sha256(path.read_bytes()).hexdigest())['version'], 1)

    def test_cli_recovery_replays_without_touching_frozen_source(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / 'features.json'
            cache = Path(tmp) / 'audit-v1.json'
            snap.write_text(json.dumps(dict(generated='2026-11-22', tracks=[track(0)], unmatched=[])))
            frozen = snap.read_bytes()
            command = [sys.executable, '-m', 'audit.coverage', '--features', str(snap)]
            live = subprocess.run(command + ['--recover', str(cache), '--rb-limit', '0',
                                             '--preview-limit', '0'], check=True, capture_output=True)
            digest = json.loads(live.stdout)['recovery_cache_sha256']
            replay = subprocess.run(command + ['--cache', str(cache), '--cache-sha256', digest], check=True, capture_output=True)
            output = json.loads(replay.stdout)
            self.assertEqual(json.loads(live.stdout), output)
            self.assertEqual(output['paired_ranking']['delta'], None)
            self.assertEqual(output['snapshot_generated'], '2026-11-22')
            self.assertEqual(output['cohorts']['all']['baseline']['neither'], 1)
            self.assertNotIn(sid(0), replay.stdout.decode())
            self.assertEqual(snap.read_bytes(), frozen)

    def test_out_of_range_provider_value_does_not_count_as_a_panel(self):
        self.assertFalse(c.panel_usable(dict(panel(1), energy=1.5)))
        self.assertFalse(c.panel_usable(dict(panel(1), tempo=-1)))

    def test_unpaired_recovery_is_not_reported_as_a_measured_rank_delta(self):
        result = c.paired_ranking([track(0)], [track(0)], None, [track(10, 10, lb=AUDIO)])
        self.assertEqual(result['scope'], 'partial supplied pools')
        self.assertIsNone(result['delta'])

    def test_replay_rejects_invalid_cached_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audit.json'
            c.save_cache(path, dict(version=1, features_sha256='a'*64,
                panels={sid(1): dict(panel(1), tempo=None)}, previews={}))
            with self.assertRaises(ValueError):
                c.load_cache(path, 'a'*64, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_fixed_pool_reports_source_specific_effects_and_stable_membership(self):
        rows = [track(0, 10, lb=AUDIO), track(1, 11), track(2, 12), track(3, 13)]
        for i in range(4, 12):
            rows.append(track(i, 20+i, rb=dict(panel(i, f'USXX1{i:07}'), energy=i/20), lb=dict(AUDIO, bpm=i*12)))
        rejects = [track(i, 100+i, rb=dict(panel(i, f'USXX1{i:07}'), energy=0), lb=dict(AUDIO, bpm=i*4))
                   for i in range(20, 25)]
        catalog = [track(i, 100+i, lb=dict(AUDIO, bpm=260-i)) for i in range(30, 45)]
        new = c.apply_features(rows, {sid(0): panel(0)}, {
                               '11': {'librosa': AUDIO, 'preview_sha256': 'b'*64}})
        result = c.paired_ranking(rows, new, rejects, catalog, cutoff=4)
        self.assertEqual(result['source_deltas']['preview'], {'p_at_10': 0, 'ndcg_at_10': 0})
        self.assertEqual(result['source_deltas']['reccobeats'], result['delta'])
        self.assertEqual(result['shared_pool'], 18)
        self.assertEqual(result['newly_evaluable'], 1)

    def test_full_pool_requires_held_out_likes_rejects_and_ten_positions(self):
        train = [track(i, i + 50, lb=dict(AUDIO, bpm=100+i)) for i in range(4, 11)]
        rows = [track(0, 110, lb=AUDIO)] + train
        negatives = [track(i, 40+i, lb=dict(AUDIO, bpm=80+i)) for i in range(40, 45)]
        catalog = [track(i, 100+i, lb=dict(AUDIO, bpm=i)) for i in range(50, 65)]
        for likes, rejects, pool in [(train, negatives, catalog), (rows, negatives, []),
                                     (rows, [], catalog), (rows, negatives, catalog[:1])]:
            with self.subTest(counts=[len(likes), len(rejects), len(pool)]):
                info = c.paired_ranking(likes, likes, rejects, pool, cutoff=4)
                self.assertEqual(info['status'], 'insufficient_evaluation_pool')
                self.assertIsNone(info['delta'])

    def test_digest_and_independent_authority_must_support_replayed_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.json'
            obj = dict(version=1, features_sha256='a'*64, spotify_tracks_sha256='b'*64,
                       verified_isrcs={sid(1): 'USAB19876543'},
                       panels={sid(1): panel(2)}, previews={})
            c.save_cache(path, obj)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                c.load_cache(path, 'a'*64, sha)
            obj['verified_isrcs'][sid(1)] = ISRC
            path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                c.load_cache(path, 'a'*64, sha)
            fresh_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(c.load_cache(path, 'a'*64, fresh_sha)['panels'][sid(1)]['isrc'], ISRC)

    def test_derived_patch_must_target_an_actual_missing_source(self):
        rows = [track(0, 14, rb=panel(0), lb=AUDIO), track(1, 15)]
        for panels, previews in [({sid(2): panel(2)}, {}), ({sid(0): panel(0)}, {}),
                                 ({}, {'17': {'librosa': AUDIO}}),
                                 ({}, {'14': {'librosa': AUDIO}})]:
            with self.subTest(panels=panels, previews=previews), self.assertRaises(ValueError):
                c.apply_features(rows, panels, previews)

    def test_conflicting_frozen_identity_prevents_cache_publication(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap, spotify, dest = [root / x for x in ('snapshot.json', 'spotify.json', 'derived.json')]
            snap.write_text(json.dumps({'tracks': [dict(track(1), isrc='USAB19876543')]}))
            spotify.write_text(json.dumps([{'id': sid(1), 'external_ids': {'isrc': ISRC}}]))
            argv = ['coverage', '--features', str(snap), '--spotify-tracks', str(spotify),
                    '--recover', str(dest), '--preview-limit', '0']
            with patch('sys.argv', argv), patch.object(c, 'fetch_panels', return_value=[panel(1)]):
                with self.assertRaises(ValueError):
                    c.main()
            self.assertFalse(dest.exists())
