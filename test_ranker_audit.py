"""Offline regressions for the v2 ranker and its final Spotify identity check."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from subprocess import CompletedProcess
import unittest
from unittest.mock import patch

import numpy as np

import recommend_v2 as R
import scoring


AF1 = '0pFz5fQBkspNYZGKBgVfdl'
DRIFT = '69zSQXthMS87H2AvfnMWvd'


def hit(title, artist, sid):
    return {'title': title, 'subtitle': artist, 'spotify_uri': 'spotify:track:' + sid}


class VerifyTests(unittest.TestCase):
    def verify(self, tracks, resolver, liked_ids=(), liked_keys=()):
        with patch.object(R.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            return R.verify_candidates(tracks, resolver, set(liked_ids), set(liked_keys), set(), set())

    def test_wrong_song_search_response_does_not_pass_known_track_verification(self):
        # A second ID for af1 is absent from liked IDs. The candidate's own key
        # is drift, so testing its metadata cannot repair a wrong search hit.
        row = {'title': 'drift (feat. Lames)', 'artist': 'Lilbubblegum'}
        response = {'sections': [{'items': [hit('af1', 'lilbubblegum', AF1)]}]}
        result = self.verify([row], lambda t, a: R.pick_search_hit(t, a, response),
                             liked_keys=[('af1', 'lilbubblegum')])
        self.assertEqual(result, [])

    def test_search_requires_meaningful_title_artist_and_matching_version(self):
        cases = [('', 'Lilbubblegum', '', 'Lilbubblegum'),
                 ('drift', '', 'drift', ''),
                 ('drift [Live]', 'Lilbubblegum', 'drift', 'Lilbubblegum'),
                 ('af1 (sped up)', 'Lilbubblegum', 'af1', 'Lilbubblegum')]
        for title, artist, other_title, other_artist in cases:
            with self.subTest(title=title, artist=artist):
                self.assertIsNone(R.pick_search_hit(title, artist, [hit(other_title, other_artist, AF1)]))
        self.assertIsNone(R.pick_search_hit('drift', 'A', [hit('drift', 'A', 'malformed/id')]))
        response = [hit('af1', 'lilbubblegum', AF1), hit('drift', 'lilbubblegum, Lames', DRIFT)]
        self.assertEqual(R.pick_search_hit('drift (feat. Lames)', 'Lilbubblegum', response), DRIFT)

    def test_different_ids_and_distinct_parenthesized_titles_stay_eligible_in_both_directions(self):
        pairs = [('af1', 'af1 (sped up)'), ('A Milli', 'A Milli (Album Version Explicit)'),
                 ('A Milli (Album Version)', 'A Milli (Album Version Explicit)')]
        for left, right in pairs:
            for known, candidate in [(left, right), (right, left)]:
                with self.subTest(known=known, candidate=candidate), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'songs'
                    path.write_text(f'{known}\tArtist, Guest\told\n')
                    with patch.multiple(R, LIKED_FILE=path, HIST_FILE=path):
                        blocked = R.load_id_sets()
                    self.assertFalse(R.is_already_known('new', candidate, 'Artist', *blocked))
                    self.assertTrue(R.is_already_known('old', candidate, 'Artist', *blocked))
                    self.assertIsNone(R.pick_search_hit(candidate, 'Artist', [hit(known, 'Artist', AF1)]))

    def test_matching_version_with_optional_feature_credit_is_excluded_in_both_directions(self):
        base = 'A Milli (Album Version Explicit)'
        for known, candidate in [(base, base + ' (feat. Guest)'), (base + ' (feat. Guest)', base)]:
            with self.subTest(known=known), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'songs'
                path.write_text(f'{known}\tLil Wayne, Guest\told\n')
                with patch.multiple(R, LIKED_FILE=path, HIST_FILE=path):
                    blocked = R.load_id_sets()
                self.assertTrue(R.is_already_known('new', candidate, 'Lil Wayne', *blocked))
                self.assertFalse(R.is_already_known('new', candidate, 'Guest, Lil Wayne', *blocked))

    def test_known_fields_survive_partial_export_rows_and_padded_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            liked, history = Path(tmp) / 'liked', Path(tmp) / 'history'
            liked.write_text('Known Title\tFirst Artist, Guest\n')
            history.write_text('Old Name\tOld Artist\t spotify:track:previous \n')
            with patch.multiple(R, LIKED_FILE=liked, HIST_FILE=history):
                known = R.load_id_sets()
        self.assertTrue(R.is_already_known('new', 'Known Title', 'First Artist', *known))
        self.assertTrue(R.is_already_known('previous', 'New Title', 'New Artist', *known))

    def test_hash_prefixed_song_name_is_not_skipped_as_a_group_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'songs'
            path.write_text('# Playlist summary\n#Song\tArtist\tspotify:track:known\n')
            with patch.multiple(R, LIKED_FILE=path, HIST_FILE=path):
                known = R.load_id_sets()
        self.assertTrue(R.is_already_known('new', '#Song', 'Artist', *known))

    def test_known_source_id_is_excluded_even_when_search_resolves_another_id(self):
        row = {'title': 'Unfamiliar alias', 'artist': 'Artist', 'spotify_id': 'liked'}
        self.assertEqual(self.verify([row], lambda *_: 'different', liked_ids=['liked']), [])

    def test_missing_export_ids_do_not_link_unrelated_songs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'songs'
            path.write_text('Known\tArtist\tNone\n')
            with patch.multiple(R, LIKED_FILE=path, HIST_FILE=path):
                liked_ids, _, _, _ = R.load_id_sets()
        row = {'title': 'Different', 'artist': 'Artist', 'spotify_id': 'None'}
        self.assertEqual([r['spotify_id'] for r in self.verify([row], lambda *_: 'real', liked_ids)], ['real'])

    def test_first_verified_record_wins_for_duplicate_ids_and_recording_keys(self):
        rows = [{'title': 'First song', 'artist': 'Solo', 'deezer_id': 1},
                {'title': 'Alternate spelling', 'artist': 'Solo', 'deezer_id': 2},
                {'title': 'FIRST SONG (feat. Guest)', 'artist': 'Solo, Guest', 'deezer_id': 3},
                {'title': 'First song', 'artist': 'Cover artist, Solo', 'deezer_id': 4}]
        ids = {(r['title'], r['artist']): sid for r, sid in
               zip(rows, ['sid1', 'sid1', 'sid2', 'sid3'])}
        result = self.verify(rows, lambda title, artist: ids[title, artist])
        self.assertEqual([(r['deezer_id'], r['spotify_id']) for r in result], [(1, 'sid1'), (4, 'sid3')])


class GenerationTests(unittest.TestCase):
    """Run recommend() against file-backed scoring artifacts; all writes go to temp."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.liked = self.root / 'liked.tsv'
        self.liked.write_text('')
        self.hist = self.root / 'history.tsv'
        self.hist.write_text('')
        self.out = self.root / 'recommendations_v2.json'
        self.pool = self.root / 'pool_features.json'
        (self.root / 'features_v2.json').write_text('{"tracks":[]}')
        (self.root / 'model_v2.json').write_text('{"gmm_k":1,"clusters":[]}')
        # A small transparent placeholder for the absent fitted NPZ. One LR
        # coefficient and a constant distance component give known v2 scores.
        n = len(scoring.feat_vector({})[0])
        coef = np.zeros((1, n))
        coef[0, 0] = 1
        np.savez(self.root / 'model_v2.npz', imputer_median=np.zeros(n),
                 scaler_mean=np.zeros(n), scaler_scale=np.ones(n), pca_components=np.zeros((1, n)),
                 pca_mean=np.zeros(n), centroids=np.zeros((1, 1)), lr_coef=coef, lr_intercept=[0.],
                 negdist_mean=0., negdist_std=1., lr_mean=0., lr_std=1.)
        self.enterContext(patch.multiple(R, BASE=self.root, DATA=self.root,
                                         LIKED_FILE=self.liked, HIST_FILE=self.hist))
        self.enterContext(patch.object(scoring, 'BASE', self.root))
        self.enterContext(patch.object(R.time, 'sleep'))
        self.enterContext(patch.object(R.F, 'fetch_reccobeats', side_effect=AssertionError('unexpected network fill')))

    def run_generation(self, rows, responses=None):
        self.pool.write_text(json.dumps(rows))
        def search(cmd, **_):
            title, artist = cmd[cmd.index('--query') + 1], cmd[cmd.index('--artist') + 1]
            items = (responses[title] if responses is not None else
                     [hit(title, artist, title.replace(' ', '').rjust(22, '0'))])
            return CompletedProcess(cmd, 0, stdout=json.dumps({'sections': [{'items': items}]}))
        with patch.object(R.subprocess, 'run', search), contextlib.redirect_stdout(io.StringIO()):
            return R.recommend()

    @staticmethod
    def row(i, score=1.):
        return {'title': f'Track {i:03d}', 'artist': 'Artist', 'deezer_id': i,
                'librosa': {'bpm': 120., 'rms_mean': .3, 'brightness_mean': 2700.},
                'reccobeats': {'danceability': score}}

    def test_mmr_breaks_ties_independently_of_pool_order(self):
        for ids in [[3, 1, 2], [2, 3, 1]]:
            rows = [self.row(i) for i in ids]
            indices = R.mmr_select(rows, np.ones((3, 2)), np.ones(3), 3)
            self.assertEqual([ids[i] for i in indices], [1, 2, 3])

    def test_ties_at_both_scoring_windows_have_reproducible_membership_and_order(self):
        rows = [self.row(i) for i in range(101, 206)]
        for order in [rows, rows[::-1]]:
            result = self.run_generation(order)
            self.assertEqual([r['deezer_id'] for r in result], list(range(101, 111)))
            self.assertEqual(json.loads(self.out.read_text()), result)
            self.assertTrue(all(r['v2_score'] == 0.5 for r in result))

    def test_thirty_blocked_top_scores_leave_ten_clean_picks(self):
        rows = [self.row(i, 1. - i / 100.) for i in range(40)]
        self.hist.write_text(''.join(f'Track {i:03d}\tARTIST, Guest\tother{i}\n' for i in range(30)))
        self.assertEqual([r['deezer_id'] for r in self.run_generation(rows)], list(range(30, 40)))

    def test_historical_release_keys_are_dropped_even_with_other_spotify_ids(self):
        rows = [dict(self.row(1), title='af1 (sped up)', artist='Lilbubblegum'),
                dict(self.row(2), title='A Milli (Album Version Explicit)', artist='Lil Wayne'),
                self.row(3)]
        self.hist.write_text('af1 (sped up)\tLilbubblegum\told1\nA Milli (Album Version Explicit)\tLil Wayne\told2\n')
        responses = {r['title']: [hit(r['title'], r['artist'], str(i + 1).zfill(22))]
                     for i, r in enumerate(rows)}
        self.assertEqual([r['deezer_id'] for r in self.run_generation(rows, responses)], [3])

    def test_historical_drift_search_exact_or_absent(self):
        row = dict(self.row(1), title='drift (feat. Lames)', artist='Lilbubblegum')
        wrong = hit('af1', 'Lilbubblegum', AF1)
        right = hit('drift', 'Lilbubblegum', DRIFT)
        for items, expected in [([wrong, right], [DRIFT]), ([wrong], [])]:
            with self.subTest(hits=len(items)):
                out = self.run_generation([row], {row['title']: items})
                self.assertEqual([r['spotify_id'] for r in out], expected)

    def test_offline_archive_replay_checks_the_full_output_and_missing_search_evidence(self):
        from audit.replay_v10 import replay
        data = self.root / 'data'
        data.mkdir()
        rows = [self.row(i, 1. - i / 100.) for i in range(10)]
        (data / 'pool_features.json').write_text(json.dumps(rows))
        (self.root / 'spotify-liked-songs.txt').write_text('')
        (self.root / 'spotify-recommended-tracks.txt').write_text('')
        responses, expected = [], []
        for i, r in enumerate(rows):
            sid = str(i + 1).zfill(22)
            responses.append({'title': r['title'], 'artist': 'Artist',
                              'response': [hit(r['title'], 'Artist', sid)]})
            expected.append(dict(title=r['title'], artist='Artist', spotify_id=sid, deezer_id=i,
                                 v2_score=round((1. - i / 100.) / 2, 3), v6_score_per_bucket=.39,
                                 v6_bucket='global', v6_score_global=.39, nearest_cluster=0, audio_source=None))
        responses_file = self.root / 'spotify_search_responses.json'
        expected_file = self.root / 'recommendations_v10.json'
        responses_file.write_text(json.dumps(responses))
        expected_file.write_text(json.dumps(expected))
        self.assertTrue(replay(self.root)['output_identical'])
        expected_file.write_text(json.dumps(expected[::-1]))
        delta = replay(self.root)
        self.assertEqual((delta['common_ids'], delta['same_positions'], delta['output_identical']), (10, 0, False))
        responses_file.write_text(json.dumps(responses[:-1]))
        with self.assertRaisesRegex(ValueError, 'search is missing'):
            replay(self.root)

    def test_exhausted_run_replaces_stale_disk_output_with_empty_list(self):
        self.out.write_text('[{"spotify_id":"previously_offered"}]')
        self.hist.write_text('Track 010\tArtist\tolderid\n')
        self.assertEqual(self.run_generation([self.row(10)]), [])
        self.assertEqual(json.loads(self.out.read_text()), [])

    def test_failed_spotify_response_leaves_last_successful_output_intact(self):
        self.pool.write_text(json.dumps([self.row(10)]))
        previous = '[{"spotify_id":"last_successful_run"}]'
        for status, payload in [(9, ''), (0, '<bad json>'), (0, '{"error":"unavailable"}')]:
            with self.subTest(status=status, payload=payload):
                self.out.write_text(previous)
                with (patch.object(R.subprocess, 'run', return_value=CompletedProcess([], status, stdout=payload)),
                      contextlib.redirect_stdout(io.StringIO()), self.assertRaises((RuntimeError, ValueError))):
                    R.recommend()
                self.assertEqual(self.out.read_text(), previous)

    def test_pool_with_no_audio_produces_empty_output(self):
        self.out.write_text('[{"spotify_id":"previously_offered"}]')
        self.assertEqual(self.run_generation([dict(self.row(10), librosa=None)]), [])
        self.assertEqual(json.loads(self.out.read_text()), [])


if __name__ == '__main__':
    unittest.main()
