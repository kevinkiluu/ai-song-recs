import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import integrate_v4 as v4
from model_v2 import fit_records
from scoring import V2Model
from test_evaluate_v4 import track


def fixture():
    likes = [track(f'l{i}', i, position=i, deezer_id=1000 + i) for i in range(32)]
    rejects = [track(f'r{i}', i + 80, label_source='original rating') for i in range(8)]
    catalog = [track(f'c{i}', i + 40) for i in range(16)]
    catalog += [track('alias', 5, deezer_id=1000)]
    return likes, rejects, catalog


class ExperimentTests(unittest.TestCase):
    def test_refit_with_missing_columns_keeps_scoring_dimensions(self):
        likes, rejects, catalog = fixture()
        config, values = fit_records(likes, rejects)
        model = V2Model(values, config)
        scores = model.score_records(catalog)['final']
        self.assertEqual(len(model.median), 78)
        self.assertTrue(np.isfinite(scores).all())

    def test_full_and_sampled_experiment_share_eligibility_and_audit(self):
        likes, rejects, catalog = fixture()
        result = v4.run_experiment(likes, rejects, catalog, cutoff=10, sample_size=4)
        again = v4.run_experiment(likes, rejects, catalog, cutoff=10, sample_size=4)
        self.assertEqual(result, again)
        self.assertEqual(result['offline_status'], 'INSUFFICIENT_EVIDENCE')
        self.assertEqual(result['status'], 'INSUFFICIENT_EVIDENCE')
        full = result['pools']['full']
        sample = result['pools']['sample_42']
        self.assertEqual(set(full['metrics']), {'v2_refit', 'v6_frozen', 'embeat_full',
                                               'embeat_minus_ch1'})
        full_ids = {r['recording_id'] for r in full['rows']}
        self.assertEqual(len(full_ids), len(full['rows']))
        self.assertEqual(len(sample['rows']), 17)
        full_scores = {r['recording_id']: r['scores'] for r in full['rows']}
        self.assertTrue(all(r['scores'] == full_scores[r['recording_id']]
                            for r in sample['rows']))
        self.assertTrue({r['recording_id'] for r in sample['rows']} <= full_ids)
        self.assertTrue(all(len(row['scores']) == len(row['ranks']) == 4
                            for row in full['rows']))
        training = {r['recording_id'] for r in result['manifest']
                    if r['partition'] == 'train'}
        self.assertFalse(training & full_ids)

    def test_heldout_taste_bucket_is_not_used_as_candidate_metadata(self):
        likes, rejects, catalog = fixture()
        original = v4.run_experiment(likes, rejects, catalog, cutoff=10, sample_size=4)
        edited = [dict(row, genre_bucket='H' if row['genre_bucket'] != 'H' else 'E')
                  if row['position'] < 10 else row for row in likes]
        rerun = v4.run_experiment(edited, rejects, catalog, cutoff=10, sample_size=4)
        self.assertEqual(original['pools'], rerun['pools'])

    def test_blind_output_has_no_method_score_or_old_tracks(self):
        likes, rejects, catalog = fixture()
        result = v4.run_experiment(likes, rejects, catalog, cutoff=10)
        candidates = [r for r in result['manifest'] if r['label'] == 'unlabeled']
        verified = {r['recording_id']: 'track' + str(i).zfill(17)
                    for i, r in enumerate(candidates)}
        excluded_id = next(iter(verified.values()))
        result['offline_status'] = 'ELIGIBLE_FOR_BLIND'
        blind, reveal = v4.prepare_blind(result, verified, {excluded_id}, set())
        self.assertEqual(len(blind), 10)
        self.assertEqual(set(blind[0]), {'token', 'title', 'artist', 'spotify_id'})
        self.assertFalse(any(r['spotify_id'] == excluded_id for r in blind))
        self.assertEqual(sum(r['method'] == v4.TRIAL for r in reveal), 5)
        self.assertEqual({r['token'] for r in blind}, {r['token'] for r in reveal})
        self.assertTrue(all(len(r['scores']) == 4 for r in reveal))
        one_round = dict(trial_id=result['trial_id'], reveal=reveal,
                         ratings={row['token']: True for row in blind})
        summary = v4.summarize_live([one_round], result['trial_id'], result['pools'])
        self.assertEqual((summary['keeps'], summary['n']), (5, 5))
        with self.assertRaisesRegex(ValueError, 'repeat'):
            v4.summarize_live([one_round, one_round], result['trial_id'], result['pools'])
        with self.assertRaisesRegex(ValueError, 'trial'):
            v4.summarize_live([one_round], 'another-model', result['pools'])

    def test_gate_requires_ablated_arm_to_beat_intact_embeat(self):
        pool = {
            'counts': {'like': 20, 'reject': 4},
            'chance_p_at_k': .2, 'chance_mean_rank': 50,
            'metrics': {v4.TRIAL: {'p_at_k': .6, 'negative_top_k': 0,
                                   'mean_negative_rank': 90}},
            'deltas': {
                'v2_refit': {'ndcg_at_k': .2, 'p_at_k': .2, 'map_at_k': .1},
                'v6_frozen': {'ndcg_at_k': .1, 'p_at_k': .1, 'map_at_k': .1},
                'embeat_full': {'ndcg_at_k': -.1, 'p_at_k': 0, 'map_at_k': 0}
            }
        }
        self.assertEqual(v4.offline_decision({'full': pool}, fresh=True), 'NO_GO_OFFLINE')
        pool['deltas']['embeat_full']['ndcg_at_k'] = .05
        self.assertEqual(v4.offline_decision({'full': pool}, fresh=True), 'ELIGIBLE_FOR_BLIND')

    def test_absent_fresh_matches_are_reported_as_insufficient(self):
        likes, rejects, catalog = fixture()
        fresh = [track('no-frozen-match', label='like', label_source='new observation')]
        result = v4.run_experiment(likes, rejects, catalog, fresh=fresh, sample_size=0)
        self.assertEqual(result['status'], 'INSUFFICIENT_EVIDENCE')
        self.assertIn('No eligible known likes in the evaluation pool', result['blockers'])
        self.assertFalse(result['pools'])

    def test_missing_private_inputs_are_reported_without_scoring(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            available = root / 'likes.json'
            available.write_text(json.dumps({'tracks': []}))
            out = root / 'audit.json'
            code = v4.main(['--likes', str(available), '--rejects', str(root / 'rejects.json'),
                            '--catalog', str(root / 'catalog.json'), '--output', str(out)])
            report = json.loads(out.read_text())
        self.assertEqual(code, 2)
        self.assertEqual(report['status'], 'INSUFFICIENT_EVIDENCE')
        self.assertEqual(set(report['missing_inputs']), {'rejects', 'catalog'})
        self.assertEqual(report['pools'], {})


if __name__ == '__main__':
    unittest.main()
