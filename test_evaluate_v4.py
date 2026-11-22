import copy
import math
import random
import unittest

from sklearn.metrics import ndcg_score

import evaluate_v4 as v4


def track(key, value=0, **extra):
    return {
        'spotify_id': key, 'artist': f'artist-{value % 4}', 'title': key,
        'genre_bucket': ('E', 'R', 'H')[value % 3],
        'librosa': {'bpm': 90 + value, 'brightness_mean': 2100 + (value * 13) % 700,
                    'rms_mean': .2 + (value % 17) / 100,
                    'mfcc_mean': [value + (j * value) % 17 for j in range(13)]},
        **extra,
    }


class ManifestTests(unittest.TestCase):
    def test_transitive_aliases_exclude_training_twin_and_catalog_copy(self):
        likes = [track('old', position=3, deezer_id=42),
                 track('recent', position=0, isrc='aa-123')]
        pool = [track('bridge', deezer_id='42', isrc='AA123')]
        manifest = v4.build_manifest(likes, [], pool, cutoff=2)
        by_id = {row['record']['spotify_id']: row for row in manifest}
        self.assertEqual({row['recording_id'] for row in manifest},
                         {by_id['recent']['recording_id']})
        self.assertEqual(by_id['recent']['partition'], 'test')
        self.assertEqual(by_id['old']['reason'], 'held-out recording alias')
        self.assertEqual(by_id['bridge']['partition'], 'excluded')

    def test_reject_labels_partition_by_recording_and_rotate(self):
        negatives = [track('r1', deezer_id=51), track('r2'),
                     track('r3'), track('r1-alias', deezer_id=51)]
        folds = [v4.build_manifest([], negatives, [], reject_fold=i, reject_folds=2)
                 for i in range(2)]
        judged = []
        for fold in folds:
            train = {r['recording_id'] for r in fold if r['partition'] == 'train'}
            test = {r['recording_id'] for r in fold if r['partition'] == 'test'}
            self.assertFalse(train & test)
            judged.append(test)
        self.assertEqual(len(judged[0] | judged[1]), 3)
        self.assertFalse(judged[0] & judged[1])

    def test_missing_ids_do_not_merge_and_conflicted_labels_are_quarantined(self):
        likes = [track('like', position=2, preview_sha256='a' * 64)]
        rejects = [track('reject', preview_sha256='a' * 64)]
        pool = [track(None, deezer_id=10), track('None', deezer_id=11), track(None)]
        manifest = v4.build_manifest(likes, rejects, pool)
        self.assertEqual([r['reason'] for r in manifest[:2]], ['conflicting labels'] * 2)
        self.assertEqual(len({r['recording_id'] for r in manifest[2:]}), 3)
        self.assertEqual(manifest[-1]['reason'], 'unverified recording identity')

    def test_fresh_holdout_cannot_reuse_previously_labeled_recording(self):
        likes = [track('historic', position=0, deezer_id=90)]
        fresh = [track('new-id', deezer_id=90, label='like', label_source='export'),
                 track('new', label='like', label_source='export')]
        frozen = track('new', value=8)
        fresh += [track('absent', label='like', label_source='export')]
        manifest = v4.build_manifest(likes, [], [frozen], fresh=fresh)
        self.assertEqual(manifest[2]['reason'], 'previously labeled recording')
        self.assertEqual(manifest[3]['reason'], 'judgment joined to frozen catalog')
        self.assertEqual(manifest[4]['reason'], 'recording absent from eligible frozen catalog')
        self.assertIs(manifest[1]['record'], frozen)
        self.assertEqual((manifest[1]['partition'], manifest[1]['label']), ('test', 'like'))

    def test_conflicting_new_rating_quarantines_historical_alias(self):
        likes = [track('old', position=7, deezer_id=42)]
        fresh = [track('new', deezer_id=42, label='reject', label_source='blind')]
        manifest = v4.build_manifest(likes, [], [], fresh=fresh)
        self.assertEqual([row['reason'] for row in manifest], ['conflicting labels'] * 2)

    def test_historical_sensitivity_does_not_train_on_newer_rows(self):
        likes = [track(str(i), position=i) for i in range(5)]
        manifest = v4.build_manifest(likes, [], [], holdout_start=2, cutoff=4)
        self.assertEqual([r['partition'] for r in manifest],
                         ['excluded', 'excluded', 'test', 'test', 'train'])

    def test_provider_sentinels_are_unverified(self):
        manifest = v4.build_manifest([], [], [track('spotify:track:None', deezer_id=-1)])
        self.assertEqual(manifest[0]['reason'], 'unverified recording identity')


class MetricsTests(unittest.TestCase):
    def test_standard_ndcg_uses_all_eligible_likes(self):
        rows = [{'label': 'like'}, {'label': 'unlabeled'}, {'label': 'reject'},
                {'label': 'like'}]
        metrics = v4.ranking_metrics(rows, k=3)
        expected = ndcg_score([[1, 0, 0, 1]], [[4, 3, 2, 1]], k=3)
        self.assertAlmostEqual(metrics['ndcg_at_k'], expected)
        self.assertAlmostEqual(metrics['ndcg_at_k'], 1 / (1 + 1 / math.log2(3)))
        self.assertEqual(metrics['negative_ranks'], [3])

    def test_position_pool_uses_fixed_rows_and_without_replacement_samples(self):
        likes = [track('a', position=0, deezer_id=9), track('old', position=3, deezer_id=9)]
        catalog = [track(f'c{i}') for i in range(12)] + [track('c-copy', deezer_id=9)]
        rows = v4.build_manifest(likes, [], catalog, cutoff=2)
        pool = v4.evaluation_pool(rows)
        draw = v4.sample_pool(pool, 4, 42)
        self.assertEqual(len(draw), 5)
        self.assertEqual(len({row['recording_id'] for row in draw}), 5)
        self.assertEqual([row['recording_id'] for row in draw],
                         [row['recording_id'] for row in v4.sample_pool(pool[::-1], 4, 42)])
        self.assertNotIn('old', [row['record']['spotify_id'] for row in pool])
        self.assertEqual(len(pool), 13)

    def test_live_gate_requires_prespecified_look_and_offline_eligibility(self):
        self.assertEqual(v4.live_decision('ELIGIBLE_FOR_BLIND', 10, 10), 'PENDING_LIVE')
        self.assertEqual(v4.live_decision('ELIGIBLE_FOR_BLIND', 41, 41), 'PENDING_LIVE')
        self.assertEqual(v4.live_decision('ELIGIBLE_FOR_BLIND', 32, 42), 'GO')
        self.assertEqual(v4.live_decision('ELIGIBLE_FOR_BLIND', 31, 42), 'NO_GO_LIVE')
        self.assertEqual(v4.live_decision('ELIGIBLE_FOR_BLIND', 37, 50), 'GO')
        self.assertEqual(v4.live_decision('ELIGIBLE_FOR_BLIND', 32, 42, [7, 7]), 'NO_GO_LIVE')
        self.assertEqual(v4.live_decision('ELIGIBLE_FOR_BLIND', 32, 42, [7, 7, 8, 8]),
                         'NO_GO_LIVE')
        self.assertEqual(v4.live_decision('INSUFFICIENT_EVIDENCE', 42, 42),
                         'INSUFFICIENT_EVIDENCE')
        with self.assertRaises(ValueError):
            v4.live_decision('ELIGIBLE_FOR_BLIND', 43, 42)


if __name__ == '__main__':
    unittest.main()
