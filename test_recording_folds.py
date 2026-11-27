import copy
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate_v4 import build_manifest
import unittest

import integrate_v4
from test_integrate_v4 import fixture
from test_evaluate_v4 import track


class FrozenIdentityTests(unittest.TestCase):
    def test_only_a_verified_spotify_href_can_supply_a_nested_isrc(self):
        sid = 'A' * 22
        source = {'spotify_id': sid, 'reccobeats': {
            'href': f'https://open.spotify.com/track/{sid}', 'isrc': 'gb-abc-23-00001'}}
        enriched = integrate_v4._enrich_records([source], {}, None)[0]
        self.assertEqual(enriched.get('isrc'), 'GBABC2300001')
        self.assertNotIn('isrc', source)
        for href in ('https://open.spotify.com/track/' + 'B' * 22, 'https://example.com/track/' + sid):
            row = dict(source, reccobeats=dict(source['reccobeats'], href=href))
            self.assertIsNone(integrate_v4._enrich_records([row], {}, None)[0].get('isrc'))
        invalid = dict(source, reccobeats=dict(source['reccobeats'], isrc='unknown'))
        self.assertIsNone(integrate_v4._enrich_records([invalid], {}, None)[0].get('isrc'))
        with self.assertRaisesRegex(ValueError, 'ISRC'):
            integrate_v4._enrich_records([dict(source, isrc='GBABC2300002')], {}, None)

    def test_programmatic_split_uses_verified_embedded_identity(self):
        likes, rejects, catalog = fixture()
        likes[0]['spotify_id'] = 'A' * 22
        likes[0]['reccobeats'] = {**(likes[0].get('reccobeats') or {}),
                                  'href': 'https://open.spotify.com/track/' + 'A'*22,
                                  'isrc': 'GBABC2300001'}
        likes[21]['isrc'] = 'GBABC2300001'
        manifest = build_manifest(likes, rejects, catalog, cutoff=10)
        first, alias = manifest[0], manifest[21]
        self.assertEqual(first['recording_id'], alias['recording_id'])
        self.assertEqual((first['partition'], alias['partition']), ('test', 'excluded'))
        self.assertNotIn('isrc', likes[0])

    def test_strict_verifier_rejects_omitted_embedded_identity(self):
        import check_v4
        from test_audit_mutations import synthetic_inputs, execute, write_raw
        inputs = list(synthetic_inputs())
        sid = 'S' * 22
        inputs[0][1] = dict(inputs[0][1], spotify_id=sid, reccobeats={
            **(inputs[0][1].get('reccobeats') or {}), 'isrc':'GBABC2300001',
            'href':f'https://open.spotify.com/track/{sid}'})
        # Simulate a producer that used every score field but lost the nested
        # provider identity before constructing its split.
        damaged = copy.deepcopy(inputs)
        del damaged[0][1]['reccobeats']['isrc']
        result = execute(damaged)
        with TemporaryDirectory() as d:
            result['source_files'] = write_raw(Path(d), inputs)
            with self.assertRaisesRegex(ValueError, 'ISRC'):
                check_v4.verify(result)


class FoldTests(unittest.TestCase):
    def test_measures_transitive_leakage_and_scores_identical_pool(self):
        from audit.recording_folds import measure_fold
        likes, rejects, catalog = fixture()
        likes[0]['isrc'] = 'GBABC2300001'
        likes[21]['preview_sha256'] = 'a'*64
        catalog.append(track('bridge', isrc=likes[0]['isrc'], preview_sha256='a'*64))
        originals = copy.deepcopy(likes)
        result = measure_fold(likes, rejects, catalog, start=0, end=10, reject_fold=0)
        self.assertEqual(result['position_split']['overlapping_liked_components'], 1)
        self.assertEqual(result['position_split']['older_alias_rows'], 1)
        self.assertEqual(result['recording_split']['cross_partition_components'], 0)
        self.assertEqual(result['recording_split']['duplicate_eval_components'], 0)
        comparison = result['paired_fit']
        self.assertEqual(comparison['reintroduced_training_rows'], 1)
        expected = integrate_v4.run_experiment(likes, rejects, catalog, cutoff=10)
        for arm in ('v2_refit', 'embeat_full', 'embeat_minus_ch1'):
            self.assertEqual(comparison['after'][arm]['p_at_10'],
                             expected['pools']['full']['metrics'][arm]['p_at_k'])
            self.assertEqual(comparison['after'][arm]['ndcg_at_10'],
                             expected['pools']['full']['metrics'][arm]['ndcg_at_k'])
            self.assertIn(arm, comparison['before'])
        self.assertEqual(likes, originals)

    def test_partial_sources_preserve_counts_without_generating_a_score(self):
        from audit.recording_folds import measure_fold
        likes, _, _ = fixture()
        result = measure_fold(likes, None, None, start=0, end=10, reject_fold=0)
        self.assertIsNone(result['paired_fit'])
        self.assertEqual(set(result['missing_inputs']), {'rejects','catalog'})
        self.assertEqual(result['recording_split']['eval_rows'], 10)
        self.assertEqual(result['recording_split']['identity_coverage']['evaluation']['deezer'], 10)
        self.assertEqual(result['recording_split']['identity_coverage']['training']['spotify'], 22)
