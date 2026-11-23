"""Small, independent counterexamples for evaluation correctness."""
import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import embeat_port
import evaluate
import evaluate_v3 as v3
import integrate_v3
import integrate_v4
import model_v2
import scoring
import troi_patch
from test_evaluate_v4 import track


PASSING_OFFLINE = dict(delta_ndcg_ci_low=.04, delta_ndcg=.1, delta_p10=.1,
                       delta_map10=.03, p10_v3=.6, p10_v3_ci_low=.4,
                       random_p10=.3, n_neg_top10=0, n_neg_top10_H=0,
                       mean_rank_negative_v3=200, random_mean_rank=150)


class GateTests(unittest.TestCase):
    def test_one_perfect_list_is_still_pending(self):
        verdict = v3.verdict_4leg(PASSING_OFFLINE | dict(live_keeps=10, live_n=10,
                                                         live_per_list_keeps=[10]))
        self.assertTrue(verdict['verdict'].startswith('PENDING'), verdict['verdict'])

    def test_earlier_stop_cannot_be_erased_by_later_lists(self):
        verdict = v3.verdict_4leg(PASSING_OFFLINE | dict(live_keeps=36, live_n=42,
                                                         live_per_list_keeps=[7, 7, 10, 10]))
        self.assertTrue(verdict['verdict'].startswith('NO-GO'), verdict['verdict'])

    def test_historical_fixed_panel_cannot_qualify_for_promotion(self):
        result = v3.verdict_4leg(PASSING_OFFLINE | dict(inferential=False, live_n=42,
                                                        live_keeps=42))
        self.assertFalse(result['offline_pass'])
        self.assertFalse(result['verdict'].startswith('GO'))

    def test_no_eligible_reject_cannot_pass_the_control(self):
        result = v3.verdict_4leg(PASSING_OFFLINE | dict(mean_rank_negative_v3=None,
                                                        n_ranked=10, n_rejects=0))
        self.assertFalse(result['offline_pass'])
        self.assertEqual(result['legs']['leg3_negative_control']['power_label'], 'UNAVAILABLE')

    def test_invalid_rating_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            v3.verdict_4leg(PASSING_OFFLINE | dict(live_keeps=17, live_n=10))


class FormulaTests(unittest.TestCase):
    def test_ndcg_denominator_includes_missed_relevant_rows(self):
        rel = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ideal = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(evaluate.ndcg_at_k(rel), 1 / ideal)
        self.assertEqual(v3.metrics_from_rel(rel[:10], 2)['NDCG@10'], round(1 / ideal, 3))

    def test_precision_missing_slots_count_as_misses(self):
        self.assertEqual(evaluate.precision_at_k([1, 1]), .2)
        self.assertEqual(evaluate.precision_at_k([]), 0)

    def test_random_baselines_use_k_and_finite_pool(self):
        # Enumerate the two rankings of one relevant and one unlabeled row.
        for k, p, n in [(1, .5, .5), (10, .1, (1 + 1 / math.log2(3)) / 2)]:
            baseline = v3.random_baseline(1, 2, k=k)
            self.assertAlmostEqual(baseline[f'random_P@{k}'], p)
            self.assertAlmostEqual(baseline[f'random_NDCG@{k}'], n)

    def test_paired_points_and_replicates_use_same_canonical_ties(self):
        items = [dict(spotify_id='z', kind='heldout', relevant=1, a=0, b=1),
                 dict(spotify_id='a', kind='distractor', relevant=0, a=0, b=0)]
        # With one row per stratum, every draw is the same panel.
        for order in (items, items[::-1]):
            result = v3.paired_bootstrap(order, 'a', 'b', higher_b=True, B=20, k=1)
            self.assertEqual(result['delta_p1'], -1)
            self.assertEqual(result['delta_p1_ci'], [-1, -1])
            self.assertEqual(result['delta_ndcg'], -1)
            self.assertEqual(result['p1_a'], 0)

    def test_a_bootstrap_point_is_the_observed_precision_on_short_lists(self):
        rows = [dict(spotify_id='a', kind='heldout', relevant=1, a=1, b=0),
                dict(spotify_id='b', kind='distractor', relevant=0, a=0, b=1)]
        boot = v3.paired_bootstrap(rows, 'a', 'b', higher_b=True, B=10, k=10)
        self.assertEqual(boot['p10_a'], .1)

    def test_nondefault_cutoffs_are_labeled_with_the_evaluated_k(self):
        result = v3.metrics_from_rel([0, 1, 0], 1, k=2)
        self.assertIn('NDCG@2', result)
        self.assertEqual(result['P@2'], .5)
        self.assertNotIn('P@10', result)

    def test_troi_short_panel_and_descriptive_interval(self):
        rows = [dict(kind='heldout', relevant=1, troi=2, v6b=1, spotify_id='a'),
                dict(kind='distractor', relevant=0, troi=1, v6b=2, spotify_id='b')]
        boot = troi_patch._paired_bootstrap(rows, B=20)
        self.assertEqual(boot['P@10_troi']['ci_95'], [.1, .1])
        self.assertFalse(boot['inferential'])

    def test_overlap_outside_top_ten_is_still_training_leakage(self):
        for heldout, distractors in [({'overlap'}, set()), (set(), {'overlap'})]:
            with self.assertRaisesRegex(AssertionError, 'LEAKAGE'):
                v3.check_leakage([], {'overlap'}, heldout, set(), distractors)

    def test_negative_controls_use_without_replacement_and_need_observations(self):
        # Of the six unordered pairs of positions, just one misses both top slots.
        self.assertAlmostEqual(v3.negative_control_stats(2, 4, 2)
                               ['p_zero_neg_in_top10_random'], 1/6, places=3)
        self.assertNotEqual(v3.negative_control_stats(0, 4, 2)['power_label'], 'POWERED')


def full_track(key, value, **extra):
    row = track(key, value, **extra)
    for name in scoring.LIB_KEYS:
        row['librosa'].setdefault(name, .1 * value)
    for name, n in scoring.LIB_VEC:
        row['librosa'].setdefault(name, [value] * n)
    row['librosa']['key'] = 'A minor'
    row['reccobeats'] = {name: .1 * value for name in scoring.RB_KEYS} | {'key': value % 12}
    return row


class PartitionTests(unittest.TestCase):
    def test_v2_evaluation_refits_without_heldout_aliases_and_test_rejects(self):
        with tempfile.TemporaryDirectory() as work, threadpool_limits(limits=1), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(work)
            data = root / 'data'
            data.mkdir()
            train = [full_track(f'old{i}', i + 3, position=130+i) for i in range(10)]
            test = [full_track(f'new{i}', i, position=i) for i in range(4)]
            # These two training rows connect to the first test like through
            # a catalog bridge, despite distinct provider IDs.
            train[0]['isrc'] = 'US-ABC-24-12345'
            train[1]['preview_sha256'] = 'b' * 64
            test[0]['deezer_id'] = 77
            bridge = full_track('bridge', 12, deezer_id=77, isrc='USABC2412345',
                                preview_sha256='b' * 64)
            rejects = [full_track(f'rej{i}', 20+i) for i in range(6)]
            catalog = [bridge] + [full_track(f'cat{i}', 30+i) for i in range(10)]
            (data / 'negatives_features.json').write_text(json.dumps(rejects))
            (data / 'pool_features.json').write_text(json.dumps(catalog))
            (root / 'features_v2.json').write_text(json.dumps(dict(tracks=train + test)))
            frozen = root / 'evaluation_v2.json'
            frozen.write_text('{"frozen":true}')
            # A valid old model is available so a regression which just loads it
            # can actually run and produce the wrong pool.
            config, values = model_v2.fit_records(train, rejects)
            (root / 'model_v2.json').write_text(json.dumps(config))
            np.savez(root / 'model_v2.npz', **values)
            with patch.object(evaluate, 'BASE', work), patch.object(evaluate, 'DATA', str(data)), \
                    patch.object(scoring, 'BASE', work):
                evaluate.main()
            ranked = json.loads((data / 'evaluation_audit.json').read_text())
            self.assertEqual(sum(r['kind'] == 'negative' for r in ranked), 2)
            self.assertNotIn('bridge', [r['spotify_id'] for r in ranked])
            self.assertEqual(json.loads(frozen.read_text()), {'frozen': True})
            summary = json.loads((data / 'evaluation_v2_refit.json').read_text())
            self.assertEqual(summary['training_counts'], {'like': 8, 'reject': 4})
            for row in train + test:
                row.pop('genre_bucket', None)
            (root / 'features_v2.json').write_text(json.dumps(dict(tracks=train + test)))
            with patch.object(evaluate, 'BASE', work), patch.object(evaluate, 'DATA', str(data)):
                evaluate.main()
            summary = json.loads((data / 'evaluation_v2_refit.json').read_text())
            self.assertIsNone(summary['genre_sanity']['accuracy'])
            self.assertEqual(summary['v2']['n_ranked'], len(ranked))


    def test_v4_refit_is_invariant_to_features_of_transitive_training_twins(self):
        train = [track(f't{i}', i + 1, position=10+i) for i in range(9)]
        heldout = track('h', 5, position=0, deezer_id=10)
        train[0]['isrc'] = 'GB-ABC-24-12345'
        train[1]['preview_sha256'] = 'b' * 64
        bridge = track('x', deezer_id=10, isrc='GBABC2412345', preview_sha256='b'*64)
        rejects = [track(f'r{i}', 30+i) for i in range(6)]
        with contextlib.redirect_stdout(io.StringIO()):
            def run(records):
                return integrate_v4.run_experiment(records+[heldout], rejects, [bridge], cutoff=10)
            original = run(train)
            altered = [dict(row, librosa=track('other', 99)['librosa']) if i < 2 else row
                       for i, row in enumerate(train)]
            edited = run(altered)
        self.assertEqual(original['training_counts'], {'like': 7, 'reject': 4})
        self.assertEqual(original['pools'], edited['pools'])
        self.assertEqual(original['seed_recording_id'], edited['seed_recording_id'])


class HistoricalReplayTests(unittest.TestCase):
    def test_helpers_use_the_checkout_that_owns_the_integration(self):
        # Exercise a tiny copied checkout with its own sentinel snapshot, no
        # model and no private pool. A different home must not supply either.
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in Path(__file__).resolve().parent.glob('*.py'):
                shutil.copy(path, root / path.name)
            (root / 'features_v2.json').write_text(json.dumps({'tracks': []}))
            probe = """
import json
from unittest.mock import patch
with patch('os.path.expanduser', side_effect=lambda path: path.replace('~', '/different-home')):
    import integrate_v3, embeat_port
    results = []
    for fn in (integrate_v3.main, embeat_port.load_pool, embeat_port.get_model):
        try: fn()
        except FileNotFoundError as exc: results.append(exc.filename)
    print(json.dumps(results))
"""
            out = subprocess.check_output([sys.executable, '-B', '-c', probe], cwd=root, text=True)
            self.assertEqual(json.loads(out.splitlines()[-1]), [
                str(root / 'model_v2.npz'), str(root / 'data/pool_features.json'),
                str(root / 'model_v2.npz')])

    def test_integration_preserves_both_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('evaluation_v3', 'v3_scores_306'):
                (root / (name + '.json')).write_bytes(b'frozen')
            with patch.object(integrate_v3, 'BASE', directory):
                integrate_v3.write_outputs({'version': 3}, {'items': []})
            for name, data in [('evaluation_v3', {'version': 3}), ('v3_scores_306', {'items': []})]:
                self.assertEqual((root / (name + '.json')).read_bytes(), b'frozen')
                self.assertEqual(json.loads((root / 'data' / (name + '_replay.json')).read_text()), data)

    def test_default_writers_preserve_frozen_embeat_and_troi(self):
        # Stand in for the unavailable private score/data loaders. Run both
        # public evaluators, their metric calculations, and actual JSON writes.
        rows = [dict(rec={'spotify_id': str(i)}, relevant=int(i < 99),
                     kind='heldout' if i < 99 else 'negative' if i < 111 else 'distractor',
                     v2=311-i, v6b=i, v6g=i, port=311-i) for i in range(311)]
        counts = dict(n_ranked=311, test_with_librosa=99, negatives_with_librosa=12)
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as cleanup:
            root = Path(directory)
            (root / 'data').mkdir()
            outputs = [('embeat', embeat_port.evaluate_port), ('troi', troi_patch.evaluate_patch)]
            for name, _ in outputs:
                (root / f'evaluation_v3_{name}.json').write_bytes(b'frozen')
            for module in (embeat_port, troi_patch):
                cleanup.enter_context(patch.object(module, 'BASE', directory))
            seed = dict(spotify_id='seed', artist='test')
            replacements = dict(build_eval_items=lambda: (rows, {}),
                                eval_seed=lambda: seed, EVAL_SEEDS=['seed'],
                                rank_eval_pool=lambda *args, **kwargs: rows,
                                load_liked=lambda: [seed], recommend=lambda *args, **kwargs: [],
                                get_index=lambda: {'bucket_counts': {}})
            for name, value in replacements.items():
                cleanup.enter_context(patch.object(embeat_port, name, value))
            cleanup.enter_context(patch('subprocess.run',
                                  return_value=subprocess.CompletedProcess([], 0, '', '')))
            for name, value in dict(_build_eval_items=lambda: (rows, counts),
                                    _patch_rank_order=lambda items: items.copy(),
                                    find_duplicate_audio_items=lambda items: [],
                                    _seed_sweep=lambda: {}).items():
                cleanup.enter_context(patch.object(troi_patch, name, value))
            for name, writer in outputs:
                with self.subTest(arm=name), contextlib.redirect_stdout(io.StringIO()):
                    result = writer()
                    self.assertEqual((root / f'evaluation_v3_{name}.json').read_bytes(), b'frozen')
                    self.assertEqual(json.loads((root / f'data/evaluation_v3_{name}_replay.json')
                                                .read_text()), result)


if __name__ == '__main__':
    unittest.main()
