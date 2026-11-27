"""Fault injections on disposable v4 inputs. No provider calls or frozen writes."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import check_v4 as checker
import integrate_v4 as runner
from test_integrate_v4 import fixture


def synthetic_inputs():
    likes, rejects, catalog = fixture()
    # A deliberately eligible positive control; these IDs have no listener meaning.
    kept = (3, 11, 7, 8, 4, 0, 12, 9, 10, 5)
    ratings = [dict(spotify_id=f'c{i}', label='like' if i in kept else 'reject',
                    label_source='synthetic judgment') for i in (*kept, 13, 15)]
    return likes, rejects, catalog, ratings


def execute(inputs, *, live=None):
    likes, rejects, catalog, ratings = inputs
    with contextlib.redirect_stdout(io.StringIO()):
        return runner.run_experiment(likes, rejects, catalog, fresh=ratings, live=live)


def plant_known_reject(inputs):
    # Plant only a challenger score, upstream of the producer's ranking,
    # metrics, sample draws, trial digest, and both eligibility decisions.
    rank = runner._rank_pool

    def plant(rows):
        for row in rows:
            if row['recording_id'] == 'spotify:c13':
                row['scores'][runner.TRIAL] = 20.0
        return rank(rows)

    with patch.object(runner, '_rank_pool', side_effect=plant):
        return execute(inputs)


def write_raw(root, inputs):
    files = {}
    likes, rejects, catalog, ratings = inputs
    for name, data in [('likes', {'tracks': likes}), ('rejects', rejects),
                       ('catalog', catalog), ('fresh', ratings)]:
        path = root / f'{name}.json'
        payload = json.dumps(data).encode()
        path.write_bytes(payload)
        files[name] = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()}
    return files


class AuditMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic_inputs()
        cls.baseline = execute(cls.inputs)

    def test_clean_control_qualifies_on_synthetic_judgments(self):
        full = self.baseline['pools']['full']
        self.assertEqual(self.baseline['offline_status'], 'ELIGIBLE_FOR_BLIND')
        self.assertEqual(full['metrics'][runner.TRIAL]['negative_ranks'], [15, 16])
        checker.verify(self.baseline)

    def test_count_preserving_swapped_labels_disagree_with_raw_judgments(self):
        inputs = copy.deepcopy(self.inputs)
        ratings = inputs[-1]
        ratings[0]['label'], ratings[-2]['label'] = ratings[-2]['label'], ratings[0]['label']
        swapped = execute(inputs)
        self.assertEqual(self.baseline['pools']['full']['counts'], swapped['pools']['full']['counts'])
        with tempfile.TemporaryDirectory() as tmp:
            # Genuine matched judgments can produce a real, auditable bad ranking.
            swapped['source_files'] = write_raw(Path(tmp), inputs)
            self.assertEqual(swapped['pools']['full']['metrics'][runner.TRIAL]['negative_ranks'][0], 1)
            self.assertEqual(swapped['offline_status'], 'NO_GO_OFFLINE')
            checker.verify(swapped)
            swapped['source_files'] = write_raw(Path(tmp), self.inputs)
            with self.assertRaisesRegex(ValueError, 'source.*label'):
                checker.verify(swapped)

    def test_cli_detects_swapped_source_even_after_rerunning_and_rehashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_path = root / 'audit.json'
            lock_path = root / 'trusted-digests.json'
            files = write_raw(root, self.inputs)
            lock_path.write_text(json.dumps({key: info['sha256'] for key, info in files.items()}))
            # Generate through the regular file-based entry point with relative
            # inputs; then run the checker from the repository, outside root.
            run = [sys.executable, str(Path(runner.__file__)), '--likes', 'likes.json',
                   '--rejects', 'rejects.json', '--catalog', 'catalog.json',
                   '--fresh', 'fresh.json', '--output', audit_path.name]
            run_result = subprocess.run(run, cwd=root, capture_output=True, text=True)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)
            cmd = [sys.executable, str(Path(checker.__file__)), str(audit_path),
                   '--source-lock', str(lock_path)]
            clean = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(clean.returncode, 0, clean.stderr)
            inputs = copy.deepcopy(self.inputs)
            inputs[-1][0]['label'], inputs[-1][-2]['label'] = 'reject', 'like'
            write_raw(root, inputs)
            run_result = subprocess.run(run, cwd=root, capture_output=True, text=True)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)
            corrupt = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(corrupt.returncode, 0)
            self.assertIn('trusted source digest', corrupt.stderr)

    def test_cli_fails_closed_on_stripped_evidence_for_promoted_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_path = root / 'audit.json'
            report = copy.deepcopy(self.baseline)
            self.assertEqual(report['offline_status'], 'ELIGIBLE_FOR_BLIND')
            cmd = [sys.executable, str(Path(checker.__file__)), str(audit_path)]
            audit_path.write_text(json.dumps(report))
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('source files', result.stderr)
            report['source_files'] = write_raw(root, self.inputs)
            report['pools']['full'].pop('channel_recordings', None)
            audit_path.write_text(json.dumps(report))
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('channel evidence', result.stderr)
            report['missing_inputs'] = {'fresh': 'missing'}
            audit_path.write_text(json.dumps(report))
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unavailable-input verdict', result.stderr)

    def test_known_reject_planted_first_vetoes_otherwise_eligible_trial(self):
        planted = plant_known_reject(self.inputs)
        metrics = planted['pools']['full']['metrics'][runner.TRIAL]
        self.assertEqual(self.baseline['offline_status'], 'ELIGIBLE_FOR_BLIND')
        self.assertEqual(metrics['negative_ranks'], [1, 16])
        self.assertEqual(metrics['negative_top_k'], 1)
        self.assertEqual(planted['offline_status'], 'NO_GO_OFFLINE')
        self.assertEqual(checker._offline_status(planted, 10), 'NO_GO_OFFLINE')

    def test_forged_rank_one_score_is_detected_even_with_honest_no_go(self):
        planted = plant_known_reject(self.inputs)
        self.assertEqual(planted['offline_status'], 'NO_GO_OFFLINE')
        with self.assertRaisesRegex(ValueError, 'channel score'):
            checker.verify(planted)

    def test_top_ten_reject_alone_vetoes_offline_eligibility(self):
        # Isolate the policy veto. The rank-1 injection above also degrades NDCG.
        report = copy.deepcopy(self.baseline)
        report['pools']['full']['metrics'][runner.TRIAL]['negative_top_k'] = 1
        self.assertEqual(runner.offline_decision(report['pools'], fresh=True), 'NO_GO_OFFLINE')
        self.assertEqual(checker._offline_status(report, 10), 'NO_GO_OFFLINE')

    def test_corrupt_pool_row_fails_identity_join(self):
        edited = copy.deepcopy(self.baseline)
        rows = edited['pools']['full']['rows']
        rows[0]['row_id'] = rows[1]['row_id']
        with self.assertRaisesRegex(ValueError, 'pool row_id mismatch'):
            checker.verify(edited)

    def test_edited_raw_catalog_feature_fails_source_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = copy.deepcopy(self.baseline)
            audit['source_files'] = write_raw(Path(tmp), self.inputs)
            path = Path(audit['source_files']['catalog']['path'])
            rows = json.loads(path.read_bytes())
            rows[0]['librosa']['bpm'] += 50
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'source.*digest'):
                checker.verify(audit)

    def test_run_binds_catalog_digest_to_bytes_actually_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = write_raw(root, self.inputs)
            catalog = Path(files['catalog']['path'])
            output = root / 'audit.json'
            run = runner.run_experiment

            def edit_after_scores(*args, **kwargs):
                report = run(*args, **kwargs)
                rows = json.loads(catalog.read_bytes())
                rows[0]['librosa']['bpm'] += 50
                catalog.write_text(json.dumps(rows))
                return report

            argv = [arg for name, info in files.items() for arg in ('--' + name, info['path'])]
            with patch.object(runner, 'run_experiment', side_effect=edit_after_scores), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = runner.main([*argv, '--output', str(output)])
            self.assertEqual(code, 0)
            report = json.loads(output.read_bytes())
            self.assertEqual(report['source_files']['catalog']['sha256'], files['catalog']['sha256'])
            with self.assertRaisesRegex(ValueError, 'source.*digest'):
                checker.verify(report)

    def test_flipped_popular_weight_fails_score_replay(self):
        with patch.dict(runner.embeat.CHANNEL_WEIGHTS, {'popular': (-1.0, 0.8)}):
            flipped = execute(self.inputs)
        before = {r['recording_id']: r['scores'][runner.TRIAL]
                  for r in self.baseline['pools']['full']['rows']}
        changed = [r for r in flipped['pools']['full']['rows']
                   if r['scores'][runner.TRIAL] != before[r['recording_id']]]
        self.assertGreater(len(changed), 0)
        with self.assertRaisesRegex(ValueError, 'channel score'):
            checker.verify(flipped)

    def test_two_claimed_keeps_disagree_with_locked_empty_live_file(self):
        rows = self.baseline['pools']['full']['rows']
        verified = {r['recording_id']: str(i).zfill(22) for i, r in enumerate(rows, 1)}
        deck, reveal = runner.prepare_blind(self.baseline, verified, set(), set())
        rounds = [dict(trial_id=self.baseline['trial_id'], reveal=reveal,
                       ratings={entry['token']: True for entry in deck})]
        audit = execute(self.inputs, live=rounds)
        self.assertEqual((audit['live']['keeps'], audit['live']['n']), (2, 2))
        checker.verify(audit)  # In-memory replay can verify counts without raw files.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit['source_files'] = write_raw(root, self.inputs)
            path = root / 'rated-rounds.json'
            path.write_bytes(b'[]')
            empty_hash = hashlib.sha256(b'[]').hexdigest()
            audit['source_files']['live'] = dict(path=str(path), sha256=empty_hash)
            with self.assertRaisesRegex(ValueError, 'live.*source|source.*live'):
                checker.verify(audit)
            with self.assertRaisesRegex(ValueError, 'live.*source|source.*live'):
                checker.verify(audit, trusted_digests={'live': empty_hash})
            checker.verify(audit, structure_only=True)
            del audit['source_files']['live']
            with self.assertRaisesRegex(ValueError, 'live.*source|source.*live'):
                checker.verify(audit)
            raw = json.dumps(rounds).encode()
            path.write_bytes(raw)
            audit['source_files']['live'] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
            checker.verify(audit)
            numeric = copy.deepcopy(rounds)
            numeric[0]['ratings'] = {token: 1 for token in numeric[0]['ratings']}
            raw = json.dumps(numeric).encode()
            path.write_bytes(raw)
            audit['source_files']['live']['sha256'] = hashlib.sha256(raw).hexdigest()
            with self.assertRaisesRegex(ValueError, 'live.*source|source.*live'):
                checker.verify(audit)

    def test_checker_vetoes_sample_ndcg_reversal_independently(self):
        report = copy.deepcopy(self.baseline)
        effect = report['pools']['sample_42']['deltas']['v6_frozen']
        self.assertGreater(effect['ndcg_at_k'], 0)
        effect['ndcg_at_k'] *= -1
        self.assertEqual(runner.offline_decision(report['pools'], fresh=True), 'NO_GO_OFFLINE')
        self.assertEqual(checker._offline_status(report, 10), 'NO_GO_OFFLINE')


if __name__ == '__main__':
    unittest.main()
