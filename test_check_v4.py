import copy
import unittest

from check_v4 import verify
from integrate_v4 import run_experiment
from test_integrate_v4 import fixture


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = run_experiment(*fixture(), cutoff=10, sample_size=4)

    def test_independent_checker_accepts_the_emitted_record(self):
        verify(self.audit)

    def test_checker_rejects_rank_and_metric_tampering(self):
        audit = copy.deepcopy(self.audit)
        audit['pools']['full']['rows'][0]['ranks']['v2_refit'] = 0
        with self.assertRaisesRegex(ValueError, 'rank'):
            verify(audit)

        audit = copy.deepcopy(self.audit)
        audit['pools']['full']['metrics']['v6_frozen']['ndcg_at_k'] = -1
        with self.assertRaisesRegex(ValueError, 'ndcg_at_k'):
            verify(audit)

    def test_checker_disallows_go_and_unmatched_sample_rows(self):
        audit = copy.deepcopy(self.audit)
        audit['status'] = 'GO'
        with self.assertRaisesRegex(ValueError, 'verdict'):
            verify(audit)

        audit = copy.deepcopy(self.audit)
        audit['pools']['sample_42']['rows'][0]['label'] = 'unlabeled'
        with self.assertRaisesRegex(ValueError, 'label'):
            verify(audit)

    def test_checker_rejects_training_identity_in_evaluation(self):
        audit = copy.deepcopy(self.audit)
        train = next(r for r in audit['manifest'] if r['partition'] == 'train')
        train['recording_id'] = audit['pools']['full']['rows'][0]['recording_id']
        with self.assertRaisesRegex(ValueError, 'identity|recording'):
            verify(audit)


if __name__ == '__main__':
    unittest.main()
