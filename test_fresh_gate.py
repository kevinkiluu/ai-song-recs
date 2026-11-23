"""Reconstruct real input shapes from generated, non-listener test records."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from evaluate_v4 import build_manifest, evaluation_pool


ROOT = Path(__file__).resolve().parent
CLI = ROOT / 'audit' / 'fresh_gate.py'


def sid(n):
    return str(n).zfill(22)


def track(n):
    return {'title': f'Track {n} [Live]', 'artist': 'First, Guest', 'spotify_id': sid(n)}


def code(n):
    return f'USABC26{n:05d}'


def audio(n):
    return dict(track(n), spotify_id=None, deezer_id=n, isrc=code(n),
                librosa={'bpm': 120, 'rms_mean': .3, 'brightness_mean': 2000})


def tsv(rows):
    return ''.join(f"{r['title']}\t{r['artist']}\tspotify:track:{r['spotify_id']}\n" for r in rows)


class Bundle:
    def __init__(self, root):
        self.root = root
        self.sources = {}
        self.manifest = {'dataset': 'fixture', 'challenger_revision': 'a'*40, 'frozen_at': '2026-10-01T00:00:00Z',
                         'library_id': 'spotify:user:owner:collection', 'sources': self.sources,
                         'rounds': []}
        self.historic = [dict(track(4), position=0, deezer_id=4)]
        self.rejects = [dict(track(6), deezer_id=6)]
        self.catalog = [audio(n) for n in range(1, 13)]
        rb = {code(n): {'isrc': code(n), 'href': f'https://open.spotify.com/track/{sid(n)}'}
              for n in range(1, 13)}
        self.source('likes', {'tracks': self.historic})
        self.source('rejects', self.rejects)
        self.source('catalog', self.catalog)
        self.source('crosswalk', rb)
        self.source('original_likes', tsv(self.historic), at='2026-09-30T12:00:00Z',
                    collection=self.manifest['library_id'], complete=True)
        self.round('v8', [1, 2, 3, 4, 5, 6, 7, 20], [1, 2, 4], [1, 4], [5])

    def source(self, name, content, **metadata):
        ext = '.txt' if isinstance(content, str) else '.json'
        data = (content if ext == '.txt' else json.dumps(content)).encode()
        path = self.root / (name + ext)
        path.write_bytes(data)
        self.sources[name] = dict(path=path.name, sha256=sha256(data).hexdigest(),
                                  role=name, **metadata)

    def round(self, name, offered, retained, liked, history, day=5):
        playlist = sid(100 + day)
        spec = dict(name=name, playlist_id=playlist, cohort='post_freeze', untouched=True,
                    cutoff=f'2026-10-{day:02}T18:00:00Z')
        for role, nums, hour in [('offered', offered, 10), ('retained', retained, 16),
                                 ('cutoff_likes', liked, 18), ('prior_likes', [], 9),
                                 ('prior_history', history, 9)]:
            rows = [track(n) for n in nums]
            key = f'{name}_{role}'
            source = rows if role == 'offered' else tsv(rows)
            collection = (self.manifest['library_id'] if role in ('cutoff_likes', 'prior_likes') else
                          'prior-recommendations' if role == 'prior_history' else playlist)
            self.source(key, source, at=f'2026-10-{day:02}T{hour:02}:00:00Z',
                        collection=collection, complete=True)
            self.sources[key].update(role=role, round=name)
            spec[role] = key
        self.manifest['rounds'].append(spec)

    def call(self, digest=None, out=None):
        path = self.root / 'bundle.json'
        path.write_text(json.dumps(self.manifest))
        return subprocess.run([sys.executable, str(CLI), str(path), '--sha256',
                               digest or sha256(path.read_bytes()).hexdigest(),
                               '--output-dir', str(out or self.root / 'out')],
                              capture_output=True, text=True)

    def result(self):
        result = self.call()
        assert result.returncode == 0, result.stderr
        return [json.loads((self.root / 'out' / name).read_text()) for name in
                ('gate.json', 'fresh.json')]


class FreshGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = Bundle(self.root)

    def test_catalog_labels_replay_exactly_in_v4_and_ok_never_becomes_a_like(self):
        gate, fresh = self.bundle.result()
        counts = gate['rounds'][0]
        self.assertEqual([counts[k] for k in ('offered', 'kept', 'left', 'removed')], [8, 2, 1, 5])
        self.assertEqual(gate['funnel']['joined'], 7)
        self.assertEqual(gate['funnel']['already_known'], 3)
        self.assertEqual(gate['eligible'], {'like': 1, 'reject': 2})
        self.assertEqual(gate['shortfall'], {'like': 9, 'reject': 0})
        self.assertEqual(gate['status'], 'INSUFFICIENT_EVIDENCE')
        pool = evaluation_pool(build_manifest(self.bundle.historic, self.bundle.rejects,
                                              self.bundle.catalog, fresh=fresh))
        self.assertEqual({(r['record']['deezer_id'], r['label']) for r in pool
                          if r['label'] != 'unlabeled'}, {(1, 'like'), (3, 'reject'), (7, 'reject')})

    def test_version_mismatch_and_clashing_tiers_quarantine_records(self):
        self.bundle.catalog[0]['title'] = 'Track 1'
        self.bundle.source('catalog', self.bundle.catalog)
        self.bundle.round('v9', [3, 7], [3], [], [], day=6)
        gate, fresh = self.bundle.result()
        self.assertEqual(gate['funnel']['conflicting'], 1)
        self.assertEqual(gate['eligible'], {'like': 0, 'reject': 1})
        self.assertEqual([r['spotify_id'] for r in fresh], [sid(7)])

    def test_incomplete_likes_cannot_establish_a_reject(self):
        self.bundle.sources['v8_cutoff_likes']['complete'] = False
        gate, fresh = self.bundle.result()
        self.assertEqual(gate['rounds'][0]['unknown'], 6)
        self.assertEqual(gate['eligible'], {'like': 1, 'reject': 0})
        self.assertEqual(gate['shortfall']['reject'], 2)

    def test_full_gate_only_qualifies_for_measurement_and_deduplicates_reoffers(self):
        self.bundle.manifest['rounds'].clear()
        self.bundle.source('likes', {'tracks': []})
        self.bundle.source('original_likes', '', at='2026-09-01T00:00:00Z',
                           collection=self.bundle.manifest['library_id'], complete=True)
        self.bundle.source('rejects', [])
        self.bundle.round('later', list(range(1,13)), [], list(range(1,11)), [])
        self.bundle.round('repeat', [1], [1], [1], [], day=6)
        gate, fresh = self.bundle.result()
        self.assertEqual(gate['eligible'], {'like': 10, 'reject': 2})
        self.assertEqual(gate['status'], 'READY_FOR_OFFLINE_EVALUATION')
        self.assertEqual(len(fresh), 12)

    def test_prefreeze_or_tuned_rounds_do_not_supply_postfreeze_evidence(self):
        self.bundle.manifest['frozen_at'] = '2026-10-05T12:00:00Z'
        self.bundle.round('later', [8, 9], [8], [8], [], day=6)
        self.bundle.manifest['rounds'][-1]['untouched'] = False
        gate, fresh = self.bundle.result()
        self.assertEqual(fresh, [])
        self.assertEqual(gate['shortfall'], {'like': 10, 'reject': 2})

    def test_cheaper_title_match_never_bridges_across_different_provider_ids(self):
        x = json.loads((self.root / 'crosswalk.json').read_text())
        x[code(3)]['isrc'] = code(7)
        x[code(2)]['href'] = 'https://example.com/track/' + sid(2)
        self.bundle.source('crosswalk', x)
        gate, fresh = self.bundle.result()
        self.assertNotIn(sid(3), {r['spotify_id'] for r in fresh})

    def test_fixture_cannot_claim_the_original_590_catalog(self):
        self.bundle.manifest['dataset'] = 'original_v5'
        self.assertNotEqual(self.bundle.call().returncode, 0)
        self.assertFalse((self.root / 'out').exists())

    def test_already_saved_before_offer_is_not_a_new_like(self):
        self.bundle.source('v8_prior_likes', tsv([track(1)]), complete=True,
                           collection=self.bundle.manifest['library_id'], at='2026-10-05T09:00:00Z')
        self.bundle.sources['v8_prior_likes'].update(role='prior_likes', round='v8')
        gate, _ = self.bundle.result()
        self.assertEqual(gate['eligible'], {'like': 0, 'reject': 2})

    def test_ambiguous_reoffer_cannot_leave_a_matching_component_eligible(self):
        self.bundle.round('v9', [7], [], [], [], day=6)
        self.bundle.source('v9_offered', [dict(track(7), title='Different recording')], complete=True,
                           collection=sid(106), at='2026-10-06T10:00:00Z')
        self.bundle.sources['v9_offered'].update(role='offered', round='v9')
        _, fresh = self.bundle.result()
        self.assertNotIn(sid(7), {row['spotify_id'] for row in fresh})

    def test_locked_preview_bytes_quarantine_a_historical_recording_alias(self):
        # One historical reject and a separately identified offered recording
        # ultimately have the same cached audio bytes.
        for n in (6, 7):
            filename = f'{n}.mp3'
            content = b'shared synthetic preview'
            (self.root / filename).write_bytes(content)
            self.bundle.sources[f'preview/{n}'] = dict(path=filename, role='preview',
                                                        sha256=sha256(content).hexdigest())
        gate, fresh = self.bundle.result()
        self.assertNotIn(sid(7), {row['spotify_id'] for row in fresh})
        self.assertEqual(gate['verified_preview_files'], 2)

    def test_disputed_saved_track_cannot_supply_a_like_for_a_spotify_only_catalog_row(self):
        self.bundle.catalog[0] = dict(track(1), librosa=audio(1)['librosa'])
        self.bundle.source('catalog', self.bundle.catalog)
        self.bundle.source('v8_cutoff_likes', tsv([dict(track(1), title='Another song'), track(4)]),
                           complete=True, collection=self.bundle.manifest['library_id'],
                           at='2026-10-05T18:00:00Z')
        self.bundle.sources['v8_cutoff_likes']['role'] = 'cutoff_likes'
        gate, fresh = self.bundle.result()
        self.assertNotIn(sid(1), {r['spotify_id'] for r in fresh})
        self.assertEqual(gate['rounds'][0]['unknown'], 1)

    def test_prior_history_text_excludes_an_unmapped_alternate_spotify_id(self):
        self.bundle.source('v8_prior_history', tsv([dict(track(1), spotify_id=sid(76)), track(5)]),
                           complete=True, collection='prior-recommendations', at='2026-10-05T09:00:00Z')
        self.bundle.sources['v8_prior_history'].update(role='prior_history', round='v8')
        gate, fresh = self.bundle.result()
        self.assertNotIn(sid(1), {r['spotify_id'] for r in fresh})
        self.assertEqual(gate['funnel']['already_known'], 4)

    def test_new_gate_cannot_mix_an_original_replay_like_into_new_evidence(self):
        self.bundle.manifest['rounds'][0]['cohort'] = 'historical_replay'
        self.bundle.round('later', [8, 9], [8], [8], [], day=6)
        gate, fresh = self.bundle.result()
        self.assertEqual({r['spotify_id'] for r in fresh}, {sid(8), sid(9)})
        self.assertEqual(gate['eligible_by_cohort']['historical_replay'], {'like': 1, 'reject': 2})
        self.assertEqual(gate['shortfall'], {'like': 9, 'reject': 1})

    def test_explicit_historical_replay_counts_the_original_untouched_labels(self):
        self.bundle.manifest['mode'] = 'historical_replay'
        self.bundle.manifest['frozen_at'] = '2026-10-09T00:00:00Z'
        self.bundle.manifest['rounds'][0]['cohort'] = 'historical_replay'
        gate, fresh = self.bundle.result()
        self.assertEqual(gate['mode'], 'historical_replay')
        self.assertEqual({r['spotify_id'] for r in fresh}, {sid(1), sid(3), sid(7)})

    def test_prior_reject_with_a_new_like_is_both_known_and_conflicting(self):
        self.bundle.source('v8_cutoff_likes', tsv([track(1), track(4), track(6)]), complete=True,
                           collection=self.bundle.manifest['library_id'], at='2026-10-05T18:00:00Z')
        self.bundle.sources['v8_cutoff_likes']['role'] = 'cutoff_likes'
        gate, fresh = self.bundle.result()
        self.assertEqual(gate['funnel']['conflicting'], 1)
        self.assertEqual(gate['funnel']['already_known'], 3)
        self.assertNotIn(sid(6), {r['spotify_id'] for r in fresh})

    def test_new_private_checkout_directory_can_receive_outputs(self):
        with TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp) / 'private'
            result = self.bundle.call(out=out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((out / 'fresh.json').stat().st_mode & 0o777, 0o600)

    def test_manifest_bytes_sources_roles_timestamps_and_collection_are_bound(self):
        b = self.bundle
        with self.subTest('manifest'):
            self.assertNotEqual(b.call('0'*64).returncode, 0)
        for name, field, value in [('v8_offered', 'sha256', '0'*64),
                                   ('v8_retained', 'role', 'offered'),
                                   ('v8_retained', 'collection', sid(999)),
                                   ('v8_cutoff_likes', 'at', '2026-10-06T18:00:00Z'),
                                   ('v8_prior_history', 'at', '2026-10-06T18:00:00Z')]:
            with self.subTest(field=field, name=name):
                before = b.sources[name][field]
                b.sources[name][field] = value
                self.assertNotEqual(b.call().returncode, 0)
                self.assertFalse((self.root / 'out').exists())
                b.sources[name][field] = before
        b.result()
        original = (self.root / 'out' / 'fresh.json').read_bytes()
        self.assertNotEqual(b.call().returncode, 0)
        self.assertEqual(original, (self.root / 'out' / 'fresh.json').read_bytes())
