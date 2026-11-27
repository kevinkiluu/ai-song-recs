"""Behavioral checks for the fixed-score Bliss pool comparison."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scoring import feat_vector

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'v3_scores_306.json'


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'audit.bliss_sensitivity', *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def run_report(output, *args):
    result = cli(*args, '--output', output)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_bytes())


def test_archive_slices_reuse_frozen_scores_and_keep_all_judgments(tmp_path):
    report = run_report(tmp_path / 'replay.json', '--archive', ARCHIVE)
    assert report['scope'] == 'archived_sample_only'
    assert report['full_pool_reversal'] is None
    assert report['archived_rows']['count'] == 306
    assert report['archived_rows']['metrics']['v6b']['ndcg_at_10'] == pytest.approx(.36865805)
    primary = report['reference']
    assert primary['counts'] == {'like': 93, 'reject': 12, 'unlabeled': 200}
    assert primary['metrics']['bliss_v2_blend']['ndcg_at_10'] == pytest.approx(.538885692)
    assert primary['metrics']['v6b']['ndcg_at_10'] == pytest.approx(.363623707)
    assert report['input_sha256'][ARCHIVE.name] == hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    encoded = json.dumps(report['score_panel'], sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    assert report['score_panel_sha256'] == hashlib.sha256(encoded).hexdigest()
    rows = {r['recording_id']: r for r in report['score_panel']}
    judged = {key for key, row in rows.items() if row['label'] != 'unlabeled'}
    draws = report['samples']
    assert [s['seed'] for s in draws] == [42, 43, 44, 45, 46]
    assert len({tuple(s['members']) for s in draws}) == 5
    for sample in draws:
        assert sample['counts'] == {'like': 93, 'reject': 12, 'unlabeled': 100}
        assert set(sample['members']) >= judged
        for method, result in sample['metrics'].items():
            descending = method == 'bliss_v2_blend'
            ordered = sorted(sample['members'], key=lambda key: (
                rows[key]['scores'][method] * (-1 if descending else 1), key))
            assert result['top10'] == ordered[:10]
        baseline = set(sample['metrics']['v6b']['top10'])
        for method, delta in sample['vs_v6b'].items():
            challenger = set(sample['metrics'][method]['top10'])
            assert set(delta['entrants']) == challenger - baseline
            assert set(delta['displaced']) == baseline - challenger


def raw_snapshot(directory):
    directory.mkdir()
    data = directory / 'data'
    data.mkdir()

    def row(dz, energy, pos=None):
        rec = dict(deezer_id=dz, spotify_id=f'sp{dz}', artist='Artist', genre_bucket='E',
                   reccobeats={'energy': energy},
                   librosa=dict(bpm=123, brightness_mean=2668, rms_mean=.320))
        return rec if pos is None else dict(rec, position=pos)

    likes = [row(1, 0, 130), row(2, 0, 131), row(3, .1234, 0), row(1, 0, 1)]
    # 1 is both held out and in the frozen training set. Neither panel may use it.
    likes += [row(4, 0, 132), dict(row(3, .1234, 2), spotify_id='alias')]
    rejects = [row(5, .5)]
    catalog = [row(i, i / 100) for i in range(6, 19)] + [row(100, .9)]
    vectors = {i: {'deezer_id': i, 'bliss23': [v] + [0.] * 22, 'source': 'preview'}
               for i, v in {1: 99, 2: 0, 3: 4, 4: 2, 5: 8, **{n: n for n in range(6, 19)}}.items()}
    for path, obj in [(directory / 'features_v2.json', {'tracks': likes}),
                      (directory / 'model_v2.json', {'gmm_k': 1, 'clusters': []}),
                      (data / 'negatives_features.json', rejects),
                      (data / 'pool_features.json', catalog), (data / 'bliss23.json', vectors)]:
        path.write_text(json.dumps(obj))
    size = len(feat_vector({})[0])
    zeros = np.zeros(size)
    lr = zeros.copy()
    lr[1] = 2
    np.savez(directory / 'model_v2.npz', imputer_median=zeros, scaler_mean=zeros,
             scaler_scale=np.ones(size), pca_components=zeros[None, :], pca_mean=zeros,
             centroids=[[0.]], lr_coef=lr[None, :], lr_intercept=[0.],
             negdist_mean=0., negdist_std=1., lr_mean=0., lr_std=1.)


def test_raw_scores_from_model_and_train_only_bliss_and_covers_every_eligible_row(tmp_path):
    raw = tmp_path / 'snapshot'
    raw_snapshot(raw)
    report = run_report(tmp_path / 'out.json', '--raw-root', raw, '--sample-size', 3)
    assert report['scope'] == 'full_eligible_supplied_pool'
    assert report['reference']['counts'] == {'like': 1, 'reject': 1, 'unlabeled': 13}
    rows = {r['recording_id']: r for r in report['score_panel']}
    assert set(rows) == {f'deezer:{i}' for i in [3, 5, *range(6, 19)]}
    assert report['eligibility']['bliss_seed_recordings'] == 2
    assert report['eligibility']['excluded']['missing preview Bliss vector'] == 1
    assert rows['deezer:3']['scores']['bliss_fixed'] == 1.5
    assert rows['deezer:3']['scores']['v2'] == .123
    assert rows['deezer:3']['scores']['v6b'] == 0.
    fixed = np.array([r['scores']['bliss_fixed'] for r in rows.values()])
    ml = np.array([r['scores']['v2'] for r in rows.values()])
    expected = ((ml - ml.mean()) / ml.std() - (fixed - fixed.mean()) / fixed.std()) / 2
    assert [r['scores']['bliss_v2_blend'] for r in rows.values()] == pytest.approx(expected)
    assert all(len(s['members']) == 5 for s in report['samples'])
    assert report['samples'][0]['members'] == ['deezer:3', 'deezer:5', 'deezer:6', 'deezer:15', 'deezer:17']


def test_input_is_preserved_when_output_aliases_it(tmp_path):
    copied = tmp_path / 'archive.json'
    copied.write_bytes(ARCHIVE.read_bytes())
    alias = tmp_path / 'alias.json'
    alias.symlink_to(copied)
    result = cli('--archive', copied, '--output', alias)
    assert result.returncode != 0 and 'exists' in result.stderr.lower()
    assert copied.read_bytes() == ARCHIVE.read_bytes()


def test_raw_request_names_missing_inputs_without_substituting_archive(tmp_path):
    result = cli('--raw-root', tmp_path)
    assert result.returncode != 0
    for name in ('pool_features.json', 'negatives_features.json', 'bliss23.json', 'model_v2.npz'):
        assert name in result.stderr


def test_opposite_signs_are_reported_for_both_metrics_and_ties_do_not_reverse(tmp_path):
    source = []
    for i in range(23):
        kind = 'heldout' if i < 10 else 'negative' if i < 15 else 'distractor'
        # v6 puts all five rejects first. Fixed Bliss puts all eight distractors first.
        baseline = {'negative': 0, 'heldout': 1, 'distractor': 2}[kind]
        fixed = {'distractor': 0, 'heldout': 1, 'negative': 2}[kind]
        source.append(dict(deezer_id=i + 1, kind=kind, relevant=int(i < 10),
                           scores=dict(v6b=baseline, v2=1., bliss_fixed=fixed,
                                       bliss_adaptive=baseline, bliss_v2_blend=-fixed)))
    archive = tmp_path / 'scores.json'
    archive.write_text(json.dumps({'items': source}))
    report = run_report(tmp_path / 'out.json', '--archive', archive, '--sample-size', 2)
    fixed = report['reference']['vs_v6b']['bliss_fixed']
    assert fixed['p_sign'] == fixed['ndcg_sign'] == -1
    assert fixed['p_at_10'] == -.3
    for sample in report['samples']:
        assert sample['vs_v6b']['bliss_fixed']['p_at_10'] == .3
        assert sample['vs_v6b']['bliss_fixed']['ndcg_sign'] == 1
    for metric in ('p_at_10', 'ndcg_at_10'):
        assert report['opposite_sign_seeds_vs_reference']['bliss_fixed'][metric] == list(range(42, 47))
        assert report['opposite_sign_seeds_vs_reference']['bliss_adaptive'][metric] == []


def test_excludes_unplaced_liked_component_from_fit_and_panel(tmp_path):
    raw = tmp_path / 'snapshot'
    raw_snapshot(raw)
    likes_file = raw / 'features_v2.json'
    doc = json.loads(likes_file.read_text())
    # The same known like has no usable position on either of its source aliases.
    for r in doc['tracks']:
        if r['deezer_id'] == 3:
            r['position'] = None
    doc['tracks'].append(dict(doc['tracks'][2], deezer_id=10, spotify_id='sp10', position=4))
    likes_file.write_text(json.dumps(doc))
    report = run_report(tmp_path / 'out.json', '--raw-root', raw)
    assert 'deezer:3' not in report['reference']['members']
    assert report['reference']['counts']['like'] == 1
    assert report['eligibility']['bliss_seed_recordings'] == 2


def test_records_historical_provenance_and_known_fit_rejects(tmp_path):
    replay = run_report(tmp_path / 'archive.json', '--archive', ARCHIVE)
    assert replay['matches_pinned_input']['v3_scores_306.json'] is True
    assert len(replay['historical_fit']['configured_reject_ids_in_panel']) == 12
    assert replay['historical_fit']['inferential'] is False

    raw = tmp_path / 'snapshot'
    raw_snapshot(raw)
    negatives = raw / 'data' / 'negatives_features.json'
    docs = json.loads(negatives.read_bytes())
    docs[0]['spotify_id'] = '7zDu3l7DDlfSlt20YMKkWo'  # archived Martini Blue fit reject
    negatives.write_text(json.dumps(docs))
    full = run_report(tmp_path / 'full.json', '--raw-root', raw)
    assert full['historical_fit']['configured_reject_ids_in_panel'] == ['deezer:5']
    assert full['matches_pinned_input'] == {'model_v2.json': False, 'features_v2.json': False}


def test_matched_embedded_isrc_connects_bliss_seed_to_catalog_alias(tmp_path):
    raw = tmp_path / 'snapshot'
    raw_snapshot(raw)
    likes_file = raw / 'features_v2.json'
    catalog_file = raw / 'data' / 'pool_features.json'
    likes, catalog = json.loads(likes_file.read_bytes()), json.loads(catalog_file.read_bytes())
    seed = next(r for r in likes['tracks'] if r['deezer_id'] == 2)
    seed['spotify_id'] = '1' * 22
    seed['reccobeats'].update(href='https://open.spotify.com/track/' + '1' * 22, isrc='USAAA2600001')
    catalog[0]['isrc'] = 'USAAA2600001'
    likes_file.write_text(json.dumps(likes))
    catalog_file.write_text(json.dumps(catalog))
    report = run_report(tmp_path / 'out.json', '--raw-root', raw)
    assert report['reference']['counts']['unlabeled'] == 12
    assert 'deezer:6' not in report['reference']['members']
    assert report['eligibility']['bliss_seed_recordings'] == 2


@pytest.mark.parametrize('catalog_count', [0, 5])
def test_full_pool_without_a_distinct_draw_is_unable_to_test_reversal(tmp_path, catalog_count):
    raw = tmp_path / 'snapshot'
    raw_snapshot(raw)
    catalog_path = raw / 'data' / 'pool_features.json'
    catalog_path.write_text(json.dumps(json.loads(catalog_path.read_bytes())[:catalog_count]))
    report = run_report(tmp_path / 'out.json', '--raw-root', raw)
    assert report['full_pool_reversal'] is None
    assert report['opposite_sign_seeds_vs_reference'] is None
    assert report['draw_membership'] == {
        'status': 'not_testable',
        'distinct_sample_memberships': 1,
        'distinct_sample_memberships_other_than_reference': 0,
        'seeds_with_membership_different_from_reference': [],
        'eligible_distractors': catalog_count,
    }
    assert report['eligibility']['catalog_status'] == ('empty_input' if catalog_count == 0 else 'eligible')
    assert len(report['reference']['members']) == 2 + catalog_count
    assert all(s['membership_matches_reference'] and s['vs_v6b'] for s in report['samples'])


def test_catalog_with_no_eligible_distractors_differs_from_empty_input(tmp_path):
    raw = tmp_path / 'snapshot'
    raw_snapshot(raw)
    catalog_path = raw / 'data' / 'pool_features.json'
    catalog_path.write_text(json.dumps(json.loads(catalog_path.read_bytes())[-1:]))  # no Bliss vector
    report = run_report(tmp_path / 'report.json', '--raw-root', raw)
    assert report['eligibility']['catalog_status'] == 'no_eligible_distractors'
    assert report['draw_membership']['status'] == 'not_testable'
    assert report['draw_membership']['eligible_distractors'] == 0
    assert report['full_pool_reversal'] is None
