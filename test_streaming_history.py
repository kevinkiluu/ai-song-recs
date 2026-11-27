import hashlib
import json
from datetime import date
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from scoring import feat_vector


def audit():
    import importlib.util
    assert importlib.util.find_spec('audit.streaming_history'), 'extended-history audit missing'
    from audit import streaming_history
    return streaming_history


def sid(n):
    return str(n).zfill(22)


def stamp(day, hour=0):
    return f'2026-10-{day:02d}T{hour:02d}:00:00Z'


def play(n, ts, ms, **fields):
    return dict(ts=ts, spotify_track_uri='spotify:track:' + sid(n),
                ms_played=ms, spotify_episode_uri=None, skipped=False, **fields)


def file(path, value):
    path.write_text(json.dumps(value))
    return path


def test_import_overlapping_archive_utc_coverage_and_duration(tmp_path):
    rows = [play(1, stamp(1), 29999), play(1, stamp(2), 30000),
            play(1, stamp(3), 120000), play(2, stamp(4), 80000),
            play(3, 'not a timestamp', 60000),
            dict(play(3, stamp(4), 30000), spotify_track_uri='local'),
            play(4, stamp(4), True), dict(play(5, stamp(4), 30000), spotify_episode_uri='podcast'),
            play(5, stamp(10), 30000)]
    rows[1]['skipped'] = True
    file(tmp_path / 'endsong_0.json', rows)
    with ZipFile(tmp_path / 'my_spotify_data.zip', 'w') as z:
        z.writestr('Spotify Extended Streaming History/Streaming_History_Audio_2026_0.json',
                   json.dumps([dict(rows[1], ts='2026-10-01T20:00:00-04:00')]))
    history = audit().read_history([tmp_path], as_of=date(2026, 10, 5))
    counts = history.summary({sid(1): None, sid(2): 100000})
    assert counts['raw_rows'] == 10 and counts['valid_timestamp_rows'] == 9
    assert counts['valid_track_id_rows'] == 8
    assert counts['valid_ms_rows'] == 9
    assert counts['duplicate_events'] == counts['after_as_of'] == 1
    assert counts['exposures'] == 4 and counts['qualifying_plays'] == 3
    assert counts['repeat_qualifying_plays'] == counts['tracks_with_repeat_plays'] == 1
    assert counts['short_skips'] == 1 and counts['completions'] == 2
    assert counts['joined_events'] == 4 and counts['joined_track_fraction'] == 1
    assert counts['spotify_flagged_skips'] == 1


def test_overlap_conflict_basic_and_naive_date(tmp_path):
    module = audit()
    file(tmp_path / 'endsong_0.json', [play(1, stamp(1), 30000)])
    file(tmp_path / 'endsong_1.json', [play(1, stamp(1), 31000)])
    with pytest.raises(ValueError, match='conflict|ambiguous'):
        module.read_history([tmp_path])
    with pytest.raises(ValueError, match='basic|Account'):
        module.read_history([file(tmp_path / 'StreamingHistory0.json', [{'endTime': 'bad'}])])
    history = module.read_history([file(tmp_path / 'endsong_0.json',
                                        [play(1, '2026-10-01T12:45:00', 30000)])])
    assert history.summary({})['valid_timestamp_rows'] == 0


def test_absence_before_and_after_due_date_has_no_estimate(tmp_path):
    for check, status in [(date(2026, 11, 23), 'awaiting_nov_26_check'),
                          (date(2026, 11, 26), 'export_unavailable')]:
        result = audit().run([tmp_path], as_of=check)
        assert result['status'] == status and result['comparison'] is None


def make_panel(tmp_path, nblocks=1):
    # A tiny frozen but fully executable v2 artifact. Its first feature alone
    # controls rank. Likes sort last on v2 and have completions in history.
    n = len(feat_vector({})[0])
    np.savez(tmp_path / 'model_v2.npz', imputer_median=np.zeros(n), scaler_mean=np.zeros(n),
             scaler_scale=np.ones(n), pca_components=np.zeros((1, n)), pca_mean=np.zeros(n),
             centroids=[[0]], lr_coef=[np.eye(1, n)[0]], lr_intercept=[0],
             negdist_mean=0, negdist_std=0, lr_mean=0, lr_std=1)
    file(tmp_path / 'model_v2.json', dict(gmm_k=1, clusters=[], lr_n_pos=1, lr_n_neg=1))
    hashes = {f: hashlib.sha256((tmp_path / f).read_bytes()).hexdigest()
              for f in ('model_v2.npz', 'model_v2.json')}
    panel = dict(frozen_at=stamp(2), model_sha256=hashes,
                 training_recordings=[sid(950), sid(951)], blocks=[])
    plays = []
    for i in range(nblocks + 1):
        start = 4 + 2*i
        block = dict(role='train' if i == 0 else 'test', start=stamp(start),
                     end=stamp(start+1), judgment_source='timestamped explicit listener ratings',
                     rows=[])
        for j in range(12):
            track_id = 12*i + j
            label = 'like' if j >= 10 else 'reject' if j == 0 else 'ok'
            record = dict(spotify_id=sid(track_id), recording_id=sid(track_id),
                          label=label, judged_at=stamp(start, 12),
                          reccobeats=dict(danceability=0.2 if j < 10 else 0.1))
            block['rows'].append(record)
            for h in (1, 2, 3):
                plays.append(play(track_id, stamp(start-1, h),
                                  120000 if j >= 10 else 3000 if j == 0 else 30000))
        panel['blocks'].append(block)
    return panel, plays


def compare(tmp_path, panel, plays):
    history = audit().read_history([file(tmp_path / 'endsong_0.json', plays)])
    return audit().compare(history, panel, tmp_path, as_of=date(2026, 11, 1))


def test_earlier_fit_scores_shared_later_panel_and_ignores_future_behavior(tmp_path):
    panel, plays = make_panel(tmp_path)
    result = compare(tmp_path, panel, plays)
    row = result['blocks'][0]
    assert row['judged'] == 12 and row['exposed_before_start'] == 12
    assert row['fixed_v2']['p_at_10'] == row['fixed_v2']['ndcg_at_10'] == 0
    assert row['history_v2']['p_at_10'] == .2 and row['history_v2']['ndcg_at_10'] == 1
    assert row['fixed_v2']['rejects_at_10'] == 1 and row['history_v2']['rejects_at_10'] == 1
    assert result['uncertainty']['intervals'] is None
    extra = [play(12, stamp(6, h), 180000) for h in range(6)]
    after = compare(tmp_path, panel, plays + extra)
    assert result == after
    cmd = [sys.executable, '-m', 'audit.streaming_history', '--history', str(tmp_path),
           '--panel', str(file(tmp_path / 'panel.json', panel)), '--model-dir', str(tmp_path),
           '--as-of', '2026-11-01']
    cli = subprocess.run(cmd, capture_output=True, text=True)
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert json.loads(cli.stdout)['comparison']['blocks'][0] == row


@pytest.mark.parametrize('damage', ['late_training', 'reused_recording', 'v2_overlap',
                                    'score_forgery', 'unjudged', 'missing_features',
                                    'short_pool', 'overlap', 'missing_controls', 'no_history'])
def test_missing_or_leaking_evidence_never_scores(tmp_path, damage):
    panel, plays = make_panel(tmp_path)
    train, test = panel['blocks']
    row = test['rows'][0]
    if damage == 'late_training':
        train['rows'][0]['judged_at'] = test['start']
    elif damage == 'reused_recording':
        row['recording_id'] = train['rows'][0]['recording_id']
    elif damage == 'v2_overlap':
        row['recording_id'] = panel['training_recordings'][0]
    elif damage == 'score_forgery':
        panel['model_sha256']['model_v2.npz'] = '0'*64
    elif damage == 'unjudged':
        row['label'] = 'unsaved'
    elif damage == 'missing_features':
        row['reccobeats'] = {}
    elif damage == 'short_pool':
        test['rows'] = test['rows'][:9]
    elif damage == 'overlap':
        train['end'] = test['end']
    elif damage == 'missing_controls':
        row['label'] = 'ok'
    else:
        plays = []
    with pytest.raises(ValueError):
        compare(tmp_path, panel, plays)


def test_intervals_require_design_and_sufficient_disjoint_blocks(tmp_path):
    panel, plays = make_panel(tmp_path, 8)
    assert compare(tmp_path, panel, plays)['uncertainty']['intervals'] is None
    panel['independence_basis'] = 'Eight preselected independent sessions in this fixture.'
    result = compare(tmp_path, panel, plays)
    assert result['mean_paired_delta']['p_at_10'] == pytest.approx(.2)
    intervals = result['uncertainty']['intervals']
    assert set(intervals) == {'p_at_10', 'ndcg_at_10', 'rejects_at_10'}
    assert intervals['p_at_10'][0] < .2 < intervals['p_at_10'][1]
    assert intervals['rejects_at_10'][0] < 0 < intervals['rejects_at_10'][1]


def test_basic_account_files_next_to_extended_parts_do_not_mask_the_export(tmp_path):
    file(tmp_path / 'StreamingHistory_music_0.json', [{'endTime': '2026-01-02'}])
    file(tmp_path / 'endsong_1.json', [play(6, stamp(1), 30000)])
    assert audit().read_history([tmp_path]).summary({})['exposures'] == 1


def test_model_training_alias_cannot_enter_panel_under_another_recording_key(tmp_path):
    panel, plays = make_panel(tmp_path)
    panel['training_recordings'] = [
        dict(recording_id='reference-one', spotify_id=sid(950),
             spotify_aliases=[panel['blocks'][1]['rows'][0]['spotify_id']]),
        dict(recording_id='reference-two', spotify_id=sid(951))]
    with pytest.raises(ValueError, match='overlap'):
        compare(tmp_path, panel, plays)


def test_missing_or_duplicate_model_training_ids_do_not_authenticate_overlap(tmp_path):
    panel, plays = make_panel(tmp_path)
    panel['training_recordings'] = [sid(950), 'reference-two']
    with pytest.raises(ValueError, match='Spotify|identity'):
        compare(tmp_path, panel, plays)


def test_enriched_training_inventory_accepts_disjoint_recording_keys(tmp_path):
    panel, plays = make_panel(tmp_path)
    panel['training_recordings'] = [
        dict(recording_id='reference-one', spotify_id=sid(950)),
        dict(recording_id='reference-two', spotify_id=sid(951))]
    assert compare(tmp_path, panel, plays)['training_rows'] == 12


def test_empty_export_blocks_and_partial_coverage_remains_visible(tmp_path):
    module = audit()
    path = file(tmp_path / 'endsong_0.json', [])
    assert module.run([path], as_of=date(2026, 10, 25))['status'] == 'unusable_history'
    path = file(path, [play(7, stamp(4), 45000), play(12, 'bad', 30000)])
    result = module.run([path], as_of=date(2026, 10, 25))
    assert result['status'] == 'pending_judged_panel' and result['coverage']['exposures'] == 1


def test_conflicting_devices_are_ambiguous_even_when_play_length_matches(tmp_path):
    event = play(7, stamp(3), 30000)
    file(tmp_path / 'endsong_0.json', [dict(event, platform='web')])
    file(tmp_path / 'endsong_1.json', [dict(event, platform='mobile')])
    with pytest.raises(ValueError, match='ambiguous|conflict'):
        audit().read_history([tmp_path])


def test_local_files_stay_out_of_music_id_denominator_without_blocking_valid_exposures(tmp_path):
    event = play(7, stamp(3), 30000)
    path = file(tmp_path / 'endsong_0.json', [
        event, dict(event, spotify_track_uri='spotify:local:artist:album:title:240')])
    actual = audit().run([path], as_of=date(2026, 11, 1))
    assert actual['status'] == 'pending_judged_panel'
    assert actual['coverage']['nontrack_rows'] == 1
    assert actual['coverage']['valid_track_id_fraction'] == .5


def test_unresolved_events_only_gate_the_affected_judged_recordings(tmp_path):
    panel, plays = make_panel(tmp_path)
    unrelated = plays + [play(777, 'unknown', 80000), play(12, stamp(7), 'unknown')]
    assert compare(tmp_path, panel, unrelated)['blocks'][0]['exposed_before_start'] == 12
    with pytest.raises(ValueError, match='unresolved'):
        compare(tmp_path, panel, unrelated + [play(12, 'unknown', 80000)])


def test_scores_use_the_exact_model_bytes_that_passed_digest_check(tmp_path, monkeypatch):
    panel, plays = make_panel(tmp_path)
    read_bytes = Path.read_bytes

    def replace_after_read(path):
        payload = read_bytes(path)
        if path == tmp_path / 'model_v2.npz':
            path.write_bytes(b'replaced after read')
        return payload

    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    assert compare(tmp_path, panel, plays)['blocks'][0]['fixed_v2']['p_at_10'] == 0


def test_keep_all_planned_time_blocks_when_known_judgments_are_unbalanced(tmp_path):
    panel, plays = make_panel(tmp_path, 2)
    # The first test block happens to contain only OKs. The next has a like and reject.
    for row in panel['blocks'][1]['rows']:
        row['label'] = 'ok'
    first = panel['blocks'][1]
    first['rows'][0]['judged_at'] = first['start']
    result = compare(tmp_path, panel, plays)
    assert len(result['blocks']) == 2
    assert result['blocks'][0]['fixed_v2']['rejects_at_10'] == 0
    assert result['blocks'][0]['fixed_v2']['ndcg_at_10'] == 0
    assert result['mean_paired_delta']['p_at_10'] == pytest.approx(.1)


def test_embedded_spotify_matched_isrc_catches_track_alias_across_fit_and_test(tmp_path):
    panel, plays = make_panel(tmp_path)
    train, test = panel['blocks']
    for block in (train, test):
        row = block['rows'][0]
        row['reccobeats'].update(href='https://open.spotify.com/track/' + row['spotify_id'],
                                 isrc='USABC2600001')
    with pytest.raises(ValueError, match='overlap'):
        compare(tmp_path, panel, plays)


def test_history_from_different_identifiable_listeners_cannot_be_pooled(tmp_path):
    file(tmp_path / 'endsong_0.json', [dict(play(1, stamp(2), 50000), username='one')])
    file(tmp_path / 'endsong_1.json', [dict(play(1, stamp(3), 60000), username='two')])
    with pytest.raises(ValueError, match='listener'):
        audit().read_history([tmp_path])


def test_midnight_exclusive_end_includes_a_completed_historical_utc_day(tmp_path):
    panel, plays = make_panel(tmp_path)
    history = audit().read_history([file(tmp_path / 'endsong_0.json', plays)])
    result = audit().compare(history, panel, tmp_path, as_of=date(2026, 10, 6))
    assert result['blocks'][0]['judged'] == 12
