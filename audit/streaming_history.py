"""Local Spotify Extended Streaming History audit. Run python -m audit.streaming_history."""
import argparse
import fnmatch
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean
from zipfile import BadZipFile, ZipFile

import numpy as np

from evaluate_v4 import embedded_spotify_isrc, identity_tokens, ranking_metrics
from scoring import V2Model, feat_vector


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = Path.home() / 'workspace/audio-pipeline'
DUE = date(2026, 11, 26)
ID = re.compile('[a-zA-Z0-9]{22}')
PATTERNS = ('Streaming_History_Audio_*.json', 'endsong_*.json')
METRICS = {'p_at_10': 1, 'ndcg_at_10': 1, 'rejects_at_10': 10}


def timestamp(value):
    if isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value)
            if ts.tzinfo is not None:
                return ts.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            pass
    raise ValueError('timestamp needs an ISO date, time and UTC offset')


def cutoff_utc(day):
    # Historical dates include the entire UTC day. Today's check stops at observation time.
    now = datetime.now(timezone.utc)
    return now if day == now.date() else datetime.combine(day + timedelta(days=1), time(), timezone.utc)


def is_extended(name):
    return any(fnmatch.fnmatchcase(Path(name).name, pattern) for pattern in PATTERNS)


def parts(paths):
    """Read named export parts inside directories or ZIPs without extracting private files."""
    files = set()
    for path in map(Path, paths):
        if path.is_dir():
            for p in path.rglob('*'):
                if p.is_file() and (is_extended(p.name) or p.suffix.lower() == '.zip'
                                    or p.match('StreamingHistory*.json')):
                    files.add(p.resolve())
        elif path.is_file():
            files.add(path.resolve())
    saw_basic = saw_extended = False
    for p in sorted(files):
        if p.suffix.lower() == '.zip':
            with ZipFile(p) as archive:
                members = [m for m in archive.infolist() if not m.is_dir()
                           and '__MACOSX' not in Path(m.filename).parts]
                for m in members:
                    if is_extended(m.filename):
                        saw_extended = True
                        with archive.open(m) as stream:
                            yield json.load(stream)
                saw_basic |= any(Path(m.filename).match('StreamingHistory*.json') for m in members)
        elif p.match('StreamingHistory*.json'):
            saw_basic = True
        else:
            saw_extended = True
            with p.open() as stream:
                yield json.load(stream)
    if saw_basic and not saw_extended:
        raise ValueError('basic Account data export lacks Extended Streaming History')


def complete(ms, duration):
    return ms >= 120000 or (duration is not None and ms * 5 >= duration * 4)


@dataclass
class History:
    # A unique exposure is identifiable only to track and UTC end timestamp.
    tracks: dict = field(default_factory=lambda: defaultdict(dict))
    unresolved: dict = field(default_factory=lambda: defaultdict(list))
    counts: Counter = field(default_factory=Counter)

    def summary(self, catalog):
        plays = Counter()
        completions = flagged = joined = 0
        times = []
        for sid, events in self.tracks.items():
            for ts, (ms, skipped, _) in events.items():
                times.append(ts)
                plays[sid] += ms >= 30000
                completions += complete(ms, catalog.get(sid))
                flagged += skipped is True
            if sid in catalog:
                joined += len(events)
        raw = self.counts['raw_rows']
        counts = {name: self.counts[name] for name in
                  ('parts', 'raw_rows', 'valid_timestamp_rows', 'valid_track_id_rows',
                   'valid_ms_rows', 'nontrack_rows', 'invalid_track_events',
                   'duplicate_events', 'after_as_of')}
        for name in ('valid_timestamp', 'valid_track_id', 'valid_ms'):
            counts[name + '_fraction'] = counts[name + '_rows'] / raw if raw else None
        tracks = set(self.tracks)
        return dict(counts, exposures=len(times), distinct_exposed_tracks=len(tracks),
                    qualifying_plays=sum(plays.values()), short_skips=len(times) - sum(plays.values()),
                    completions=completions, spotify_flagged_skips=flagged,
                    repeat_qualifying_plays=sum(max(0, n-1) for n in plays.values()),
                    tracks_with_repeat_plays=sum(n >= 2 for n in plays.values()),
                    catalog_tracks=len(catalog), joined_tracks=len(tracks & catalog.keys()),
                    joined_track_fraction=len(tracks & catalog.keys()) / len(tracks) if tracks else None,
                    joined_events=joined, joined_event_fraction=joined / len(times) if times else None,
                    first_event_utc=min(times).isoformat() if times else None,
                    last_event_utc=max(times).isoformat() if times else None)


def read_history(paths, *, as_of=None):
    history = History()
    counts = history.counts
    cutoff = cutoff_utc(as_of) if as_of is not None else None
    listener = None
    for rows in parts(paths):
        counts['parts'] += 1
        if not isinstance(rows, list):
            raise ValueError('extended history part must be a JSON array')
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not {'ts', 'ms_played', 'spotify_track_uri'} <= row.keys():
                raise ValueError('expected Extended Streaming History event schema')
            owner = row.get('username')
            if owner:
                if listener is not None and owner != listener:
                    raise ValueError('export parts identify more than one listener')
                listener = owner
            counts['raw_rows'] += 1
            try:
                ts = timestamp(row['ts'])
            except ValueError:
                ts = None
            uri = row['spotify_track_uri']
            sid = uri[len('spotify:track:'):] if isinstance(uri, str) and uri.startswith('spotify:track:') else ''
            music = not (row.get('spotify_episode_uri') or row.get('spotify_audiobook_uri'))
            valid_id = music and bool(ID.fullmatch(sid))
            ms = row['ms_played']
            valid_ms = type(ms) is int and ms >= 0
            counts['valid_timestamp_rows'] += ts is not None
            counts['valid_track_id_rows'] += valid_id
            counts['valid_ms_rows'] += valid_ms
            if not music or not uri or (isinstance(uri, str) and uri.startswith('spotify:local:')):
                counts['nontrack_rows'] += 1
                continue
            flag = row.get('skipped')
            if not valid_id or ts is None or not valid_ms or (flag is not None and type(flag) is not bool):
                counts['invalid_track_events'] += 1
                if valid_id and (ts is None or cutoff is None or ts < cutoff):
                    history.unresolved[sid].append(ts)
                continue
            if cutoff is not None and ts >= cutoff:
                counts['after_as_of'] += 1
                continue
            key = sid, ts
            old = history.tracks[sid].get(ts)
            details = tuple(row.get(k) for k in ('platform', 'reason_start', 'reason_end',
                                                 'offline', 'offline_timestamp', 'incognito_mode'))
            event = (ms, flag, details)
            if key in seen or (old is not None and old != event):
                raise ValueError('ambiguous or conflicting events share track and UTC end timestamp')
            seen.add(key)
            counts['duplicate_events'] += old is not None
            history.tracks[sid][ts] = event
    return history


def history_features(history, row, aliases, boundary):
    duration = row.get('duration_ms')
    if duration is not None and (type(duration) is not int or duration <= 0):
        raise ValueError('duration_ms must be a positive integer when supplied')
    if any(ts is None or ts < boundary for sid in aliases for ts in history.unresolved.get(sid, ())):
        raise ValueError('unresolved event could change a judged recording’s prior history')
    listened = [ms for sid in aliases for ts, (ms, _, _) in history.tracks.get(sid, {}).items()
                if ts < boundary]
    plays = sum(ms >= 30000 for ms in listened)
    skips = len(listened) - plays
    completions = sum(complete(ms, duration) for ms in listened)
    return [math.log1p(plays), math.log1p(completions), float(skips >= 3 and completions == 0)], len(listened)


def fit_adjustment(features, v2, labels):
    """Fit a ridge logistic adjustment with the frozen v2 score as a fixed offset."""
    from scipy.optimize import minimize
    from scipy.special import expit

    x, y, offset = np.asarray(features), np.asarray(labels), np.asarray(v2)
    if len(set(labels)) != 2 or not np.any(x) or np.all(x == x[0]):
        raise ValueError('training needs varying history features and both explicit like and nonlike outcomes')

    def loss(w):
        z = offset + x @ w
        value = np.logaddexp(0., z).sum() - y @ z + 0.5 * w @ w
        return value, x.T @ (expit(z) - y) + w

    opt = minimize(loss, np.zeros(x.shape[1]), method='L-BFGS-B', jac=True)
    if not opt.success or not np.isfinite(opt.x).all():
        raise ValueError('history adjustment failed to converge')
    return opt.x


def block_metrics(rows, scores):
    ordered = sorted(zip(rows, scores), key=lambda p: (-p[1], p[0]['recording_id']))
    result = ranking_metrics([row for row, _ in ordered])
    return dict(p_at_10=result['p_at_k'], ndcg_at_10=result['ndcg_at_k'],
                rejects_at_10=result['negative_top_k'])


def paired_summary(blocks, basis):
    """Bound the macro mean over prespecified independent test blocks, conditional on fit."""
    n = len(blocks)
    deltas = {key: mean(b['delta'][key] for b in blocks) for key in METRICS}
    if n < 8 or not isinstance(basis, str) or not basis.strip():
        return deltas, dict(intervals=None, reason='requires >=8 independent test blocks and documented basis')
    # Bonferroni + two-sided Hoeffding. Per-block differences lie in [-limit, limit].
    widths = {k: 2 * limit * math.sqrt(math.log(2*len(METRICS)/.05) / (2*n))
              for k, limit in METRICS.items()}
    bounds = {k: [max(-limit, deltas[k] - widths[k]), min(limit, deltas[k] + widths[k])]
              for k, limit in METRICS.items()}
    return deltas, dict(intervals=bounds, method='simultaneous 95% independent-block Hoeffding bounds',
                        conditional_on='fixed training fit and supplied independent-block design')


def recording_identity(row):
    key = row['recording_id']
    if not isinstance(key, str) or not key.strip():
        raise ValueError('verified recording identity required')
    aliases = [row['spotify_id'], *row.get('spotify_aliases', [])]
    if not all(isinstance(a, str) and ID.fullmatch(a) for a in aliases):
        raise ValueError('valid Spotify ID required for every recording and alias')
    keys = set(identity_tokens(row)) | {'recording:' + key} | {'spotify:' + a for a in aliases}
    matched_isrc = embedded_spotify_isrc(row)
    if matched_isrc:
        if any(k.startswith('isrc:') and k != 'isrc:' + matched_isrc for k in keys):
            raise ValueError('conflicting ISRCs in frozen recording record')
        keys.add('isrc:' + matched_isrc)
    return keys, set(aliases)


def compare(history, panel, model_dir, *, as_of):
    if not history.tracks:
        raise ValueError('usable track history with resolved timestamps, IDs and durations is required')
    frozen = timestamp(panel['frozen_at'])
    snapshot = {}
    for name in ('model_v2.json', 'model_v2.npz'):
        snapshot[name] = (Path(model_dir) / name).read_bytes()
        if hashlib.sha256(snapshot[name]).hexdigest() != panel['model_sha256'][name]:
            raise ValueError('frozen v2 artifact digest mismatch')
    config = json.loads(snapshot['model_v2.json'])
    with np.load(io.BytesIO(snapshot['model_v2.npz']), allow_pickle=False) as artifact:
        model = V2Model(artifact, config)
    inventory = panel['training_recordings']
    if not isinstance(inventory, list) or not inventory:
        raise ValueError('a complete recording-level v2 training inventory is required')
    if 'lr_n_pos' in config and 'lr_n_neg' in config and len(inventory) != config['lr_n_pos'] + config['lr_n_neg']:
        raise ValueError('v2 training inventory row count disagrees with the frozen model')
    tokens = set()
    for row in inventory:
        if isinstance(row, str):
            row = dict(recording_id=row, spotify_id=row)
        tokens.update(recording_identity(row)[0])
    previous_end = frozen
    cutoff = cutoff_utc(as_of)
    train_x, train_v2, train_y, blocks = [], [], [], []
    weights = None
    for block in panel['blocks']:
        start, end = timestamp(block['start']), timestamp(block['end'])
        if start < previous_end or start <= frozen or end <= start or end > cutoff:
            raise ValueError('blocks must close by as-of, follow the freeze and be ordered without overlap')
        previous_end = end
        role, rows = block['role'], block['rows']
        if role not in ('train', 'test') or (role == 'train' and weights is not None):
            raise ValueError('all training blocks must precede all test blocks')
        if not isinstance(block.get('judgment_source'), str) or not block['judgment_source'].strip():
            raise ValueError('each block needs a documented explicit judgment source')
        if not rows or (role == 'test' and len(rows) < 10):
            raise ValueError('a nonempty train block and at least ten judged rows per test block are required')
        x, y, audio, exposed = [], [], [], 0
        for row in rows:
            label = row['label']
            if not start <= timestamp(row['judged_at']) < end or label not in ('like', 'ok', 'reject'):
                raise ValueError('each judgment must be explicit and within its half-open block')
            identities, aliases = recording_identity(row)
            if tokens & identities:
                raise ValueError('evaluation contains overlapping recording identities')
            tokens.update(identities)
            raw = feat_vector(row)[0]
            if not np.isfinite(raw).any() or np.isinf(raw).any():
                raise ValueError('each judged recording needs usable frozen v2 features')
            audio.append(raw)
            features, events = history_features(history, row, aliases, start)
            x.append(features)
            exposed += bool(events)
            y.append(int(label == 'like'))
        scores = model.score_matrix(np.stack(audio))['final']
        if not np.isfinite(scores).all():
            raise ValueError('missing finite score for fixed cohort')
        if role == 'train':
            train_x.extend(x)
            train_v2.extend(scores)
            train_y.extend(y)
            continue
        if weights is None:
            weights = fit_adjustment(train_x, train_v2, train_y)
        fixed = block_metrics(rows, scores)
        augmented = block_metrics(rows, scores + np.asarray(x) @ weights)
        blocks.append(dict(start_utc=start.isoformat(), end_utc=end.isoformat(), judged=len(rows),
                           likes=sum(y), rejects=sum(row['label'] == 'reject' for row in rows),
                           exposed_before_start=exposed, fixed_v2=fixed, history_v2=augmented,
                           delta={k: augmented[k] - fixed[k] for k in METRICS}))
    if not blocks:
        raise ValueError('no later judged test blocks')
    if not any(b['likes'] for b in blocks) or not any(b['rejects'] for b in blocks):
        raise ValueError('the full test panel needs a judged like and a known reject control')
    delta, uncertainty = paired_summary(blocks, panel.get('independence_basis'))
    return dict(blocks=blocks, training_rows=len(train_y), mean_paired_delta=delta, uncertainty=uncertainty)


def run(paths, *, as_of, catalog=None, panel=None, model_dir=ROOT):
    history = read_history(paths, as_of=as_of)
    result = dict(as_of=as_of.isoformat(), scheduled_check=DUE.isoformat(), comparison=None)
    if not history.counts['parts']:
        return dict(result, status='awaiting_nov_26_check' if as_of < DUE else 'export_unavailable')
    reference = {}
    if catalog is not None:
        with Path(catalog).open() as stream:
            rows = json.load(stream)
            for row in rows['tracks'] if isinstance(rows, dict) else rows:
                sid = row.get('spotify_id')
                if isinstance(sid, str) and ID.fullmatch(sid):
                    duration = row.get('duration_ms')
                    if duration is not None and (type(duration) is not int or duration <= 0):
                        raise ValueError('invalid catalog duration_ms')
                    reference[sid] = duration
    result.update(coverage=history.summary(reference))
    if not history.tracks:
        return dict(result, status='unusable_history')
    if panel is None:
        return dict(result, status='pending_judged_panel')
    try:
        with Path(panel).open() as stream:
            result['comparison'] = compare(history, json.load(stream), model_dir, as_of=as_of)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(result, status='comparison_blocked', reason=str(exc))
    return dict(result, status='evaluated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, action='append', help='ZIP, JSON or directory (repeatable)')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'features_v2.json')
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--model-dir', type=Path, default=ROOT)
    parser.add_argument('--as-of', type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    args = parser.parse_args()
    if args.as_of > datetime.now(timezone.utc).date():
        parser.error('the check date is in the future')
    paths = args.history or [ROOT/'data', PIPELINE/'data', Path.home()/'downloads',
                             Path('/workspace/shared/downloads'), Path('/workspace/library-files')]
    try:
        result = run(paths, as_of=args.as_of, catalog=args.catalog, panel=args.panel, model_dir=args.model_dir)
    except (OSError, BadZipFile, ValueError, TypeError, KeyError) as exc:
        result = dict(status='invalid_source', reason=str(exc), comparison=None)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'evaluated' else 2


if __name__ == '__main__':
    raise SystemExit(main())
