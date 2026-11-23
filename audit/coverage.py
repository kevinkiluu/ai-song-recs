"""Measure snapshot coverage and replay separately versioned provider recoveries.

Usage: python -m audit.coverage --help
"""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

from evaluate_v4 import build_manifest, evaluation_pool, has_preview_features, ranking_metrics
from scoring import RB_KEYS

FEATURES = Path(__file__).resolve().parents[1] / 'features_v2.json'
RB_FIELDS = (*RB_KEYS, 'key')
UNIT_FIELDS = ('danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
               'speechiness', 'liveness')
SPOTIFY_ID = re.compile(r'[0-9a-zA-Z]{22}')
SPOTIFY_HREF = re.compile(r'https://open\.spotify\.com/track/([0-9a-zA-Z]{22})(?:\?[^#]*)?')
ISRC = re.compile(r'[A-Z]{2}[A-Z0-9]{3}[0-9]{7}')


def isrc(value):
    value = str(value or '').strip().replace('-', '').upper()
    return value if ISRC.fullmatch(value) else None


def spotify_isrcs(rows):
    """Read independent Spotify track metadata. Fail closed on conflicting IDs."""
    result = {}
    for row in rows:
        sid = row.get('id') or ''
        code = isrc((row.get('external_ids') or {}).get('isrc'))
        if SPOTIFY_ID.fullmatch(sid) and code:
            if result.setdefault(sid, code) != code:
                raise ValueError('conflicting Spotify ISRC observations')
    return result


def panel_usable(panel):
    if not isinstance(panel, dict):
        return False
    return (all(type(panel.get(k)) in (float, int) and math.isfinite(panel[k]) for k in RB_FIELDS)
            and all(0 <= panel[k] <= 1 for k in UNIT_FIELDS)
            and panel['key'] in range(12) and panel['mode'] in (0, 1) and panel['tempo'] >= 0)


def href_id(panel):
    match = SPOTIFY_HREF.fullmatch(str(panel.get('href') or ''))
    return match[1] if match else None


def recover_panels(rows, authority, fetch, *, limit):
    """Exact ID lookup and independently anchored ISRC lookup, at most 20 keys/request."""
    targets = [r['spotify_id'] for r in sorted(rows, key=lambda r: r['position'])
               if not panel_usable(r.get('reccobeats')) and SPOTIFY_ID.fullmatch(r['spotify_id'] or '')]
    targets = list(dict.fromkeys(targets))[:limit]
    keys = list(dict.fromkeys([*targets, *(authority[s] for s in targets if s in authority)]))
    by_sid, by_isrc = defaultdict(list), defaultdict(list)
    counts = Counter(requested_spotify_ids=len(targets), independent_isrc_keys=len(keys)-len(targets))
    failed_keys = set()
    for i in range(0, len(keys), 20):
        batch = keys[i:i + 20]
        try:
            items = fetch(batch)
            if not isinstance(items, list):
                raise ValueError('expected ReccoBeats content list')
            for item in items:
                sid = href_id(item)
                code = isrc(item.get('isrc'))
                if sid in batch or code in batch:
                    if sid:
                        by_sid[sid].append(item)
                        if code:
                            by_isrc[code].append(item)
            counts['returned_panels'] += len(items)
        except (OSError, ValueError):
            failed_keys.update(batch)
            counts['failed_requests'] += 1
    patches = {}
    for sid in targets:
        code = authority.get(sid)
        direct = by_sid[sid]
        if not code:
            counts['missing_independent_isrc'] += 1
            continue
        if any(isrc(item.get('isrc')) != code for item in direct):
            counts['conflicting_isrc'] += 1
            continue
        panels = direct or by_isrc[code]
        if not panels:
            counts['provider_unavailable' if {sid, code} & failed_keys else 'no_panel'] += 1
            continue
        if not all(panel_usable(p) for p in panels):
            counts['incomplete_panel'] += 1
            continue
        if len({tuple(p[k] for k in RB_FIELDS) for p in panels}) != 1:
            counts['ambiguous_panel'] += 1
            continue
        panel = min(panels, key=lambda p: (p['href'], str(p.get('id', ''))))
        patches[sid] = {k: panel[k] for k in ('href', 'isrc', *RB_FIELDS)}
        counts['verified'] += 1
    return patches, dict(counts)


def recover_previews(rows, fetch_track, download, analyze, *, limit):
    """Retry known Deezer IDs, retaining only verified numeric features and a byte hash."""
    ids = [str(r.get('deezer_id')) for r in sorted(rows, key=lambda r: r['position'])
           if not has_preview_features(r) and re.fullmatch(r'[1-9][0-9]*', str(r.get('deezer_id')))]
    result, stages = {}, Counter()
    for did in list(dict.fromkeys(ids))[:limit]:
        try:
            track = fetch_track(did)
            if not isinstance(track, dict) or 'error' in track or 'id' not in track:
                raise ValueError('provider did not return track metadata')
        except (OSError, ValueError, KeyError):
            stages['metadata_unavailable'] += 1
            continue
        if str(track.get('id')) != did:
            stages['identity_mismatch'] += 1
            continue
        if not track.get('preview'):
            stages['matched_no_preview'] += 1
            continue
        try:
            data = download(track['preview'])
            if not data:
                raise ValueError('empty preview')
        except (OSError, ValueError):
            stages['transfer_failure'] += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'preview.mp3'
            path.write_bytes(data)
            try:
                lb = analyze(path)
                if not has_preview_features({'librosa': lb}):
                    raise ValueError('missing preview features')
            except ImportError:
                raise
            except Exception:
                stages['decode_failure'] += 1
                continue
        result[did] = dict(librosa=lb, preview_sha256=hashlib.sha256(data).hexdigest())
        stages['redecoded'] += 1
    return result, dict(stages)


def failure_stages(snapshot, matches, deep):
    reasons = {r['spotify_id']: r.get('reason') for r in snapshot.get('unmatched', [])}
    counts = Counter()
    for r in snapshot['tracks']:
        if has_preview_features(r):
            continue
        sid, did = r['spotify_id'], r.get('deezer_id')
        match = matches.get(sid, {})
        if str(match.get('deezer_id')) != str(did):
            match = {}
        reason = match.get('reason') or reasons.get(sid)
        analysis = deep.get(str(did), {}) if did else {}
        if reason == 'no preview':
            stage = 'matched_no_preview'
        elif not did and reason == 'no match':
            stage = 'no_match'
        elif did and analysis.get('error') == 'analysis failed':
            stage = 'decode_failure'
        elif did and match.get('download_error'):
            stage = 'transfer_failure'
        else:
            stage = 'unknown'
        counts[stage] += 1
    return dict(counts)


def apply_features(rows, panels, previews):
    targets = {r.get('spotify_id') for r in rows if not panel_usable(r.get('reccobeats'))}
    known_missing = {str(r['deezer_id']) for r in rows if r.get('deezer_id')
                     and not has_preview_features(r)}
    if set(panels) - targets or set(previews) - known_missing:
        raise ValueError('patch must target a missing feature in the frozen snapshot')
    updated = []
    for row in rows:
        r = dict(row)
        sid, did = r.get('spotify_id'), str(r.get('deezer_id'))
        if sid in panels and not panel_usable(r.get('reccobeats')):
            code = isrc(panels[sid].get('isrc'))
            if not code or (r.get('isrc') and isrc(r['isrc']) != code):
                raise ValueError('recovered ISRC conflicts with snapshot evidence')
            r.update(reccobeats=panels[sid], isrc=code)
        if did in previews and not has_preview_features(r):
            r.update(previews[did], audio_source='deezer_preview_30s')
        updated.append(r)
    return updated


def summary(rows):
    rb = [panel_usable(r.get('reccobeats')) for r in rows]
    lb = [has_preview_features(r) for r in rows]
    return dict(rows=len(rows), reccobeats=sum(rb), preview=sum(lb),
                neither=sum(not (a or b) for a, b in zip(rb, lb)))


def cohorts(old, new, *, cutoff=130):
    """Keep row denominators fixed; count accepted ISRCs separately from row gains."""
    result = {}
    for name in ('all', 'newest', 'older'):
        pairs = [(a, b) for a, b in zip(old, new, strict=True) if name == 'all'
                 or (a['position'] < cutoff) == (name == 'newest')]
        before, after = summary([a for a, _ in pairs]), summary([b for _, b in pairs])
        gained_codes = {isrc(b['reccobeats'].get('isrc')) for a, b in pairs if
                        not panel_usable(a.get('reccobeats')) and panel_usable(b.get('reccobeats'))}
        result[name] = dict(baseline=before, after=after,
                            new_reccobeats_recordings=len(gained_codes - {None}),
                            original_neither_gained_any_source=before['neither']-after['neither'])
    return result


def paired_ranking(rows, updated, rejects, catalog, *, cutoff=130):
    """Fit v2 once on frozen training features; rank identical held-out recordings twice."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    from model_v2 import fit_records
    from scoring import V2Model, feat_vector

    # New verified identities constrain both arms. A new eligible alias never
    # displaces the archived representative in the fixed shared comparison.
    base = [dict(a, **{k: b[k] for k in ('isrc', 'preview_sha256') if b.get(k)})
            for a, b in zip(rows, updated, strict=True)]
    before = build_manifest(base, rejects or [], catalog or [], cutoff=cutoff)
    after = build_manifest(updated, rejects or [], catalog or [], cutoff=cutoff)
    old_pool, new_pool = evaluation_pool(before), evaluation_pool(after)
    after_ids = {r['recording_id'] for r in new_pool}
    shared = [r for r in old_pool if r['recording_id'] in after_ids]
    old_ids = {r['recording_id'] for r in old_pool}
    scope = 'supplied pools'
    if catalog is None and rejects is None:
        scope = 'likes only'
    elif catalog is None or rejects is None:
        scope = 'partial supplied pools'
    info = dict(shared_pool=len(shared), newly_evaluable=len(after_ids - old_ids),
                newly_evaluable_label_counts=dict(Counter(
                    r['label'] for r in new_pool if r['recording_id'] not in old_ids)),
                scope=scope)
    if rejects is None or catalog is None:
        return dict(info, status='missing_inputs', missing=[k for k,v in
                    [('rejects', rejects), ('catalog', catalog)] if v is None], delta=None)
    controls = Counter(r['label'] for r in shared)
    if len(shared) < 10 or any(controls[label] == 0 for label in ('like', 'reject', 'unlabeled')):
        return dict(info, status='insufficient_evaluation_pool', shared_labels=dict(controls),
                    delta=None)
    training = defaultdict(list)
    for r in before:
        if r['partition'] == 'train' and np.isfinite(feat_vector(r['record'])[0]).any():
            training[r['label']].append(r['record'])
    if len(training['like']) < 3 or not training['reject']:
        return dict(info, status='insufficient_pool_or_training', delta=None)
    with threadpool_limits(limits=1), redirect_stdout(io.StringIO()):
        cfg, artifacts = fit_records(training['like'], training['reject'])
        model = V2Model(artifacts, cfg)
        changed = {r['row_id']: r['record'] for r in after}
        original = [r['record'] for r in shared]
        revised = [changed[r['row_id']] for r in shared]

        def measure(records):
            scores = model.score_records(records)['final']
            if not np.isfinite(scores).all():
                raise ValueError('nonfinite v2 score')
            ranked = sorted(zip(scores, shared), key=lambda pair: (-pair[0], pair[1]['recording_id']))
            result = ranking_metrics([r for _, r in ranked], k=10)
            return dict(p_at_10=result['p_at_k'], ndcg_at_10=result['ndcg_at_k'])

        baseline, final = measure(original), measure(revised)
        def delta(value):
            return {key: value[key] - baseline[key] for key in baseline}

        source_deltas = {}
        for name, field in [('preview', 'librosa'), ('reccobeats', 'reccobeats')]:
            variant = [dict(a, **{field: b.get(field)}) for a, b in zip(original, revised)]
            source_deltas[name] = delta(measure(variant))
    return dict(info, status='scored', baseline=baseline, after=final, delta=delta(final),
                source_deltas=source_deltas)


def save_cache(path, obj):
    """Publish a complete owner-only cache once. Concurrent writers cannot clobber it."""
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', delete=False) as tmp:
        staged = Path(tmp.name)
        try:
            json.dump(obj, tmp, allow_nan=False)
            tmp.flush()
            os.link(staged, path)
        finally:
            staged.unlink(missing_ok=True)


def load_cache(path, digest, expected_sha256):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('cache SHA-256 differs from the original recovery run')
    cache = json.loads(data)
    if cache.get('version') != 1 or cache.get('features_sha256') != digest:
        raise ValueError('cache version or frozen snapshot digest mismatch')
    authority = cache.get('verified_isrcs', {})
    if set(authority) != set(cache['panels']) or (authority and not re.fullmatch(
            r'[0-9a-f]{64}', str(cache.get('spotify_tracks_sha256', '')))):
        raise ValueError('missing independent Spotify authority for cached panel')
    for sid, panel in cache['panels'].items():
        if not (SPOTIFY_ID.fullmatch(sid) and panel_usable(panel) and
                href_id(panel) and isrc(panel.get('isrc')) and
                authority[sid] == isrc(panel['isrc'])):
            raise ValueError('invalid cached ReccoBeats feature or identity')
    for did, preview in cache['previews'].items():
        if not (re.fullmatch('[1-9][0-9]*', did) and has_preview_features(preview)
                and re.fullmatch('[0-9a-f]{64}', str(preview.get('preview_sha256', '')))):
            raise ValueError('invalid cached preview feature or identity')
    return cache


def provider_bytes(url, *, maximum, accept):
    req = urllib.request.Request(url, headers={'User-Agent': 'recording-coverage-audit/1.0',
                                                'Accept': accept})
    with urllib.request.urlopen(req, timeout=12) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('provider response exceeded size limit')
    return data


def provider_json(url):
    return json.loads(provider_bytes(url, maximum=1_000_000, accept='application/json'))


def fetch_panels(keys):
    query = urllib.parse.urlencode({'ids': ','.join(keys)}, safe=',')
    reply = provider_json('https://api.reccobeats.com/v1/audio-features?' + query)
    if not isinstance(reply, dict) or not isinstance(reply.get('content'), list):
        raise ValueError('missing provider content')
    return reply['content']


def preview_bytes(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ValueError('invalid preview URL')
    return provider_bytes(url, maximum=8_000_000, accept='audio/*')


def analyze_preview(path):
    from features_v2 import analyze_deep
    return analyze_deep(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=FEATURES)
    parser.add_argument('--spotify-tracks', type=Path,
                        help='Independent Spotify track objects with id and external_ids.isrc')
    parser.add_argument('--matches', type=Path, help='Original Deezer match/error map by Spotify ID')
    parser.add_argument('--deep', type=Path, help='Original librosa result/error map by Deezer ID')
    parser.add_argument('--rejects', type=Path, help='Frozen historical reject feature rows')
    parser.add_argument('--catalog', type=Path, help='Frozen catalog feature rows')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--recover', type=Path, metavar='NEW_PRIVATE_CACHE',
                      help='Read providers and create an exclusive, dated derived cache')
    mode.add_argument('--cache', type=Path, help='Replay a previously created cache offline')
    parser.add_argument('--cache-sha256', help='The original recovery run’s cache digest, for --cache')
    parser.add_argument('--rb-limit', type=int, default=96,
                        help='Maximum missing, valid Spotify IDs, newest position first')
    parser.add_argument('--preview-limit', type=int, default=2,
                        help='Maximum distinct known Deezer IDs with missing previews')
    args = parser.parse_args()
    if bool(args.cache) != bool(args.cache_sha256):
        parser.error('offline cache replay requires both --cache and --cache-sha256')
    if min(args.rb_limit, args.preview_limit) < 0:
        parser.error('limits must be nonnegative')

    hashes = {}
    def read_input(name, path, default):
        if path is None:
            return default
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    snapshot = read_input('features', args.features, None)
    rows = snapshot['tracks']
    matches = read_input('matches', args.matches, {})
    deep = read_input('deep', args.deep, {})
    rejects = read_input('rejects', args.rejects, None)
    catalog = read_input('catalog', args.catalog, None)
    authority = spotify_isrcs(read_input('spotify_tracks', args.spotify_tracks, []))
    empty = dict(panels={}, previews={}, observed_utc=None, recovery_counts=None)
    cache_sha = None
    if args.recover is not None:
        if args.recover.exists():
            parser.error('derived-cache destination already exists')
        panels, rb_counts = recover_panels(rows, authority, fetch_panels, limit=args.rb_limit)
        previews, preview_counts = recover_previews(
            rows, lambda did: provider_json('https://api.deezer.com/track/' + did),
            preview_bytes, analyze_preview, limit=args.preview_limit)
        cache = dict(version=1, features_sha256=hashes['features'],
                     spotify_tracks_sha256=hashes.get('spotify_tracks'),
                     observed_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     recovery_counts=dict(reccobeats=rb_counts, previews=preview_counts,
                                          rb_limit=args.rb_limit, preview_limit=args.preview_limit),
                     verified_isrcs={sid: authority[sid] for sid in panels},
                     panels=panels, previews=previews)
    elif args.cache is not None:
        cache_sha = args.cache_sha256
        cache = load_cache(args.cache, hashes['features'], cache_sha)
        if args.spotify_tracks and (cache.get('spotify_tracks_sha256') != hashes['spotify_tracks']
                or any(authority.get(sid) != code for sid, code in cache.get('verified_isrcs', {}).items())):
            raise ValueError('Spotify authority differs from original recovery run')
    else:
        cache = empty

    from integrate_v4 import _enrich_records
    frozen = _enrich_records(rows, matches, None)
    after = apply_features(frozen, cache['panels'], cache['previews'])
    if rejects is not None:
        rejects = _enrich_records(rejects, matches, None)
    if catalog is not None:
        catalog = _enrich_records(catalog, matches, None)
    if args.recover is not None:
        save_cache(args.recover, cache)
        cache_sha = hashlib.sha256(args.recover.read_bytes()).hexdigest()
    report = dict(recovery_cache_sha256=cache_sha, snapshot_generated=snapshot.get('generated'),
                  input_sha256=hashes,
                  recovery_observed_utc=cache.get('observed_utc'),
                  recovery_authority_sha256=cache.get('spotify_tracks_sha256'),
                  recovery_counts=cache.get('recovery_counts'),
                  historical_missing_preview_stages=failure_stages(snapshot, matches, deep),
                  cohorts=cohorts(frozen, after),
                  paired_ranking=paired_ranking(frozen, after, rejects, catalog))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
