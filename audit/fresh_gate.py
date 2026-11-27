"""Count untouched three-tier judgments against the original frozen v4 catalog."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_v4 import build_manifest, evaluation_pool, has_preview_features
from integrate_v4 import _enrich_records, _read_records
from scoring import clean


TIERS = {'like': 'kept', 'ok': 'left', 'reject': 'removed', 'unknown': 'unknown'}
LIMITS = {'like': 10, 'reject': 2}
SPOTIFY = re.compile(r'[a-zA-Z0-9]{22}')
ISRC = re.compile(r'[A-Z]{2}[A-Z0-9]{3}[0-9]{7}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(date.tzinfo is not None, 'snapshot times must include a UTC offset')
    return date


def song_key(row):
    # Keep version information in square brackets and parentheticals.
    title = (row.get('title') or '').replace('[', '(').replace(']', ')')
    return clean(title), clean((row.get('artist') or '').split(',')[0])


def isrc(value):
    value = (value or '').upper().replace('-', '')
    return value if ISRC.fullmatch(value) else None


def spotify_id(value):
    value = (value or '').removeprefix('spotify:track:')
    return value if SPOTIFY.fullmatch(value) else None


def archived_tracks(data, source):
    """Read original three-column exports, ranked track rows, or Spotify API pages."""
    complete = source['complete']
    require(type(complete) is bool, 'snapshot completeness must be explicit')
    if Path(source['path']).suffix.lower() != '.json':
        rows = []
        for index, line in enumerate(data.decode('utf-8-sig').splitlines()):
            fields = line.split('\t')
            if len(fields) == 1 and (not line.strip() or line.startswith('#')):
                continue
            require(len(fields) >= 3, f'unrecognized track export row {index + 1}')
            rows.append(dict(title=fields[0], artist=fields[1], spotify_id=fields[2],
                             position=index))
        return rows, complete
    doc = json.loads(data)
    if isinstance(doc, dict):
        if 'tracks' in doc:
            if doc.get('id'):
                require(doc['id'] == source['collection'], 'Spotify playlist ID contradicts inventory')
            doc = doc['tracks']
        if isinstance(doc, dict):
            rows = doc['items']
            complete = complete and (doc.get('next') is None and doc.get('offset', 0) == 0
                                     and doc.get('total') == len(rows))
            doc = rows
    require(isinstance(doc, list), 'expected a track export or Spotify track page')
    rows = []
    for index, row in enumerate(doc):
        require(isinstance(row, dict), 'invalid archived track')
        if 'track' in row:
            item = row['track'] or {}
            artists = item.get('artists') or []
            row = dict(spotify_id=item.get('id'), title=item.get('name'),
                       artist=', '.join(a['name'] for a in artists))
        rows.append(dict(row, position=index))
    return rows, complete


class IdentityIndex:
    """Resolve source IDs to v4 components only when the source metadata agree."""

    def __init__(self, likes, rejects, catalog, crosswalk):
        self.identities = {}
        self.records = defaultdict(list)
        self.known = defaultdict(set)
        self.catalog = defaultdict(list)
        for row in build_manifest(likes, rejects, catalog, fresh=[]):
            group = row['recording_id']
            self.records[group].append(row['record'])
            for token in row['identities']:
                self.identities[token] = group
            if row['source'] == 'frozen_catalog':
                self.catalog[group].append(row['record'])
            else:
                self.known[group].add(row['label'])

        self.crosswalk = defaultdict(set)
        self.invalid = set()
        require(isinstance(crosswalk, dict), 'expected the ISRC-keyed ReccoBeats pool map')
        for key, row in crosswalk.items():
            if row is None:  # Cached lookup with no result.
                continue
            require(isinstance(row, dict), 'invalid ReccoBeats record')
            url = urlsplit(row.get('href') or '')
            match = re.fullmatch(r'/track/([a-zA-Z0-9]{22})/?', url.path)
            if url.netloc != 'open.spotify.com' or url.scheme != 'https' or not match:
                continue
            sid = match[1]
            value = isrc(key)
            declared = isrc(row.get('isrc'))
            self.crosswalk[sid].update(code for code in (value, declared) if code)
            if not value or value != declared:
                self.invalid.add(sid)

    def targets(self, sid):
        tokens = ['spotify:' + sid] + ['isrc:' + s for s in self.crosswalk.get(sid, ())]
        return {self.identities[t] for t in tokens if t in self.identities}

    def resolve(self, row):
        """Return group and join problem. Never synthesize an identity from a title."""
        sid = spotify_id(row.get('spotify_id'))
        if not sid:
            return None, 'unavailable Spotify identity'
        token = 'spotify:' + sid
        codes = self.crosswalk.get(sid, set())
        targets = self.targets(sid)
        if sid in self.invalid or len(codes) > 1 or len(targets) > 1:
            return token, 'ambiguous provider identity'
        if not targets:
            return token, None
        group, = targets
        pool = self.catalog.get(group, [])
        keys = {song_key(rec) for rec in pool or self.records[group]}
        catalog_codes = {isrc(rec.get('isrc')) for rec in pool} - {None}
        if len(catalog_codes | codes) > 1:
            return token, 'conflicting catalog ISRCs'
        if not all(song_key(row)) or keys != {song_key(row)}:
            return token, 'title or first artist disagrees'
        return group, None


class Archive:
    def __init__(self, path, digest):
        self.base = path.parent
        data = path.read_bytes()
        require(sha256(data).hexdigest() == digest, 'inventory SHA-256 mismatch')
        self.doc = json.loads(data)
        self.digest = digest
        self.files = {}
        self.paths = {}
        self.snapshot_identity = {}

    def read(self, name, role):
        source = self.doc['sources'][name]
        require(source['role'] == role, f'snapshot role mismatch for {name}')
        if name not in self.files:
            path = (self.base / source['path']).resolve()
            identity = role, source.get('collection'), source.get('at')
            require(self.snapshot_identity.setdefault(path, identity) == identity,
                    f'one file has incompatible snapshot roles or times: {name}')
            data = path.read_bytes()
            require(sha256(data).hexdigest() == source['sha256'], f'SHA-256 mismatch for {name}')
            self.files[name] = data
            self.paths[name] = path
        return self.files[name], source

    def tracks(self, name, role, *, collection=None, round_name=None):
        data, source = self.read(name, role)
        require(collection is None or source['collection'] == collection,
                f'wrong collection for {name}')
        require(round_name is None or source.get('round') == round_name,
                f'wrong round for {name}')
        rows, complete = archived_tracks(data, source)
        return rows, complete, timestamp(source['at'])


def membership(index, rows, complete):
    members, uncertain = set(), set()
    for row in rows:
        group, issue = index.resolve(row)
        if group is None:
            complete = False
            continue
        members.add(group)
        if issue:
            uncertain.add(group)
            # The Spotify ID is observed, but its link to these particular
            # catalog components is disputed. Their absence cannot be inferred.
            uncertain.update(index.targets(spotify_id(row['spotify_id'])))
    return members, uncertain, complete


def round_evidence(archive, spec, index, original_at):
    name = spec['name']
    library = archive.doc['library_id']
    playlist = spec['playlist_id']
    rated = spec.get('rated_playlist_id', playlist)
    require(SPOTIFY.fullmatch(playlist) and SPOTIFY.fullmatch(rated), f'invalid playlist ID in {name}')
    offers, complete, offered_at = archive.tracks(spec['offered'], 'offered', collection=playlist,
                                                  round_name=name)
    retained, full_after, retained_at = archive.tracks(
        spec['retained'], 'retained', collection=rated, round_name=name)
    likes, full_likes, liked_at = archive.tracks(spec['cutoff_likes'], 'cutoff_likes',
                                                collection=library)
    prior, full_prior, history_at = archive.tracks(spec['prior_history'], 'prior_history',
                                                  round_name=name)
    require(archive.paths[spec['offered']] != archive.paths[spec['retained']],
            'offer and post-rating membership require distinct source snapshots')
    require(history_at <= offered_at < retained_at <= liked_at == timestamp(spec['cutoff']),
            f'inconsistent history, offer, rating or library cutoff in {name}')
    cohort = spec['cohort']
    require(cohort in ('historical_replay', 'post_freeze') and type(spec['untouched']) is bool,
            f'missing evidence cohort or untouched declaration in {name}')
    later = cohort == 'post_freeze'
    before_likes, full_before_likes = [], not later
    if spec.get('prior_likes'):
        ref = spec['prior_likes']
        role = 'original_likes' if ref == 'original_likes' else 'prior_likes'
        before_likes, full_before_likes, before_at = archive.tracks(
            ref, role, collection=library, round_name=name if role == 'prior_likes' else None)
        require(before_at <= offered_at, f'pre-offer library postdates offers for {name}')
    prior_saved, disputed_saved, full_before_likes = membership(index, before_likes, full_before_likes)
    valid_date = not later or (original_at <= offered_at and
                               timestamp(archive.doc['frozen_at']) < offered_at)
    after, disputed_after, full_after = membership(index, retained, full_after)
    saved, disputed_likes, full_likes = membership(index, likes, full_likes)
    history, disputed_history, full_prior = membership(index, prior, full_prior)
    # Prior recommendations used the v2 exclusion key, including legacy IDs.
    # A matching text key can exclude a past exposure but never creates a join.
    history_keys = {(clean(row.get('title')), clean((row.get('artist') or '').split(',')[0]))
                    for row in prior}
    offered = {}
    uncertain = []
    for i, row in enumerate(offers):
        group, issue = index.resolve(row)
        if group is None:
            uncertain.append(dict(row=i, reason=issue))
            continue
        if group in offered and song_key(offered[group]['record']) != song_key(row):
            issue = 'inconsistent metadata on repeated offer'
        # A valid repeated occurrence cannot erase an observed identity conflict.
        issue = issue or offered.get(group, {}).get('issue')
        tier = 'unknown'
        if group in saved and group not in disputed_likes:
            tier = 'like'
        elif full_likes and group not in disputed_likes:
            if group in after and group not in disputed_after:
                tier = 'ok'
            elif full_after and group not in disputed_after:
                tier = 'reject'
        key = clean(row.get('title')), clean((row.get('artist') or '').split(',')[0])
        prior_exposure = group in history or (all(key) and key in history_keys)
        offered[group] = dict(tier=tier, issue=issue, record=row,
                              prior=prior_exposure or group in prior_saved,
                              complete_history=full_prior and group not in disputed_history,
                              prior_likes_known=full_before_likes and group not in disputed_saved,
                              untouched=spec['untouched'], valid_date=valid_date,
                              cohort=cohort, round=name, at=offered_at)
    # A playlist can gain new tracks after an offer. They have no judgment
    # denominator here; record how many post-rating tracks lack an original offer.
    counts = Counter(TIERS[o['tier']] for o in offered.values())
    counts['unknown'] += len(uncertain)
    summary = dict(name=name, offered=len(offered) + len(uncertain), source_rows=len(offers),
                   offer_complete=complete, retained_outside_offer=len(after - offered.keys()),
                   unresolved_offer_rows=uncertain,
                   **{tier: counts[tier] for tier in TIERS.values()})
    return offered, summary


def reconstruct(path, digest):
    """Return (private audit, integrate_v4 --fresh rows) from a pinned source inventory."""
    archive = Archive(Path(path), digest)
    doc = archive.doc
    require(re.fullmatch(r'[0-9a-f]{40,64}', doc['challenger_revision']) and doc['library_id'],
            'missing frozen challenger identity or library identifier')
    timestamp(doc['frozen_at'])
    mode = doc.get('mode', 'post_freeze')
    require(mode in ('historical_replay', 'post_freeze'), 'unknown fresh gate mode')
    dataset = doc.get('dataset', 'original_v5')
    require(dataset in ('original_v5', 'fixture'), 'unknown dataset')
    if dataset == 'original_v5':
        # This entire digest can be recomputed from the archived git blob.
        # The private catalog and reject hashes come from the locked inventory.
        require(doc['sources']['likes']['sha256'] ==
                'd8db0af163807d9c40fa8fcc2903ac70db059cd1492efa639e92e7d7a4488ed1',
                'liked features differ from the tracked original snapshot')
    matches = json.loads(archive.read('matches', 'matches')[0]) if 'matches' in doc['sources'] else {}
    frozen = {}
    for role in ('likes', 'rejects', 'catalog'):
        data, _ = archive.read(role, role)
        records = _read_records(data, role, 'tracks' if role == 'likes' else None)
        frozen[role] = _enrich_records(records, matches, None)
    if dataset == 'original_v5':
        require([len(frozen[k]) for k in ('catalog', 'rejects', 'likes')] == [590, 13, 650],
                'frozen feature record counts differ from the original v5 cohort')
    previews = {}
    for name in doc['sources']:
        if name.startswith('preview/'):
            did = name.removeprefix('preview/')
            require(re.fullmatch(r'[1-9][0-9]*', did), 'preview source requires a Deezer ID')
            data, source = archive.read(name, 'preview')
            require(archive.paths[name].name == did + '.mp3' and data, 'wrong or empty preview')
            previews[did] = source['sha256']
            del archive.files[name], data
    for rows in frozen.values():
        for r in rows:
            digest_ = previews.get(str(r.get('deezer_id')))
            if digest_:
                require(r.get('preview_sha256') in (None, digest_), 'frozen preview digest conflict')
                r['preview_sha256'] = digest_
    original, full, original_at = archive.tracks('original_likes', 'original_likes',
                                                 collection=doc['library_id'])
    expected = {r['position']: str(r.get('spotify_id')) for r in frozen['likes']}
    actual = {r['position']: str(r.get('spotify_id')).removeprefix('spotify:track:')
              for r in original}
    require(full and len(expected) == len(frozen['likes']) and expected == actual,
            'original liked positions/Spotify IDs differ from frozen features')
    index = IdentityIndex(**frozen, crosswalk=json.loads(archive.read('crosswalk', 'crosswalk')[0]))
    observations = defaultdict(list)
    rounds = []
    names = set()
    require(isinstance(doc['rounds'], list) and doc['rounds'], 'no round inventory')
    for spec in doc['rounds']:
        require(spec['name'] and spec['name'] not in names, 'duplicate or unnamed round')
        names.add(spec['name'])
        offered, summary = round_evidence(archive, spec, index, original_at)
        rounds.append(summary)
        for group, observation in offered.items():
            observations[group].append(observation)

    disputed = set()
    for events in observations.values():
        for event in events:
            if event['issue']:
                disputed.update(index.targets(spotify_id(event['record']['spotify_id'])))
    fresh, audit, expected_joins = [], [], set()
    cohorts = defaultdict(Counter)
    counts = Counter()
    for group, events in sorted(observations.items()):
        events.sort(key=lambda e: (e['at'], e['round']))
        tiers = {e['tier'] for e in events} - {'unknown'}
        first = events[0]
        candidate = index.catalog.get(group, [])
        if candidate:
            counts['joined'] += 1
        known = group in index.known or first['prior']
        if known:
            counts['already_known'] += 1
        if len(tiers | index.known.get(group, set())) > 1:
            reason = 'conflicting'
        elif group in disputed or any(e['issue'] for e in events):
            reason = 'ambiguous_identity'
        elif known:
            reason = 'already_known'
        elif not first['complete_history']:
            reason = 'incomplete_prior_history'
        elif not first['prior_likes_known']:
            reason = 'incomplete_prior_likes'
        elif first['tier'] == 'unknown':
            reason = 'unknown'
        elif first['tier'] == 'ok':
            reason = 'ok'
        elif not all(e['untouched'] for e in events) or not first['valid_date']:
            reason = 'not_fresh'
        elif not candidate:
            reason = 'outside_catalog'
        elif not any(has_preview_features(row) for row in candidate):
            reason = 'missing_ranking_features'
        else:
            cohorts[first['cohort']][first['tier']] += 1
            reason = 'eligible' if first['cohort'] == mode else 'outside_selected_cohort'
        if reason != 'already_known':
            counts[reason] += 1
        if reason == 'eligible':
            # Copy only verified identity from a frozen member, never score a
            # judgment using features taken from the listener's rated playlist.
            matched = next(row for row in candidate if has_preview_features(row))
            row = {k: matched[k] for k in ('isrc', 'deezer_id', 'preview_sha256') if matched.get(k)}
            row.update(spotify_id=spotify_id(first['record']['spotify_id']), label=first['tier'],
                       label_source=f"fresh-gate/{digest}/{first['round']}")
            fresh.append(row)
            expected_joins.add((group, row['label'], row['label_source']))
        audit.append(dict(recording_id=group, reason=reason, joined=bool(candidate),
                          already_known=known, events=[{k: e[k] for k in
                          ('round', 'tier', 'issue', 'cohort', 'prior')} for e in events]))

    # Assert that the exact artifact supplied to v4 replays the intended count.
    eligible = Counter(row['label'] for row in fresh)
    used = [r for r in evaluation_pool(build_manifest(**frozen, fresh=fresh))
            if r['label'] != 'unlabeled']
    actual_joins = {(r['recording_id'], r['label'], r['label_source']) for r in used}
    require(actual_joins == expected_joins, 'eligible judgments disagree with frozen v4 partitions')
    eligible = {label: eligible[label] for label in LIMITS}
    shortfall = {label: max(0, n - eligible[label]) for label, n in LIMITS.items()}
    funnel = {k: counts[k] for k in ('joined', 'conflicting', 'ambiguous_identity', 'already_known',
                                     'incomplete_prior_history', 'incomplete_prior_likes',
                                     'unknown', 'ok', 'not_fresh',
                                     'outside_catalog', 'missing_ranking_features',
                                     'outside_selected_cohort', 'eligible')}
    funnel['distinct_identified_offers'] = len(observations)
    return dict(dataset=dataset, mode=mode, verified_preview_files=len(previews),
                inventory_sha256=digest, challenger_revision=doc['challenger_revision'],
                source_sha256={name: doc['sources'][name]['sha256'] for name in archive.paths},
                catalog_rows=len(frozen['catalog']), rounds=rounds, recordings=audit,
                funnel=funnel, eligible=eligible, shortfall=shortfall,
                eligible_by_cohort={name: {k: cohorts[name][k] for k in LIMITS}
                                    for name in ('historical_replay', 'post_freeze')},
                status=('INSUFFICIENT_EVIDENCE' if any(shortfall.values())
                        else 'READY_FOR_OFFLINE_EVALUATION')), fresh


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path, help='private pinned source and round inventory JSON')
    parser.add_argument('--sha256', required=True, help='independently retained inventory SHA-256')
    parser.add_argument('--output-dir', required=True, type=Path, help='new private directory')
    args = parser.parse_args(argv)
    try:
        report, fresh = reconstruct(args.inventory, args.sha256)
        out = args.output_dir.resolve()
        out.mkdir(mode=0o700)
        created = []
        try:
            for filename, data in (('gate.json', report), ('fresh.json', fresh)):
                path = out / filename
                with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as f:
                    created.append(path)
                    json.dump(data, f, indent=2, allow_nan=False)
                    f.write('\n')
        except BaseException:
            for path in created:
                path.unlink()
            out.rmdir()
            raise
    except (OSError, KeyError, TypeError, ValueError) as e:
        parser.exit(2, f'Incomplete fresh evidence: {e}\n')
    print(json.dumps({k: report[k] for k in ('dataset', 'mode', 'status', 'eligible', 'shortfall')}))


if __name__ == '__main__':
    main()
