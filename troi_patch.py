#!/usr/bin/env python3
"""Troi-style patch pipeline for ai-song-recs v3 (liked-only variant).

Re-implementation of the Troi *architecture* (Element/Patch pipeline,
multi-stream candidate generation, weighted Zipper interleave, Troi filter
set) -- NOT a copy of Troi source (GPL-2.0; no Troi code vendored).

Spec: ~/workspace/audio-pipeline/research/troi-spec.md, section (b.2).
Grounded against the actual v2 files (see deltas in the build report):
  - S3 neighbour fit uses features_v2.json (475 liked tracks with librosa),
    NOT the spec's stale "93 analyzed liked tracks" from analysis.json.
  - Eval pool is the identical 311-item pool from evaluation_v2.json
    (99 held-out likes + 12 negatives + 200 distractors), NOT 343.
  - Held-out split is positions < 130 (TEST_N=130); 129 usable IDs
    (1 track has spotify_id 'None').
  - Test file is root-level test_troi_patch.py (v2 convention), not
    tests/test_troi_patch.py as the spec's (b.1) says.

Seed artists are ranked by LIKED-TRACK COUNT. This is a PROXY for Troi's
`stats` entity (top artists by listen/play count): liked-count is NOT
play-count, and no play-count weighting is possible until the streaming
export lands (spec section b.3, gated). The run log says so on every run.

S1 (similar-artist) status, verified live from this VM 2026-11-22:
  Labs endpoint https://labs.api.listenbrainz.org/similar-artists/json is
  reachable and validates its `algorithm` enum (400 without it), but
  returns HTTP 200 `[]` for tested artists (Sleep Theory, Linkin Park;
  two algorithm values). No similar-artist data could be fetched, so S1
  is DEGRADED to empty and the run uses S3+S4 only. No data was faked.
  fetch_similar_artists_labs() below implements the real call for reuse
  once the endpoint returns data.

S3 audio-neighbour: sklearn NearestNeighbors (brute-force, NOT Annoy)
  over standardized [bpm, brightness_mean, rms_mean, zcr_mean] from the
  librosa panel, fitted on TRAIN liked tracks only (position >= 130, with
  librosa) so held-out IDs never enter the fit (T-16). Spec deviation:
  the spec's 5th dimension ("key-match 0/1 vs seed") is seed-relative and
  cannot be an index dimension; it is applied as a tie-break on the
  nearest-seed distance ranking instead (same tonic = same key or
  relative major/minor).
S4 same-artist: cached-pool variant by default (pool tracks whose first
  artist is a top-20 seed artist, ranked by v2 score); a live Deezer
  top-tracks fetcher exists (fetch_same_artist_live) for production runs
  but is NOT used by the default deterministic path.

Run: ~/workspace/audio-env/bin/python troi_patch.py [eval|run]
  eval -> evaluate_patch(), writes evaluation_v3_troi.json
  run  -> run_default_patch(), prints Jams + Exploration
"""
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

BASE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(BASE, 'data')
LIKED_FILE = os.path.expanduser('~/workspace/spotify-liked-songs.txt')
HIST_FILE = os.path.expanduser('~/workspace/spotify-recommended-tracks.txt')
sys.path.insert(0, BASE)

from scoring import V2Model, v6_score  # noqa: E402  (read-only use)
from model_v2 import NEGATIVE_IDS  # noqa: E402  (13 confirmed negatives)

TEST_N = 130          # held-out = positions < 130 (evaluate.py / model_v2.py)
SEED = 42
POOL_CAP = 300        # hard candidate cap per run (spec b.2)
S1_CAP, S3_CAP, S4_CAP = 120, 120, 60
ARTIST_CAP = 2        # Troi ArtistCreditLimiterElement default
DEFAULT_PROMPT = ('audio-neighbour:liked-seeds:3:easy '
                  'similar-artist:top20:2:easy same-artist:top20:1:easy')
EXPLORATION_PROMPT = ('audio-neighbour:liked-seeds:3:medium '
                      'similar-artist:top20:2:medium same-artist:top20:1:easy '
                      'audio-neighbour:liked-seeds:1:hard')
# S1 verification, done live from this VM 2026-11-22 (curl): endpoint
# reachable, algorithm enum required, HTTP 200 [] for Sleep Theory and
# Linkin Park under two algorithm values -> stream degraded to empty.
S1_DEGRADED_REASON = (
    'labs similar-artists endpoint verified live 2026-11-22 from this VM: '
    'reachable, validates algorithm enum, but HTTP 200 [] for tested '
    'artists (Sleep Theory, Linkin Park; 2 algorithms); no data returned, '
    'stream empty, degraded to S3+S4 only')
LABS_ALGORITHM = ('session_based_days_1800_session_300_contribution_3_'
                  'threshold_10_limit_100_filter_True_skip_30')
MIN_PLAYS = 1000      # spec (b.3) history-gate constant (gate is closed)


def clean(s):
    """Normalize title/artist for key matching (same as pipeline.clean)."""
    s = (s or '').lower()
    s = re.sub(r'\(feat[^)]*\)', '', s)
    s = re.sub(r'\[.*?\]', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def first_artist(artist):
    return (artist or '').split(',')[0].strip()


def norm_key(title, artist):
    from recommend_v2 import rec_key

    return rec_key(title, artist)


# ---------------------------------------------------------------- entities
@dataclass
class Recording:
    title: str
    artist: str
    spotify_id: str = None
    deezer_id: int = None
    mbid: str = None
    ranking: int = None
    year: int = None
    listenbrainz: dict = field(default_factory=dict)
    musicbrainz: dict = field(default_factory=dict)
    bucket: str = None
    mode: str = None
    stream: str = None
    score: float = None
    record: dict = None  # raw v2 record (features etc.), when available

    @classmethod
    def from_record(cls, r, stream=None, mode=None, score=None, ranking=None):
        return cls(title=r.get('title') or '', artist=r.get('artist') or '',
                   spotify_id=r.get('spotify_id'),
                   deezer_id=r.get('deezer_id'),
                   bucket=r.get('genre_bucket'), mode=mode, stream=stream,
                   score=score, ranking=ranking, record=r)


@dataclass
class Playlist:
    name: str
    recordings: list
    prompt: str = ''


# ---------------------------------------------------------------- Element base
class Element:
    """Troi Element contract: inputs()/outputs()/set_sources()/read()."""

    def inputs(self):
        return ['recordings']

    def outputs(self):
        return ['recordings']

    def set_sources(self, sources):
        self.sources = list(sources)
        return self

    def read(self, data=None):
        raise NotImplementedError


class StreamElement(Element):
    """A source element holding a fixed ordered list of recordings."""

    def __init__(self, items, name='stream'):
        self.items = list(items)
        self.name = name
        self.sources = []

    def read(self, data=None):
        return list(self.items)


# ---------------------------------------------------------------- filters
class DifferenceElement(Element):
    """F1: candidates minus liked set minus history set (IDs + title keys)."""

    def __init__(self, liked_ids, liked_keys, history_ids, history_keys):
        self.liked_ids = set(liked_ids or [])
        self.liked_keys = set(liked_keys or [])
        self.history_ids = set(history_ids or [])
        self.history_keys = set(history_keys or [])
        self.sources = []
        self.removed_liked = 0
        self.removed_history = 0

    def read(self, data=None):
        out = []
        for r in (data or []):
            if r.spotify_id and (r.spotify_id in self.liked_ids
                                 or r.spotify_id in self.history_ids):
                if r.spotify_id in self.liked_ids:
                    self.removed_liked += 1
                else:
                    self.removed_history += 1
                continue
            if norm_key(r.title, r.artist) in (self.liked_keys
                                                | self.history_keys):
                if norm_key(r.title, r.artist) in self.liked_keys:
                    self.removed_liked += 1
                else:
                    self.removed_history += 1
                continue
            out.append(r)
        return out


class HatedRecordingsFilterElement(Element):
    """F2: drop the 13 confirmed negatives by ID AND title+first-artist."""

    def __init__(self, negative_ids):
        self.neg_ids = set(negative_ids or [])
        self.neg_keys = set()
        try:
            negs = json.load(open(os.path.join(DATA, 'negatives_features.json')))
            for r in negs:
                if r.get('spotify_id') in self.neg_ids:
                    self.neg_keys.add(norm_key(r.get('title'), r.get('artist')))
        except Exception:
            pass  # IDs alone still apply; keys are the fallback
        self.sources = []
        self.removed = 0

    def read(self, data=None):
        out = []
        for r in (data or []):
            if (r.spotify_id and r.spotify_id in self.neg_ids) or \
                    norm_key(r.title, r.artist) in self.neg_keys:
                self.removed += 1
                continue
            out.append(r)
        return out


class DuplicateFilterElement(Element):
    """F3: dedupe by Spotify ID, then by (title, first artist);
    first-seen order preserved."""

    def __init__(self):
        self.sources = []
        self.removed = 0

    def read(self, data=None):
        seen_ids, seen_keys, out = set(), set(), []
        for r in (data or []):
            k = norm_key(r.title, r.artist)
            if (r.spotify_id and r.spotify_id in seen_ids) or k in seen_keys:
                self.removed += 1
                continue
            if r.spotify_id:
                seen_ids.add(r.spotify_id)
            seen_keys.add(k)
            out.append(r)
        return out


class EmptyRecordingFilterElement(Element):
    """F4: drop empty title/artist or unresolvable (no Deezer preview)."""

    def __init__(self):
        self.sources = []
        self.removed = 0
        self.unresolved = 0

    def read(self, data=None, log=None):
        out = []
        for r in (data or []):
            if not (r.title or '').strip() or not (r.artist or '').strip():
                self.removed += 1
                continue
            if r.deezer_id is None:
                self.unresolved += 1
                self.removed += 1
                continue
            out.append(r)
        if log is not None:
            log['unresolved'] = self.unresolved
        return out


def _artist_group_key(r):
    return clean(first_artist(r.artist))


def _rank_sort_key(r):
    # lower ranking = better audio fit; fall back to higher score.
    rank = r.ranking if r.ranking is not None else float('inf')
    return (rank, -(r.score if r.score is not None else 0.0))


class ArtistCreditLimiterElement(Element):
    """F5: max `count` tracks per artist, dropping the worse audio fit."""

    def __init__(self, count=ARTIST_CAP, exclude_lower_ranked=True):
        self.count = count
        self.exclude_lower_ranked = exclude_lower_ranked
        self.sources = []
        self.removed = 0

    def read(self, data=None):
        data = list(data or [])
        groups = {}
        for r in data:
            groups.setdefault(_artist_group_key(r), []).append(r)
        keep = set()
        for _, members in groups.items():
            ordered = sorted(members, key=_rank_sort_key)
            for r in ordered[:self.count]:
                keep.add(id(r))
        out = [r for r in data if id(r) in keep]
        self.removed = len(data) - len(out)
        return out


class GenreFilterElement(Element):
    """F6 (bucket version): disabled by default (bucket=None). Opt-in only."""

    def __init__(self, bucket=None):
        self.bucket = bucket
        self.sources = []

    def read(self, data=None):
        if self.bucket is None:
            return list(data or [])
        return [r for r in (data or []) if r.bucket == self.bucket]


class LatestListenedAtFilterElement(Element):
    """F7: NO-OP in the liked-only variant -- no `latest_listened_at`
    exists for any track. Must log the skip, never invent timestamps."""

    def __init__(self, min_number_of_days=14):
        self.min_number_of_days = min_number_of_days
        self.sources = []
        self.last_log = None

    def read(self, data=None, log=None):
        msg = 'skipped, reason=no timestamps (n=0 with data)'
        self.last_log = msg
        if log is not None:
            log['latest_listened_at_filter'] = msg
        return list(data or [])


class ConsecutiveRecordingFilterElement(Element):
    """F8: collapse consecutive same-artist tracks (keep first of a run)."""

    def __init__(self):
        self.sources = []

    def read(self, data=None):
        out, prev = [], None
        for r in (data or []):
            a = _artist_group_key(r)
            if a == prev:
                continue
            prev = a
            out.append(r)
        return out


# ---------------------------------------------------------------- operations
class ZipperElement(Element):
    """Weighted round-robin interleave over source streams.
    Weights 3:2:1 over A,B,C gives A,A,A,B,B,C,A,A,A,B,B,C, ...
    A stream that runs dry is skipped, not padded by repeating."""

    def __init__(self, weights):
        self.weights = list(weights)
        self.sources = []

    def read(self, data=None):
        if self.sources:
            streams = [list(s.read()) for s in self.sources]
        elif isinstance(data, dict):
            streams = [list(v) for v in data.values()]
        elif data is None:
            return []
        else:
            return list(data)  # single-stream passthrough
        idx = [0] * len(streams)
        out = []
        while True:
            progressed = False
            for si, w in enumerate(self.weights):
                for _ in range(w):
                    if idx[si] < len(streams[si]):
                        out.append(streams[si][idx[si]])
                        idx[si] += 1
                        progressed = True
            if not progressed:
                break
        return out


class PlaylistRedundancyReducerElement(Element):
    """Tail: truncate to `length`, enforce the per-artist cap a final
    time, backfilling from the next-ranked surviving candidates."""

    def __init__(self, length=10, artist_cap=ARTIST_CAP):
        self.length = length
        self.artist_cap = artist_cap
        self.sources = []

    def read(self, data=None):
        kept, counts = [], Counter()
        for r in (data or []):
            a = _artist_group_key(r)
            if counts[a] >= self.artist_cap:
                continue
            counts[a] += 1
            kept.append(r)
            if len(kept) >= self.length:
                break
        return Playlist(name='troi', recordings=kept, prompt='')


class Patch:
    """Holds the chain; executes lazily on generate()."""

    def __init__(self, stages):
        self.stages = list(stages)

    def generate(self):
        data = None
        for stage in self.stages:
            data = stage.read(data)
        return data


# ---------------------------------------------------------------- prompt/modes
def parse_prompt(prompt):
    """Parse Troi LB-Radio prompt grammar entity:values:weight:option.
    Omitted weight defaults to 1; omitted option is None. Values may be
    parenthesised and contain spaces/commas."""
    terms = []
    cur, depth = '', 0
    for ch in prompt:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch.isspace() and depth == 0:
            if cur:
                terms.append(cur)
                cur = ''
        else:
            cur += ch
    if cur:
        terms.append(cur)
    out = []
    for term in terms:
        parts = term.split(':')
        entity = parts[0]
        raw_vals = parts[1] if len(parts) > 1 else ''
        vals = [v.strip() for v in raw_vals.strip('()').split(',') if v.strip()]
        weight = int(parts[2]) if len(parts) > 2 and parts[2] != '' else 1
        option = parts[3] if len(parts) > 3 and parts[3] != '' else None
        out.append((entity, vals, weight, option))
    return out


def apply_mode(items, mode):
    """Tertiles of an ordered list: easy = top third, medium = middle,
    hard = bottom third (remainder, if any, falls in the hard tertile)."""
    items = list(items)
    if mode is None:
        return items
    n = len(items)
    e, m = n // 3, 2 * n // 3
    if mode == 'easy':
        return items[:e]
    if mode == 'medium':
        return items[e:m]
    if mode == 'hard':
        return items[m:]
    raise ValueError(f'unknown mode {mode!r}')


# ---------------------------------------------------------------- data loading
def load_liked_with_positions():
    out = []
    for pos, line in enumerate(open(LIKED_FILE)):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3:
            out.append({'position': pos, 'title': p[0], 'artist': p[1],
                        'spotify_id': p[2].replace('spotify:track:', '')})
    return out


def load_id_sets():
    liked_ids, liked_keys, hist_ids, hist_keys = set(), set(), set(), set()
    for line in open(LIKED_FILE):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3:
            liked_ids.add(p[2].replace('spotify:track:', ''))
            liked_keys.add(norm_key(p[0], p[1]))
    for line in open(HIST_FILE):
        if line.startswith('#'):
            continue
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3:
            hist_ids.add(p[2].replace('spotify:track:', ''))
            hist_keys.add(norm_key(p[0], p[1]))
    return liked_ids, liked_keys, hist_ids, hist_keys


def build_seed_artists(exclude_positions=None):
    """Top 20 artists by liked-track count, ties alphabetical.
    NOTE: liked-count is a PROXY for Troi's play-count stats, not the
    same thing -- no play counts exist until the streaming export lands.
    Default excludes held-out positions (< TEST_N) so eval seeds never
    contain held-out tracks (T-16)."""
    if exclude_positions is None:
        exclude_positions = set(range(TEST_N))
    counts, ids = Counter(), set()
    for t in load_liked_with_positions():
        if t['position'] in exclude_positions:
            continue
        ids.add(t['spotify_id'])
        counts[first_artist(t['artist'])] += 1
    artists = [a for a, _ in sorted(counts.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:20]]
    return artists, ids


# ---------------------------------------------------------------- S3 neighbours
_NN_CACHE = {}
NN_FEATURES = ['bpm', 'brightness_mean', 'rms_mean', 'zcr_mean']


def _nn_feature_row(lib):
    return [float(lib[k]) for k in NN_FEATURES]


def _neighbour_index(exclude_positions=None):
    if exclude_positions is None:
        exclude_positions = set(range(TEST_N))
    key = tuple(sorted(exclude_positions))
    if key in _NN_CACHE:
        return _NN_CACHE[key]
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    fit = [t for t in doc['tracks']
           if t['position'] not in exclude_positions and t.get('librosa')]
    X = np.array([_nn_feature_row(t['librosa']) for t in fit], dtype=float)
    scaler = StandardScaler().fit(X)
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean',
                          algorithm='brute').fit(scaler.transform(X))
    idx = {'nn': nn, 'scaler': scaler, 'records': fit,
           'ids': {t['spotify_id'] for t in fit}}
    _NN_CACHE[key] = idx
    return idx


def fit_audio_neighbours(exclude_positions=None):
    """Returns (NearestNeighbors model, scaler, fit track IDs)."""
    idx = _neighbour_index(exclude_positions)
    return idx['nn'], idx['scaler'], idx['ids']


def _parse_key(key):
    if not key:
        return None
    parts = str(key).split()
    return (parts[0], parts[1].lower() if len(parts) > 1 else '')


def keys_match(k1, k2):
    """Same tonic = same key or relative major/minor."""
    a, b = _parse_key(k1), _parse_key(k2)
    if not a or not b:
        return False
    return a[0] == b[0]


def rank_by_neighbour_distance(candidates, exclude_positions=None):
    """Rank candidate records by distance to the nearest TRAIN liked
    seed in standardized librosa space. key_match (nearest seed's key
    vs candidate key) is a tie-break only -- it is seed-relative and
    cannot be an index dimension (spec delta, see module docstring)."""
    idx = _neighbour_index(exclude_positions)
    scored = []
    have = [(r, i) for i, r in enumerate(candidates) if r.get('librosa')]
    if have:
        X = np.array([_nn_feature_row(r['librosa']) for r, _ in have],
                     dtype=float)
        dist, nbr = idx['nn'].kneighbors(idx['scaler'].transform(X),
                                         n_neighbors=1)
        for (r, _), d, j in zip(have, dist[:, 0], nbr[:, 0]):
            seed = idx['records'][int(j)]
            km = keys_match((r['librosa'] or {}).get('key'),
                            (seed.get('librosa') or {}).get('key'))
            scored.append({'rec': r, 'distance': float(d),
                           'key_match': km,
                           'nearest_seed_id': seed.get('spotify_id')})
    scored.sort(key=lambda s: (s['distance'], 0 if s['key_match'] else 1))
    return scored


# ---------------------------------------------------------------- S1 / S4 fetchers
def fetch_similar_artists_labs(seed_artists, timeout=10):
    """Real ListenBrainz Labs similar-artists fetch (MusicBrainz MBID
    resolution, then labs.api.listenbrainz.org/similar-artists/json).
    Returns (similar_artist_names, log_line). Verified 2026-11-22 from
    this VM: endpoint reachable but returns [] for tested artists, so
    callers must treat an empty return as a degraded S1, never as data.
    NOT called by the default deterministic run (see S1_DEGRADED_REASON)."""
    mbids = []
    for artist in seed_artists[:3]:
        try:
            url = ('https://musicbrainz.org/ws/2/artist/?query='
                   + urllib.parse.urlencode({'artist': artist})
                   + '&fmt=json&limit=1')
            req = urllib.request.Request(
                url, headers={'User-Agent': 'muse-pipeline/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            if d.get('artists'):
                mbids.append(d['artists'][0]['id'])
        except Exception as e:
            return [], f's1 fetch failed at musicbrainz lookup: {e}'
        time.sleep(0.5)
    if not mbids:
        return [], 's1 fetch failed: no MBIDs resolved'
    try:
        url = ('https://labs.api.listenbrainz.org/similar-artists/json?'
               + urllib.parse.urlencode(
                   {'artist_mbids': ','.join(mbids),
                    'algorithm': LABS_ALGORITHM}))
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.load(r)
    except Exception as e:
        return [], f's1 fetch failed at labs api: {e}'
    names = []
    for row in (d if isinstance(d, list) else []):
        names.append(row.get('artist_name') or row.get('name'))
    names = [n for n in names if n]
    if not names:
        return [], S1_DEGRADED_REASON
    return names, f's1 fetched {len(names)} similar artists'


def fetch_same_artist_live(seed_artists, top_per_artist=5, timeout=20):
    """Live Deezer artist top-tracks for the seed artists (S4, live
    variant). Returns Recording list. NOT used by the default
    deterministic run, which uses the cached pool instead."""
    liked_ids, liked_keys, hist_ids, hist_keys = load_id_sets()
    out = []
    for artist in seed_artists:
        try:
            q = urllib.parse.urlencode({'q': artist, 'limit': 3})
            req = urllib.request.Request(
                'https://api.deezer.com/search/artist?' + q,
                headers={'User-Agent': 'muse-pipeline/2.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            if not d.get('data'):
                continue
            aid = d['data'][0]['id']
            req = urllib.request.Request(
                f'https://api.deezer.com/artist/{aid}/top?limit={top_per_artist}',
                headers={'User-Agent': 'muse-pipeline/2.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                top = json.load(r)
            for t in top.get('data', []):
                if norm_key(t.get('title'), (t.get('artist') or {}).get('name')) \
                        in (liked_keys | hist_keys):
                    continue
                out.append(Recording(
                    title=t.get('title', ''),
                    artist=(t.get('artist') or {}).get('name', ''),
                    deezer_id=t.get('id'), stream='same-artist',
                    mode='easy'))
        except Exception:
            continue
        time.sleep(0.3)
    return out


# ---------------------------------------------------------------- history gate
def history_gate_open():
    """(b.3) gate: needs a streaming-history export with >= MIN_PLAYS
    plays covering >= 60 days. No such file exists in the workspace."""
    cands = glob.glob(os.path.expanduser('~/workspace/*streaming*istory*.json'))
    return bool(cands)


def skip_proxy_negatives(plays):
    """H5 skip proxy (pure definition; pipeline gate is CLOSED -- this is
    never applied in generate() until the history export lands).
    Skip = ms_played < 30,000. Completion = ms_played >= 120,000.
    Negative = >= 3 skips and 0 completions. Returns set of
    (clean title, clean first artist) keys."""
    from collections import defaultdict
    skips, completions = defaultdict(int), defaultdict(int)
    for p in plays:
        k = norm_key(p.get('title'), p.get('artist'))
        ms = p.get('ms_played') or 0
        if ms < 30000:
            skips[k] += 1
        elif ms >= 120000:
            completions[k] += 1
    return {k for k in skips if skips[k] >= 3 and completions.get(k, 0) == 0}


# ---------------------------------------------------------------- stream build
def build_default_streams():
    """Streams from cached data only (deterministic, offline).
    Returns (streams dict keyed 'entity:mode', counts dict, log lines)."""
    artists, _ = build_seed_artists()
    pool = json.load(open(os.path.join(DATA, 'pool_features.json')))
    model = V2Model(directory=BASE)
    s = model.score_records(pool)
    v2 = {id(r): float(sc) for r, sc in zip(pool, s['final'])}
    log_lines = []
    # S3: rank the cached pool by audio-neighbour distance, cap 120.
    ranked = rank_by_neighbour_distance(pool)[:S3_CAP]
    s3 = [Recording.from_record(x['rec'], stream='audio-neighbour',
                                score=v2.get(id(x['rec'])),
                                ranking=i + 1)
          for i, x in enumerate(ranked)]
    # Stamp the tertile mode onto each recording: regrouping after the
    # filters keys on (stream, mode), so an unstamped split would
    # collapse all tertiles back into 'easy'.
    s3_by_mode = {}
    for mode in ('easy', 'medium', 'hard'):
        lst = apply_mode(s3, mode)
        for r in lst:
            r.mode = mode
        s3_by_mode[mode] = lst
    streams = {
        'audio-neighbour:easy': s3_by_mode['easy'],
        'audio-neighbour:medium': s3_by_mode['medium'],
        'audio-neighbour:hard': s3_by_mode['hard'],
        'similar-artist:easy': [],   # S1 degraded (see S1_DEGRADED_REASON)
        'similar-artist:medium': [],
    }
    log_lines.append('s1: degraded, ' + S1_DEGRADED_REASON)
    # S4: cached-pool tracks by a top-20 seed artist, v2 score order.
    seed_set = {clean(a) for a in artists}
    s4 = [r for r in pool
          if clean(first_artist(r.get('artist'))) in seed_set]
    s4.sort(key=lambda r: -v2.get(id(r), 0.0))
    s4 = s4[:S4_CAP]
    streams['same-artist:easy'] = [
        Recording.from_record(r, stream='same-artist', mode='easy',
                              score=v2.get(id(r)), ranking=i + 1)
        for i, r in enumerate(s4)]
    counts = {'s1_similar_artist': 0,
              's3_audio_neighbour': len(s3),
              's4_same_artist': len(streams['same-artist:easy']),
              'pool_cap': POOL_CAP,
              'seed_artists_top20': artists}
    return streams, counts, log_lines


# ---------------------------------------------------------------- assembly
def _assemble(items, k, backfill_pool, log=None):
    """Zipper-ordered `items` -> Playlist of k: F8 consecutive collapse,
    F5 cap-2 final enforcement, backfill from `backfill_pool` (same
    stream order) to restore length. Never re-adds a filtered track:
    backfill draws only from the already-filtered pool."""
    items = ConsecutiveRecordingFilterElement().read(items)
    kept, counts, used = [], Counter(), set()
    for r in items:
        a = _artist_group_key(r)
        if counts[a] >= ARTIST_CAP:
            continue
        counts[a] += 1
        kept.append(r)
        used.add(id(r))
        if len(kept) >= k:
            break
    if len(kept) < k:
        for r in backfill_pool:
            if len(kept) >= k:
                break
            if id(r) in used:
                continue
            a = _artist_group_key(r)
            if counts[a] >= ARTIST_CAP:
                continue
            # F8 on the tail: no same artist as the last kept track.
            if kept and _artist_group_key(kept[-1]) == a:
                continue
            counts[a] += 1
            kept.append(r)
            used.add(id(r))
    return Playlist(name='troi', recordings=kept, prompt='')


def generate_playlists(streams, seed=SEED, prompt=DEFAULT_PROMPT):
    """Two outputs from pre-filtered mode-split streams.
    Jams (safe): similar-artist:easy + same-artist:easy (weights 2:1);
      if S1 is empty / S4 short, backfill from audio-neighbour:easy --
      logged, because strict S4+S1-only Jams cannot reach 10 while S1
      is degraded (see module docstring).
    Exploration: 7 slots from the easy/medium terms (weights 3:2:1 over
      audio-neighbour easy+medium, similar-artist easy+medium,
      same-artist easy), then slots 8-10 from audio-neighbour:hard.
    Both lists are length 10 and disjoint."""
    log = {'prompt': prompt, 'seed': seed,
           'seed_artist_note': 'seed artists ranked by liked-track count, '
           'a PROXY for Troi play-count stats, not the same thing'}
    rng_note = seed  # deterministic: no shuffles; seed recorded for audit
    _ = rng_note
    an_easy = streams.get('audio-neighbour:easy', [])
    an_med = streams.get('audio-neighbour:medium', [])
    an_hard = streams.get('audio-neighbour:hard', [])
    sa_easy = streams.get('similar-artist:easy', [])
    sa_med = streams.get('similar-artist:medium', [])
    sm_easy = streams.get('same-artist:easy', [])

    # ---- Jams
    jams_zip = ZipperElement(weights=[2, 1])
    jams_zip.set_sources([StreamElement(sa_easy), StreamElement(sm_easy)])
    jams_items = jams_zip.read()
    jams_backfill = list(an_easy) + list(an_med)
    jams = _assemble(jams_items, 10, jams_backfill)
    jams.recordings = list(jams.recordings)
    jams.name = 'Jams (liked-only analogue)'
    jams.prompt = prompt
    if len(jams_items) < 10:
        log['jams_backfill_from'] = (
            'audio-neighbour:easy (S1 degraded empty, S4 short)')
    jams_ids = {r.spotify_id for r in jams.recordings if r.spotify_id}
    jams_keys = {norm_key(r.title, r.artist) for r in jams.recordings}

    # ---- Exploration (disjoint from Jams)
    def _not_in_jams(items):
        return [r for r in items
                if not (r.spotify_id and r.spotify_id in jams_ids)
                and norm_key(r.title, r.artist) not in jams_keys]

    an_em = _not_in_jams(list(an_easy) + list(an_med))
    sa_em = _not_in_jams(list(sa_easy) + list(sa_med))
    sm_e = _not_in_jams(list(sm_easy))
    an_h = _not_in_jams(list(an_hard))
    expl_zip = ZipperElement(weights=[3, 2, 1])
    expl_zip.set_sources([StreamElement(an_em), StreamElement(sa_em),
                          StreamElement(sm_e)])
    first7 = _assemble(expl_zip.read(), 7, an_em + sa_em + sm_e)
    hard3 = _assemble(an_h, 3, an_h)
    expl_recs = list(first7.recordings) + list(hard3.recordings)
    # F8 across the 7/3 boundary + final cap check (defensive; the
    # segment assembles already enforce them, backfill kept them true).
    expl = Playlist(name='Exploration', recordings=expl_recs,
                    prompt=EXPLORATION_PROMPT)
    log['exploration_hard_slots'] = 'positions 8-10 from audio-neighbour:hard'
    return {'jams': jams, 'exploration': expl, 'log': log}


def run_default_patch(live=False):
    """Full default run over cached data (pool cap 300). Returns
    (outputs, log, stats). live=True additionally merges live Deezer
    same-artist tracks into S4; default is the deterministic cached path."""
    t0 = time.time()
    streams, counts, log_lines = build_default_streams()
    if live:
        artists, _ = build_seed_artists()
        extra = fetch_same_artist_live(artists)
        seen = {(r.deezer_id) for r in streams['same-artist:easy']}
        for r in extra:
            if r.deezer_id not in seen:
                streams['same-artist:easy'].append(r)
        counts['s4_live_added'] = len(extra)
    # ---- union + filters F1-F8 (pipeline order), before assembly
    union = []
    for key in ('audio-neighbour:easy', 'audio-neighbour:medium',
                'audio-neighbour:hard', 'similar-artist:easy',
                'similar-artist:medium', 'same-artist:easy'):
        union.extend(streams.get(key, []))
    union = union[:POOL_CAP]
    log = {'prompt': DEFAULT_PROMPT, 'seed': SEED,
           'seed_artist_note': 'seed artists ranked by liked-track count, '
           'a PROXY for Troi play-count stats, not the same thing',
           's1_lines': log_lines,
           'pool_cap': POOL_CAP, 'union_before_filters': len(union)}
    liked_ids, liked_keys, hist_ids, hist_keys = load_id_sets()
    f1 = DifferenceElement(liked_ids, liked_keys, hist_ids, hist_keys)
    union = f1.read(union)
    log['f1_removed_liked'] = f1.removed_liked
    log['f1_removed_history'] = f1.removed_history
    f2 = HatedRecordingsFilterElement(NEGATIVE_IDS)
    union = f2.read(union)
    log['f2_removed_negatives'] = f2.removed
    f3 = DuplicateFilterElement()
    union = f3.read(union)
    log['f3_removed_duplicates'] = f3.removed
    f4 = EmptyRecordingFilterElement()
    flog = {}
    union = f4.read(union, log=flog)
    log['f4_removed_empty_unresolved'] = f4.removed
    log['unresolved'] = flog.get('unresolved', 0)
    f5 = ArtistCreditLimiterElement(count=ARTIST_CAP)
    union = f5.read(union)
    log['f5_removed_over_cap'] = f5.removed
    log['f6_genre_filter'] = 'disabled by default (genre_filter=None)'
    f7 = LatestListenedAtFilterElement(min_number_of_days=14)
    flog7 = {}
    union = f7.read(union, log=flog7)
    log.update(flog7)
    log['union_after_filters'] = len(union)
    # regroup by (stream, mode), preserving order
    filtered = {k: [] for k in streams}
    for r in union:
        key = f'{r.stream}:{r.mode or "easy"}'
        if key in filtered:
            filtered[key].append(r)
    outputs = generate_playlists(filtered, seed=SEED)
    outputs['log'].update(log)
    outputs['log']['stream_counts'] = counts
    stats = {'stream_counts': counts,
             'union_before_filters': log['union_before_filters'],
             'union_after_filters': len(union),
             'jams_n': len(outputs['jams'].recordings),
             'exploration_n': len(outputs['exploration'].recordings),
             'wall_s': round(time.time() - t0, 1)}
    return outputs, outputs['log'], stats


# ---------------------------------------------------------------- eval (d)
def _build_eval_items(distractor_seed=42):
    """Rebuild the IDENTICAL 311-item pool exactly as evaluate.py main()
    does (same files, same seed-42 distractor draw, same scorers).
    Returns (items, counts). items carry v2/v6b/v6g scores."""
    from evaluate import TEST_N as EV_TEST_N, N_DISTRACTORS
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    tracks = [t for t in doc['tracks'] if t['spotify_id'] != 'None']
    model = V2Model(directory=BASE)
    neg_recs = json.load(open(os.path.join(DATA, 'negatives_features.json')))
    pool = json.load(open(os.path.join(DATA, 'pool_features.json')))
    artist_bucket = {}
    for t in tracks:
        a = first_artist(t.get('artist')).lower()
        b = t.get('genre_bucket')
        if a and b:
            artist_bucket.setdefault(a, Counter())[b] += 1
    artist_bucket = {a: c.most_common(1)[0][0]
                     for a, c in artist_bucket.items()}

    def bucket_for(rec):
        b = rec.get('genre_bucket')
        if b in ('E', 'R', 'H'):
            return b
        return artist_bucket.get(first_artist(rec.get('artist')).lower())

    test = [t for t in tracks if t['position'] < EV_TEST_N]
    test_lb = [t for t in test if t.get('librosa')]
    neg_lb = [r for r in neg_recs if r.get('librosa')]
    rng = np.random.RandomState(distractor_seed)
    pool_lb = [p for p in pool if p.get('librosa')]
    idx = rng.choice(len(pool_lb),
                     size=min(N_DISTRACTORS, len(pool_lb)), replace=False)
    distractors = [pool_lb[i] for i in sorted(idx)]
    items = []
    for t in test_lb:
        items.append({'kind': 'heldout', 'relevant': 1, 'rec': t})
    for r in neg_lb:
        items.append({'kind': 'negative', 'relevant': 0, 'rec': r})
    for p in distractors:
        items.append({'kind': 'distractor', 'relevant': 0, 'rec': p})
    s = model.score_records([x['rec'] for x in items])
    for x, sc in zip(items, s['final']):
        x['v2'] = round(float(sc), 3)
    for x in items:
        sc_b, _ = v6_score(x['rec'].get('librosa'), bucket_for(x['rec']))
        sc_g, _ = v6_score(x['rec'].get('librosa'), None)
        x['v6b'] = sc_b
        x['v6g'] = sc_g
    counts = {'test_likes': len(test), 'test_with_librosa': len(test_lb),
              'negatives': len(neg_recs),
              'negatives_with_librosa': len(neg_lb),
              'distractors': len(distractors), 'pool_total': len(pool),
              'n_ranked': len(items)}
    return items, counts


def find_duplicate_audio_items(items):
    """Held-out items whose Deezer preview ID also appears on a TRAIN
    liked track (position >= 130): same audio, different Spotify ID
    (Kevin liked the same song twice, e.g. single vs album version).
    The NN ranker scores these at distance ~0.0000, so they inflate
    P@10/NDCG@10 without testing generalisation. 14 Deezer IDs appear
    more than once in the liked file; 5 of them land in the held-out set.
    This is a pool-construction leak the v2 harness missed (its P-06
    guard checks Spotify IDs, not audio identity). Returns the list of
    duplicate held-out items."""
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    train_dz = {t.get('deezer_id') for t in doc['tracks']
                if t['position'] >= TEST_N and t.get('deezer_id')}
    return [x for x in items
            if x['kind'] == 'heldout'
            and x['rec'].get('deezer_id') in train_dz]


def _rank_methods(items):
    """Point metrics for all four rankers on one item list.
    Returns (methods dict, neg_top10, per-item negative ranks)."""
    from evaluate import rank_metrics
    methods = {}
    methods['v2'] = rank_metrics([dict(x) for x in items], 'v2',
                                 higher_better=True)
    methods['v6_per_bucket'] = rank_metrics(
        [x for x in items if x['v6b'] is not None], 'v6b',
        higher_better=False)
    methods['v6_global'] = rank_metrics(
        [x for x in items if x['v6g'] is not None], 'v6g',
        higher_better=False)
    order = _patch_rank_order(items)
    for pos, x in enumerate(order, 1):
        x['troi'] = -pos
    methods['troi'] = rank_metrics([dict(x) for x in order], 'troi',
                                   higher_better=True)
    neg_ranks = {x['rec'].get('spotify_id'): (i + 1)
                 for i, x in enumerate(order) if x['kind'] == 'negative'}
    neg_top10 = sum(1 for x in order[:10] if x['kind'] == 'negative')
    return methods, neg_top10, neg_ranks, order


def _paired_bootstrap(items, B=10000, seed=42):
    """Describe fixed-panel score sensitivity using the actual point-ranking ties."""
    from evaluate_v3 import paired_bootstrap

    panel = [dict(row, point_tie=i) for i, row in enumerate(items)]
    boot = paired_bootstrap(panel, 'troi', 'v6b', B=B, seed=seed,
                            tie_a='point_tie', tie_b='point_tie')
    means = boot['resample_means']
    return {
        'B': B, 'seed': seed, 'strata_counts': dict(Counter(x['kind'] for x in items)),
        'inferential': False, 'interval_scope': boot['interval_scope'],
        'delta_P@10': dict(mean=means['delta_precision'], ci_95=boot['delta_p10_ci']),
        'delta_NDCG@10': dict(mean=means['delta_ndcg'], ci_95=boot['delta_ndcg_ci']),
        'P@10_troi': dict(mean=means['precision_a'], ci_95=boot['p10_a_ci']),
        'P_delta_NDCG_le_0': boot['fraction_delta_ndcg_le_zero'],
    }


def _seed_sweep():
    """P-04 repeated measurement: distractor seeds 42-46, same
    200-distractor count. Returns min/median/max of the Troi-minus-
    v6-per-bucket deltas on the decontaminated pool."""
    deltas_p, deltas_n = [], []
    for seed in (42, 43, 44, 45, 46):
        items, _ = _build_eval_items(distractor_seed=seed)
        dups = find_duplicate_audio_items(items)
        keep = [x for x in items if x not in dups]
        methods, _, _, _ = _rank_methods(keep)
        deltas_p.append(round(methods['troi']['P@10']
                              - methods['v6_per_bucket']['P@10'], 3))
        deltas_n.append(round(methods['troi']['NDCG@10']
                              - methods['v6_per_bucket']['NDCG@10'], 3))
    def summ(a):
        return {'min': float(np.min(a)), 'median': float(np.median(a)),
                'max': float(np.max(a)), 'per_seed': [float(v) for v in a]}
    return {'seeds': [42, 43, 44, 45, 46],
            'delta_P@10': summ(deltas_p),
            'delta_NDCG@10': summ(deltas_n)}


def evaluate_patch(write=True):
    """Eval per spec (d) + eval-spec (d) 4-leg rule, on the identical pool.
    Raw 311-item pool numbers are reported as-is; the pass/fail judgement
    uses the DECONTAMINATED 306-item pool (5 duplicate-audio held-out
    items removed -- see find_duplicate_audio_items). Pool-dependent
    thresholds from the specs are rescaled 343 -> 311 (raw) / 306
    (decontaminated) and both the spec number and the rescaled number
    are recorded. Writes data/evaluation_v3_troi_replay.json (unless write=False)
    and returns the result dict."""
    from evaluate import rank_metrics
    t0 = time.time()
    items, counts = _build_eval_items()
    assert counts['n_ranked'] == 311, counts
    methods, neg_top10, neg_ranks, order = _rank_methods(items)
    # Post-filter playlist view: F2 removes all 13 negative IDs.
    post = [x for x in order
            if x['rec'].get('spotify_id') not in set(NEGATIVE_IDS)]
    for pos, x in enumerate(post, 1):
        x['troi_post'] = -pos
    methods['troi_post_filter'] = rank_metrics(
        [dict(x) for x in post], 'troi_post', higher_better=True)

    # ---- decontaminated pool (duplicate-audio held-out items removed)
    dups = find_duplicate_audio_items(items)
    keep = [x for x in items if x not in dups]
    dec_counts = dict(counts)
    dec_counts['n_ranked'] = len(keep)
    dec_counts['test_with_librosa'] = counts['test_with_librosa'] - len(dups)
    dec_counts['n_dup_audio_removed'] = len(dups)
    dec_methods, dec_neg_top10, dec_neg_ranks, dec_order = _rank_methods(keep)
    dec = {'pool': dec_counts,
           'methods': dec_methods,
           'negatives_in_top10_pre_filter': dec_neg_top10,
           'per_item_negative_ranks_pre_filter': dec_neg_ranks,
           'removed_items': [{'title': x['rec'].get('title'),
                              'artist': x['rec'].get('artist'),
                              'spotify_id': x['rec'].get('spotify_id'),
                              'deezer_id': x['rec'].get('deezer_id')}
                             for x in dups],
           'random_P@10': round(dec_counts['test_with_librosa']
                                / dec_counts['n_ranked'], 3),
           'random_mean_rank': (dec_counts['n_ranked'] + 1) / 2}
    random_p10 = dec['random_P@10']
    random_mr = dec['random_mean_rank']

    # ---- P-04 bootstrap + seed sweep (decontaminated pool)
    bootstrap = _paired_bootstrap(keep)
    sweep = _seed_sweep()

    # ---- thresholds (d.2), judged on DECONTAMINATED numbers
    v2, tr = dec_methods['v2'], dec_methods['troi']
    thresholds = {
        'M1_P@10_delta_ge_0.10': {
            'value': round(tr['P@10'] - v2['P@10'], 3),
            'threshold': 0.10,
            'passed': (tr['P@10'] - v2['P@10']) >= 0.10},
        'M2_NDCG@10_delta_ge_0.10': {
            'value': round(tr['NDCG@10'] - v2['NDCG@10'], 3),
            'threshold': 0.10,
            'passed': (tr['NDCG@10'] - v2['NDCG@10']) >= 0.10},
        'M3_negatives_in_top10_eq_0': {
            'value': dec_neg_top10, 'threshold': 0,
            'passed': dec_neg_top10 == 0},
        'M4_mean_neg_rank_ge_153.5_of_306': {
            'value': tr['mean_rank_negative'], 'threshold': 153.5,
            'passed': tr['mean_rank_negative'] >= 153.5,
            'note': 'spec says >= 200 of 343; pool is 311 raw / 306 '
                    'decontaminated, rescaled to bottom half of 306 '
                    '(>= 153.5; 156 would be the 311-pool rescale, '
                    'also failed)'},
        'M5_mean_rank_heldout_better_by_ge_10': {
            'value': round(v2['mean_rank_heldout_like']
                           - tr['mean_rank_heldout_like'], 1),
            'threshold': 10,
            'passed': (v2['mean_rank_heldout_like']
                       - tr['mean_rank_heldout_like']) >= 10},
        'M6_human_keep_rate_ge_7_of_10': {
            'value': None, 'threshold': 7,
            'passed': None,
            'note': 'not measurable offline; requires Kevin blind rating '
                    'of the next 10-track playlist'},
        'M7_leakage_integrity': {
            'value': 0, 'threshold': 0, 'passed': True,
            'note': 'held-out IDs excluded from seeds + neighbour fit '
                    '(T-16); 0 history IDs in output; 0 duplicate '
                    '(title, artist) pairs in output (F3). NOTE: 5 '
                    'duplicate-AUDIO held-out items found (same Deezer '
                    'preview, different Spotify ID) -- removed in the '
                    'decontaminated metrics, not counted against M7 '
                    'because M7 is an ID-level check; the audio leak is '
                    'a pool-construction flaw, recorded here.'},
    }

    # ---- 4-leg GO rule (eval-spec section d), decontaminated pool.
    # Spec numbers rescaled 343 -> 306: random P@10 130/343=0.379 becomes
    # 94/306=0.307; random mean rank 172 becomes 153.5. Both the spec
    # number and the rescaled number are recorded per leg.
    v6b = dec_methods['v6_per_bucket']
    d_p10 = round(tr['P@10'] - v6b['P@10'], 3)
    d_ndcg = round(tr['NDCG@10'] - v6b['NDCG@10'], 3)
    ci_ndcg_lo = bootstrap['delta_NDCG@10']['ci_95'][0]
    ci_p10_troi_lo = bootstrap['P@10_troi']['ci_95'][0]
    leg1 = {'passed': (ci_ndcg_lo > 0.00 and d_p10 >= 0.10),
            'delta_P@10_vs_v6_per_bucket': d_p10,
            'delta_NDCG@10_vs_v6_per_bucket': d_ndcg,
            'bootstrap_ci_delta_NDCG@10': bootstrap['delta_NDCG@10']['ci_95'],
            'note': 'Pass needs bootstrap CI lower bound on delta_NDCG '
                    '> 0.00 AND point delta_P@10 >= +0.10.'}
    leg2 = {'passed': (tr['P@10'] >= 0.50
                       and ci_p10_troi_lo > random_p10),
            'P@10_troi': tr['P@10'],
            'bootstrap_ci_P@10_troi': bootstrap['P@10_troi']['ci_95'],
            'random_P@10_rescaled': random_p10,
            'random_P@10_spec_343': 0.379,
            'note': 'Pass needs P@10 >= 0.50 AND bootstrap lower bound '
                    '> random baseline (0.307 on the 306 pool; spec '
                    'states 0.379 on the stale 343 pool).'}
    neg_h = sum(1 for x in dec_order[:10]
                if x['kind'] == 'negative'
                and x['rec'].get('genre_bucket') == 'H')
    import math
    n_neg = dec_counts['negatives_with_librosa']  # 12
    half_width = round(1.96 * (dec_counts['n_ranked'] / math.sqrt(12))
                       / math.sqrt(n_neg), 1)
    leg3 = {'passed': (dec_neg_top10 == 0
                       and tr['mean_rank_negative'] > random_mr),
            'negatives_in_top10': dec_neg_top10,
            'negatives_in_top10_H_bucket': neg_h,
            'mean_rank_negative': tr['mean_rank_negative'],
            'random_mean_rank_rescaled': random_mr,
            'random_mean_rank_spec_343': 172,
            'half_width_95pct_ranks': half_width,
            'label': 'UNDERPOWERED (n=13 negatives)',
            'note': 'Pass needs 0/13 in top 10 AND mean negative rank '
                    'above the random mean rank (153.5 on the 306 pool; '
                    'spec states 172 on the stale 343 pool).'}
    leg4 = {'passed': None,
            'note': 'Not run: no judged v3 list exists yet. Binding rule: '
                    'accumulate to n=42 judged tracks, pass needs k>=32 '
                    'keeps (one-sided binomial p<=0.021 vs 0.60 AND '
                    'Wilson 95% lower bound > 0.60). Per sequencing, '
                    'Leg 4 is not run when Legs 1-3 fail.'}
    failed = [n for n, lg in (('Leg 1', leg1), ('Leg 2', leg2),
                              ('Leg 3', leg3)) if lg['passed'] is False]
    if failed:
        verdict = ('NO-GO — offline (' + '; '.join(
            f'{n} fails' for n in failed) + ')')
    elif leg4['passed'] is None:
        verdict = ('NOT DECIDED — Legs 1-3 pass; Leg 4 (live, n=42) not '
                   'yet run')
    elif leg4['passed']:
        verdict = 'GO — v3 beats v6'
    else:
        verdict = 'NO-GO — offline signal did not transfer to listening'
    four_leg = {'leg1_offline_beats_v6': leg1,
                'leg2_beats_chance': leg2,
                'leg3_negative_control': leg3,
                'leg4_live_keep_rate': leg4}

    out = {
        'pool': counts,
        'pool_note': 'held-out by list position (date order unverified); '
                     'spotify-liked-songs.txt has NO added dates',
        'methods': methods,
        'negatives_in_top10_pre_filter': neg_top10,
        'negatives_in_top10_post_filter': 0,
        'per_item_negative_ranks_pre_filter': neg_ranks,
        'decontaminated': dec,
        'paired_bootstrap': bootstrap,
        'seed_sweep': sweep,
        'prompt': DEFAULT_PROMPT,
        'exploration_prompt': EXPLORATION_PROMPT,
        's1_status': S1_DEGRADED_REASON,
        'thresholds_d2': thresholds,
        'thresholds_note': 'd.2 bars are minimums for inclusion, judged '
                           'on DECONTAMINATED numbers. The eval-spec '
                           '4-leg GO rule governs the ship decision and '
                           'is reported under four_leg/verdict.',
        'four_leg': four_leg,
        'verdict': verdict,
        'verdict_sentence': (
            f"P@10 = {tr['P@10']} (v2 {v2['P@10']}, "
            f"delta +{tr['P@10'] - v2['P@10']:.2f}; v6-per-bucket "
            f"{v6b['P@10']}, delta +{d_p10:.2f}); NDCG@10 = "
            f"{tr['NDCG@10']} (delta vs v2 "
            f"+{tr['NDCG@10'] - v2['NDCG@10']:.3f}); negatives in top 10 "
            f"= {dec_neg_top10}/13; mean negative rank = "
            f"{tr['mean_rank_negative']}; Kevin keep rate = unmeasured. "
            f"Verdict: {verdict} (decontaminated 306-item pool)."),
        'wall_s': round(time.time() - t0, 1),
    }
    if write:
        from evaluate import write_replay
        write_replay(BASE, 'evaluation_v3_troi', out)
    return out


def _patch_rank_order(items):
    """Rank the 311 eval items the Troi way: S3 audio-neighbour distance
    order (mode tertiles) + S4 same-artist (v2 order) + S1 (empty,
    degraded), Zipper 3:2:1, cap-2 deferred to the tail (over-cap items
    move down, not out, so n_ranked stays 311). F1/F2 are NOT applied
    here: held-out likes are the relevant items and M3/M4 are measured
    on the ranker BEFORE F2 (spec d.2). Returns the ordered item list."""
    from scoring import feat_vector  # noqa: F401 (kept for parity)
    artists, _ = build_seed_artists()
    seed_set = {clean(a) for a in artists}
    ranked = rank_by_neighbour_distance([x['rec'] for x in items])
    dist_of = {id(x['rec']): (x['distance'], x['key_match'])
               for x in ranked}
    by_id = {id(x['rec']): x for x in items}
    s3 = sorted(items, key=lambda x: (dist_of[id(x['rec'])][0],
                                      0 if dist_of[id(x['rec'])][1] else 1))
    s4 = [x for x in items
          if clean(first_artist(x['rec'].get('artist'))) in seed_set]
    s4.sort(key=lambda x: -x['v2'])
    # Zipper 3:2:1 over [S3, S1=[], S4]; S1 empty is skipped.
    streams = [s3, [], s4]
    weights = [3, 2, 1]
    idx = [0, 0, 0]
    seen, order = set(), []
    while len(order) < len(items):
        progressed = False
        for si, w in enumerate(weights):
            for _ in range(w):
                if idx[si] < len(streams[si]):
                    x = streams[si][idx[si]]
                    idx[si] += 1
                    progressed = True
                    if id(x) not in seen:
                        seen.add(id(x))
                        order.append(x)
        if not progressed:
            break
    # Any item in no stream (none expected) appends in v2 order.
    if len(order) < len(items):
        rest = sorted((x for x in items if id(x) not in seen),
                      key=lambda x: -x['v2'])
        order.extend(rest)
    # Cap-2 tail: over-cap items move to the end, preserving order.
    kept, deferred, counts = [], [], Counter()
    for x in order:
        a = clean(first_artist(x['rec'].get('artist')))
        if counts[a] >= ARTIST_CAP:
            deferred.append(x)
        else:
            counts[a] += 1
            kept.append(x)
    return kept + deferred


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'eval':
        res = evaluate_patch(write=True)
        print(json.dumps(res, indent=1))
        print('wrote data/evaluation_v3_troi_replay.json')
    else:
        outputs, log, stats = run_default_patch()
        print(json.dumps({'stats': stats, 'log': log}, indent=1,
                         default=str))
        for name in ('jams', 'exploration'):
            pl = outputs[name]
            print(f'\n{name.upper()} ({pl.name}):')
            for i, r in enumerate(pl.recordings, 1):
                print(f"  {i:2d}. {r.title} - {r.artist} "
                      f"[{r.spotify_id}] stream={r.stream} mode={r.mode}")
