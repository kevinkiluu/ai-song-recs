#!/usr/bin/env python3
"""Embeat retrieval-architecture port, v3 (ai-song-recs).

Portions of the retrieval architecture are adapted from Embeat
(github.com/gdstudio-org/Embeat), code MIT per its README; no Embeat
data used. Architecture only: no Embeat weights, datasets, playlists,
or Qdrant server. See research/embeat-spec.md sections (a)-(b).

What is ported: five-channel recall (channel 5 stubbed), per-channel
rank->score remap with summation, ISRC/title dedup, same-artist and
per-artist caps, popular-channel floor, track-bucket voting, block
shuffle. The acoustic encoder is NOT EmbeatMLP (no torch/CUDA, n=650):
it is the v2 fitted transform (median-impute -> standardize -> PCA(15)
from model_v2.npz) L2-normalized; cosine is a dot product. Retrieval is
sklearn brute-force cosine over the 15-d embeddings (no Qdrant server).

Two spec corrections applied (spec section b.8):
  * the verbatim 'remix' drop is REJECTED: 13/650 liked tracks are
    remixes. Only a remix OF THE SEED ITSELF (same title stem + same
    artist) is dropped; other remixes stay eligible.
  * the ISRC-missing drop is DISABLED (is_filter_no_isrc = 0): no-ISRC
    tracks dedup by normalized (title, artist) instead.

Grounded deltas vs the spec (verified against landed v2 files,
2026-11-21; full list in the build report):
  * Eval pool is 311 items (99 held-out likes + 12 negatives with
    librosa + 200 distractors), NOT 130/13/200 = 343. The 13th
    negative (Touch, no librosa) is excluded by evaluate.py itself.
  * Pool is 590 tracks (POOL_CAP 800 not reached), all with librosa,
    genre_bucket None and spotify_id 'None' on every record.
  * No Deezer `rank` is stored anywhere on disk (pool_features.json,
    deezer_matches.json, reccobeats_pool.json all lack it), so channel 2
    degrades to v2-score order within the seed bucket per spec b.2.
    Live Deezer track payloads DO carry `rank` (verified 2026-11-21 on
    5 pool tracks: 165959, 80104, 135138, 219239, 484575, present and
    varying) but it is not persisted and fetching 590 ranks per run
    would break the 500 ms budget. No popularity signal is invented
    from audio features.
  * Pool-bucket keyword table (spec b.5 step 2) is unavailable offline:
    no Deezer genre data is stored for pool artists. Substituted with
    the via_artist bucket fallback (via_artist is always a top-30 liked
    artist, so it covers 590/590; artist-map covers 112/590 directly).
  * evaluate.py ALREADY uses all 13 negatives (docstring says 9, code
    loads 13; 12 with librosa), so spec d.1 change 1 was already true.
  * The spec's final-score rescale formula is self-contradictory
    (it would map the max merged score to ~0.55, not 2.0). Implemented
    as the stated intent: linear rescale so the max is 2.0, round 2dp.
  * Spec d.2.1's "0.05 gain = >=17 additional held-out likes" is wrong:
    +0.05 P@10 is +0.5 held-out likes in a top 10.
  * Spec d.2.2's "mean neg rank >= 20 worse than the best baseline" is
    unsatisfiable vs v2 (292.9 + 20 = 312.9 > 311 max rank); evaluated
    vs the best-P@10 baseline (v6-global, 178.2) instead, and recorded.
  * Channel-3 shuffle omitted: upstream shuffles, but spec test T-06
    requires proxy-DESC order; the two conflict and the test governs.
  * Spec T-08's Y=1.7 contradicts its own linear remap (Y is rank 1 of
    3 -> 1.35). Test asserts the correct remap value; the rank-0-only
    case (1.7) is asserted separately.

Public API (spec section b.10):
  build_index(pool_features, model_npz) -> index dict
  recall_similar / recall_popular / recall_same_artist /
  recall_related_artist / recall_related_track_stub
  vote_track_bucket, merge_by_score, recommend(seed_spotify_id,
  top_k=10, random_seed=616), rank_eval_pool, evaluate_port
"""
import json, math, os, re, resource, subprocess, sys, time
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(BASE, 'data')
LIKED_FILE = os.path.expanduser('~/workspace/spotify-liked-songs.txt')
HIST_FILE = os.path.expanduser('~/workspace/spotify-recommended-tracks.txt')

from scoring import V2Model, clean, feat_vector  # noqa: E402

# ---------------------------------------------------------------- constants
CHANNEL_WEIGHTS = {
    # channel -> (score at rank 0, score at last rank); upstream defaults.
    'similar': (1.7, 1.0),
    'popular': (1.0, 0.8),
    'same_artist': (1.9, 1.0),
    'related_artist': (1.8, 1.0),
    'related_track': (2.0, 1.5),  # unused while channel 5 is stubbed
}
CHANNEL_ORDER = ['similar', 'popular', 'same_artist', 'related_artist',
                 'related_track']
RANDOM_SEED = 616
CALL_LOG = []  # every log line this module emits, for tests/audit

# 20 fixed eval seeds (train likes, by list position) — same list as
# test_embeat_port.py TWENTY_SEEDS; used for contribution/diversity/
# runtime reporting in evaluate_port().
EVAL_SEEDS = [
    '2ngaTpD87A9EL3TAMdTA9V', '44r1T1DN4jHojCmYvYUs9H',
    '3CNBmNfMJmqWMCbXO7dYH9', '77vcz54ZNapi1F9ea84rLs',
    '4NardHbUoTLDH54bOBJ9T8', '2gYj9lubBorOPIVWsTXugG',
    '5VtwUrgljtOyBa773AGSr4', '0Q5VnK2DYzRyfqQRJuUtvi',
    '5LrXxb87HY9uamlNiNugIb', '1Y4uBvlovpqcp3HPSqq5pr',
    '6LCqUn7jGSOBkBcRC3YMEa', '3T4N6wohtfOJN4okuvHiWT',
    '7GX5flRQZVHRAGd6B4TmDO', '3b3e8SeB4jzR0ZlNDwWxMK',
    '6S8oLA2FJ5AnR6XXMcOyZh', '2ePfpPUTnniAYFFtZMT81e',
    '0pLW5T2sp2N4YEEHtF7ilb', '6G0s3kdSxsawbQwnfE2jPg',
    '0j2T0R9dR9qdJYsB7ciXhf', '1VN3d3kSjNrhEx47IFZKKq',
]


def log(msg):
    CALL_LOG.append(msg)
    print(msg, flush=True)


# ---------------------------------------------------------------- data
_MODEL = None
_LIKED = None
_POOL = None
_NEGATIVES = None
_ARTIST_BUCKET = None
_INDEX_CACHE = None
_EVAL_ITEMS_CACHE = None
_EVAL_INDEX_CACHE = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = V2Model(directory=BASE)
    return _MODEL


def load_liked():
    global _LIKED
    if _LIKED is None:
        _LIKED = json.load(open(os.path.join(BASE, 'features_v2.json')))['tracks']
    return _LIKED


def load_pool():
    global _POOL
    if _POOL is None:
        _POOL = json.load(open(os.path.join(DATA, 'pool_features.json')))
    return _POOL


def load_negatives():
    global _NEGATIVES
    if _NEGATIVES is None:
        _NEGATIVES = json.load(open(os.path.join(DATA, 'negatives_features.json')))
    return _NEGATIVES


def artist_bucket_map(records=None):
    """First-artist (normalized) -> most common genre bucket among likes."""
    global _ARTIST_BUCKET
    if records is None and _ARTIST_BUCKET is not None:
        return _ARTIST_BUCKET
    m = {}
    for t in load_liked() if records is None else records:
        a = normalize_artist(t.get('artist') or '')
        b = t.get('genre_bucket')
        if a and b:
            m.setdefault(a, Counter())[b] += 1
    buckets = {a: c.most_common(1)[0][0] for a, c in m.items()}
    if records is None:
        _ARTIST_BUCKET = buckets
    return buckets


# ---------------------------------------------------------------- identity
def normalize_artist(s):
    """First artist only, normalized the way pipeline.clean() does."""
    return clean((s or '').split(',')[0].strip())


def title_stem(title):
    """Title with parentheticals, dash-suffixes and a trailing 'remix'
    removed, for seed-remix detection. 'Click (Knock2 Remix)' -> 'click';
    'U Make Me - Asketa & Natan Chaim Remix' -> 'u make me'."""
    s = title or ''
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\[[^\]]*\]', '', s)
    s = re.split(r'\s+-\s+', s)[0]
    s = clean(s)
    s = re.sub(r'\s+remix$', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def is_seed_remix(cand, seed_rec):
    """True iff the candidate is a remix of the seed track itself:
    title contains 'remix', same title stem, same artist."""
    if 'remix' not in clean(cand.get('title') or ''):
        return False
    return (title_stem(cand.get('title') or '')
            == title_stem(seed_rec.get('title') or '')
            and normalize_artist(cand.get('artist') or '')
            == normalize_artist(seed_rec.get('artist') or ''))


def cand_key(rec):
    """Merge identity: ISRC when present, else normalized (title, artist).
    No-ISRC tracks are never dropped (is_filter_no_isrc = 0)."""
    if '_evaluation_recording_id' in rec:
        return ('evaluation', rec['_evaluation_recording_id'])
    isrc = rec.get('isrc')
    if isrc:
        return ('isrc', str(isrc))
    return ('ta', clean(rec.get('title') or ''),
            normalize_artist(rec.get('artist') or ''))


def _is_seed(cand, seed_rec):
    if (cand.get('deezer_id') is not None
            and seed_rec.get('deezer_id') is not None
            and cand.get('deezer_id') == seed_rec.get('deezer_id')):
        return True
    sid_c, sid_s = cand.get('spotify_id'), seed_rec.get('spotify_id')
    if sid_c and sid_s and sid_c != 'None' and sid_c == sid_s:
        return True
    return ((clean(cand.get('title') or ''),
             normalize_artist(cand.get('artist') or ''))
            == (clean(seed_rec.get('title') or ''),
                normalize_artist(seed_rec.get('artist') or '')))


# ---------------------------------------------------------------- embeddings
def embed_records(model, records):
    """v2 fitted transform -> PCA(15) -> L2-normalized acoustic embedding.
    Reuses scoring.feat_vector and the V2Model transforms; no feature
    extraction is reimplemented here. Rows with no signal (all-NaN
    features) impute to the median and yield a defined, non-unit vector
    guarded to zero-norm instead of NaN."""
    if not records:
        return np.zeros((0, model.pca_comp.shape[0]))
    X, _ = zip(*[feat_vector(r) for r in records])
    X = np.vstack(X)
    X = np.where(np.isnan(X), model.median, X)
    Z = (X - model.mean) / model.scale
    Pv = (Z - model.pca_mean) @ model.pca_comp.T
    n = np.linalg.norm(Pv, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return Pv / n


def _assign_bucket(rec, artist_bucket):
    """Pool/eval record -> (bucket, source). Label first, then the
    liked-artist map, then the via_artist bucket (spec b.5 step 2
    keyword table is unavailable offline: no Deezer genre data is
    stored for pool artists), else 'O' / 'unknown'."""
    b = rec.get('genre_bucket')
    if b in ('E', 'R', 'H', 'O'):
        return b, 'label'
    a = normalize_artist(rec.get('artist') or '')
    if a in artist_bucket:
        return artist_bucket[a], 'artist'
    v = normalize_artist(rec.get('via_artist') or '')
    if v in artist_bucket:
        return artist_bucket[v], 'via'
    return 'O', 'unknown'


# ---------------------------------------------------------------- index
def build_index(pool_features, model_npz, *, artist_buckets=None):
    """Build the retrieval index over a record list (pool tracks, or the
    eval items). Embeddings + brute-force cosine NN + dict payload
    indexes (artist, bucket, popularity proxy). model_npz is a path to
    model_v2.npz or a V2Model instance."""
    from sklearn.neighbors import NearestNeighbors
    if isinstance(model_npz, V2Model):
        model = model_npz
    else:
        model = V2Model.__new__(V2Model)
        z = np.load(os.path.expanduser(str(model_npz)))
        model.median = z['imputer_median']
        model.mean = z['scaler_mean']
        model.scale = z['scaler_scale']
        model.pca_comp = z['pca_components']
        model.pca_mean = z['pca_mean']
        model.centroids = z['centroids']
        model.lr_coef = z['lr_coef'][0]
        model.lr_intercept = float(z['lr_intercept'][0])
        model.negdist_mean = float(z['negdist_mean'])
        model.negdist_std = float(z['negdist_std'])
        model.lr_mean = float(z['lr_mean'])
        model.lr_std = float(z['lr_std'])
    ab = artist_bucket_map() if artist_buckets is None else artist_buckets
    recs = []
    for r in pool_features:
        c = dict(r)
        b, src = _assign_bucket(c, ab)
        c['bucket'] = b
        c['bucket_source'] = src
        recs.append(c)
    emb = embed_records(model, recs)
    # Popularity proxy: DEGRADED to v2-score order within bucket.
    # No Deezer `rank` is stored in pool_features.json (verified 2026-11-21:
    # absent from pool_features.json, deezer_matches.json and
    # reccobeats_pool.json; live Deezer payloads DO carry rank, present and
    # varying on 5/5 sampled tracks, but it is not persisted and fetching
    # 590 ranks per run would break the 500 ms budget). No popularity
    # signal is invented from audio features.
    log('channel2 popularity proxy: DEGRADED to v2-score order within bucket '
        '- Deezer rank not stored in pool_features.json (live payloads '
        'carry rank, verified 5/5, but it is not persisted)')
    if recs:
        v2 = model.score_records(recs)['final']
    else:
        v2 = np.zeros(0)
    proxy = np.zeros(len(recs))
    ranges = {}
    for b in sorted({r['bucket'] for r in recs}):
        idxs = [i for i, r in enumerate(recs) if r['bucket'] == b]
        vals = v2[idxs]
        lo, hi = float(vals.min()), float(vals.max())
        ranges[b] = (lo, hi)
        proxy[idxs] = 0.5 if hi <= lo else (vals - lo) / (hi - lo)
    for i, r in enumerate(recs):
        r['popularity_proxy'] = float(proxy[i])
        r['v2_raw'] = float(v2[i])
    nn = None
    if recs:
        nn = NearestNeighbors(
            n_neighbors=min(100, len(recs)), algorithm='brute',
            metric='cosine')
        nn.fit(emb)
    artist_index = defaultdict(list)
    bucket_index = defaultdict(list)
    related_by_via = defaultdict(set)
    for i, r in enumerate(recs):
        artist_index[normalize_artist(r.get('artist') or '')].append(i)
        bucket_index[r['bucket']].append(i)
        via = normalize_artist(r.get('via_artist') or '')
        rel = normalize_artist(r.get('related_artist') or '')
        if via and rel:
            related_by_via[via].add(rel)
    return {
        'records': recs, 'embeddings': emb, 'nn': nn, 'model': model,
        'artist_index': dict(artist_index), 'artist_buckets': ab,
        'bucket_index': dict(bucket_index),
        'related_by_via': {k: set(v) for k, v in related_by_via.items()},
        'bucket_v2_range': ranges,
        'seed_bucket_cache': {},
        'bucket_counts': dict(Counter(r['bucket_source'] for r in recs)),
    }


def get_index():
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = build_index(load_pool(),
                                   os.path.join(BASE, 'model_v2.npz'))
    return _INDEX_CACHE


def seed_proxy(index, seed_rec, bucket):
    """Seed's popularity proxy: seed v2 score min-max normalized within
    the seed bucket's pool range. 0.0 (no filtering) if unavailable."""
    lo_hi = index['bucket_v2_range'].get(bucket)
    if not lo_hi:
        return 0.0
    lo, hi = lo_hi
    if hi <= lo:
        return 0.5
    s = float(index['model'].score_records([seed_rec])['final'][0])
    return min(1.0, max(0.0, (s - lo) / (hi - lo)))


def _seed_emb(index, seed_rec):
    return embed_records(index['model'], [seed_rec])[0]


def related_artist_set(index, seed_rec):
    """Deezer-related artists of the seed track's artist, as recorded in
    pool provenance (via_artist -> related_artist)."""
    return set(index['related_by_via'].get(
        normalize_artist(seed_rec.get('artist') or ''), set()))


# ---------------------------------------------------------------- filters
def passes_filters(cand, seed_rec, seed_proxy=0.0):
    """Upstream per-candidate filters (filter_similar/filter_others),
    with the two v3 corrections: only a remix OF THE SEED is dropped,
    and the popularity gate uses the proxy with min(seed, 0.1)."""
    if _is_seed(cand, seed_rec):
        return False
    if is_seed_remix(cand, seed_rec):
        return False
    ct = clean(cand.get('title') or '')
    st = clean(seed_rec.get('title') or '')
    if st and ct.startswith(st + ' '):
        return False
    gate = min(seed_proxy, 0.1) if seed_proxy else 0.0
    if float(cand.get('popularity_proxy') or 0.0) < gate:
        return False
    return True


def passes_channel1_filter(cand, seed_rec, seed_bucket, bucket_voted,
                           seed_proxy=0.0):
    """Channel-1 filter: unknown-genre ('O') candidates are excluded
    from the genre-filtered list, and (when the bucket was voted) the
    candidate must match the voted bucket."""
    if (cand.get('bucket') or 'O') == 'O':
        return False
    if bucket_voted and (cand.get('bucket') or 'O') != seed_bucket:
        return False
    return passes_filters(cand, seed_rec, seed_proxy=seed_proxy)


def _dedup_titles(cands):
    seen, out = set(), []
    for c in cands:
        t = clean(c.get('title') or '')
        if t in seen:
            continue
        seen.add(t)
        out.append(c)
    return out


def _dedup_isrc(cands):
    seen, out = set(), []
    for c in cands:
        isrc = c.get('isrc')
        if isrc:
            if isrc in seen:
                continue
            seen.add(isrc)
        out.append(c)
    return out


def _cand(index, i, cosine=None):
    c = dict(index['records'][i])
    if cosine is not None:
        c['cosine'] = float(cosine)
    return c

# ---------------------------------------------------------------- recall
def _unfiltered_similar(index, seed_rec, limit):
    """Top-`limit` pool records by acoustic cosine to the seed, with the
    common filters and title dedup applied. Returns candidate dicts with
    'cosine' set."""
    if not index['records'] or index['nn'] is None:
        return []
    se = _seed_emb(index, seed_rec)
    n = min(limit, len(index['records']))
    dist, idx = index['nn'].kneighbors(se.reshape(1, -1),
                                       n_neighbors=n)
    out = []
    for d, i in zip(dist[0], idx[0]):
        c = _cand(index, int(i), cosine=1.0 - float(d))
        out.append(c)
    sp = seed_proxy(index, seed_rec,
                    _assign_bucket(seed_rec, index['artist_buckets'])[0])
    out = [c for c in out if passes_filters(c, seed_rec, seed_proxy=sp)]
    return _dedup_titles(_dedup_isrc(out))


def vote_track_bucket(index, seed_rec):
    """Track-bucket vote (spec b.5): weight channel-1 candidates' buckets
    by 1/sqrt(rank+1) (O excluded from voting); the top bucket wins only
    if it outscores the runner-up by > 2.0x, else fall back to the seed
    artist's bucket, else 'fallback_O'. Returns (bucket, source) with
    source in {vote, artist, fallback_O}. Cached per seed."""
    sid = seed_rec.get('spotify_id')
    if sid in index['seed_bucket_cache']:
        return index['seed_bucket_cache'][sid]
    unf = _unfiltered_similar(index, seed_rec, 100)
    votes = Counter()
    for rank, c in enumerate(unf):
        b = c.get('bucket')
        if b in ('E', 'R', 'H'):
            votes[b] += 1.0 / math.sqrt(rank + 1)
    result = None
    if votes:
        top = votes.most_common(2)
        if len(top) == 1 or top[0][1] > 2.0 * top[1][1]:
            result = (top[0][0], 'vote')
    if result is None:
        b = index['artist_buckets'].get(normalize_artist(seed_rec.get('artist') or ''))
        if b in ('E', 'R', 'H', 'O'):
            result = (b, 'artist')
        else:
            result = ('O', 'fallback_O')
    index['seed_bucket_cache'][sid] = result
    return result


def recall_similar(index, seed_rec, top_k=10):
    """Channel 1: acoustic similarity, genre-filtered to the seed's
    voted bucket. Limit min(top_k*10, 512). If the bucket filter yields
    <10 candidates, fall back to unfiltered neighbours (logged)."""
    limit = min(top_k * 10, 512)
    unf = _unfiltered_similar(index, seed_rec, limit)
    bucket, source = vote_track_bucket(index, seed_rec)
    if bucket in ('E', 'R', 'H'):
        filt = [c for c in unf if c.get('bucket') == bucket]
        if len(filt) >= 10:
            return filt[:limit]
        log(f'channel1 bucket fallback: bucket {bucket} filter gave '
            f'{len(filt)} candidates for seed '
            f"{seed_rec.get('spotify_id')}; using {len(unf)} unfiltered "
            f'neighbours')
        return unf[:limit]
    log(f'channel1 bucket fallback: seed bucket {bucket} (source {source}) '
        f'is not E/R/H; using {len(unf)} unfiltered neighbours for seed '
        f"{seed_rec.get('spotify_id')}")
    return unf[:limit]


def recall_popular(index, seed_rec, top_k=10):
    """Channel 2: same-bucket popular. Order by popularity proxy DESC,
    take min(top_k*2, 128), re-rank by acoustic cosine. The proxy is the
    degraded v2-score proxy (see build_index)."""
    limit = min(top_k * 2, 128)
    bucket, _ = vote_track_bucket(index, seed_rec)
    sp = seed_proxy(index, seed_rec, bucket)
    cands = [_cand(index, i) for i in index['bucket_index'].get(bucket, [])]
    cands = [c for c in cands if passes_filters(c, seed_rec, seed_proxy=sp)]
    cands = _dedup_titles(_dedup_isrc(cands))
    cands.sort(key=lambda c: (-c['popularity_proxy'], c.get('title', '')))
    pre = cands[:limit]
    se = _seed_emb(index, seed_rec)
    pos = {}
    for i, r in enumerate(index['records']):
        pos[cand_key(r)] = i
    out = []
    for rank, c in enumerate(pre):
        i = pos.get(cand_key(c))
        cos = float(index['embeddings'][i] @ se) if i is not None else 0.0
        d = dict(c)
        d['popular_rank'] = rank
        d['cosine'] = cos
        out.append(d)
    out.sort(key=lambda c: (-c['cosine'], c.get('title', '')))
    return out


def recall_same_artist(index, seed_rec, top_k=10, random_seed=616):
    """Channel 3: same artist, ordered by popularity proxy DESC, limit
    min(top_k*2, 128). (Upstream shuffles before filtering; the shuffle
    is omitted because spec test T-06 requires proxy-DESC order and
    the two conflict. Determinism comes from the fixed seed.)"""
    limit = min(top_k * 2, 128)
    bucket, _ = vote_track_bucket(index, seed_rec)
    sp = seed_proxy(index, seed_rec, bucket)
    idxs = index['artist_index'].get(
        normalize_artist(seed_rec.get('artist') or ''), [])
    cands = [_cand(index, i) for i in idxs]
    cands = [c for c in cands if passes_filters(c, seed_rec, seed_proxy=sp)]
    cands = _dedup_titles(_dedup_isrc(cands))
    cands.sort(key=lambda c: (-c['popularity_proxy'], c.get('title', '')))
    return cands[:limit]


def recall_related_artist(index, seed_rec, top_k=10):
    """Channel 4: pool tracks whose artist is a Deezer-related artist of
    the seed artist (from pool provenance), ranked by acoustic cosine,
    limit min(top_k*2, 128), re-sorted by popularity proxy DESC."""
    limit = min(top_k * 2, 128)
    rel = related_artist_set(index, seed_rec)
    if not rel:
        return []
    bucket, _ = vote_track_bucket(index, seed_rec)
    sp = seed_proxy(index, seed_rec, bucket)
    se = _seed_emb(index, seed_rec)
    scored = []
    for i, r in enumerate(index['records']):
        if normalize_artist(r.get('artist') or '') not in rel:
            continue
        c = dict(r)
        if not passes_filters(c, seed_rec, seed_proxy=sp):
            continue
        c['cosine'] = float(index['embeddings'][i] @ se)
        scored.append(c)
    scored = _dedup_titles(_dedup_isrc(scored))
    scored.sort(key=lambda c: -c['cosine'])
    scored = scored[:limit]
    scored.sort(key=lambda c: (-c['popularity_proxy'], c.get('title', '')))
    return scored


def recall_related_track_stub(index, seed_rec, top_k=10):
    """Channel 5: STUB. No playlist corpus exists (liked file has no
    playlist membership; the 4 genre playlists are label sources, not
    co-occurrence), so no collaborative signal can be computed. Returns
    [] on every call and logs the stub. Any claim of collaborative
    filtering in v3 output would be false."""
    log('channel5=stub:no-playlist-corpus')
    return []


# ---------------------------------------------------------------- merge
def _merge_scores(channel_lists):
    """Per-channel rank->score linear remap (upstream ranges) summed
    across channels. Returns the full scored list (pre-cap), sorted by
    merged score desc, then popularity proxy, then title/artist for
    deterministic ties. Each entry carries 'score_raw' and 'sources'."""
    per_key = {}
    for ch in CHANNEL_ORDER:
        lst = channel_lists.get(ch) or []
        if not lst:
            continue
        hi, lo = CHANNEL_WEIGHTS[ch]
        ded = _dedup_isrc(list(lst))
        n = len(ded)
        for rank, c in enumerate(ded):
            s = hi if n <= 1 else hi - (hi - lo) * rank / (n - 1)
            k = cand_key(c)
            e = per_key.get(k)
            if e is None:
                e = {'cand': dict(c), 'score_raw': 0.0, 'sources': []}
                per_key[k] = e
            e['score_raw'] += s
            if ch not in e['sources']:
                e['sources'].append(ch)
    out = []
    for k, e in per_key.items():
        d = dict(e['cand'])
        d['key'] = k
        d['score_raw'] = e['score_raw']
        d['sources'] = [c for c in CHANNEL_ORDER if c in e['sources']]
        out.append(d)
    out.sort(key=lambda d: (-d['score_raw'],
                            -float(d.get('popularity_proxy') or 0.0),
                            d.get('title', ''), d.get('artist', '')))
    return out


def merge_by_score(channel_lists, top_k=10, random_seed=616,
                   seed_artist=None):
    """Upstream merge_result_by_score: summed remapped scores, then final
    assembly with same-artist cap max(1, int(top_k * U(0.15, 0.20)))
    (one fresh draw per call from random_seed), any other single artist
    <= max(1, int(top_k * 0.15)), duplicate track names excluded, a
    popular-channel floor of int(top_k * 0.1) back-filled, shortfall
    back-fill from remaining candidates. Final scores rescaled so the
    max is 2.0 (round 2dp); 'score_raw' keeps the pre-rescale sum."""
    rng = np.random.RandomState(random_seed)
    ratio = float(rng.uniform(0.15, 0.20))
    same_cap = max(1, int(top_k * ratio))
    other_cap = max(1, int(top_k * 0.15))
    popular_floor = int(top_k * 0.1)
    merged = _merge_scores(channel_lists)
    seed_a = normalize_artist(seed_artist or '')
    selected, counts, titles = [], Counter(), set()
    for d in merged:
        if len(selected) >= top_k:
            break
        t = clean(d.get('title') or '')
        if t in titles:
            continue
        a = normalize_artist(d.get('artist') or '')
        cap = same_cap if (seed_a and a == seed_a) else other_cap
        if counts[a] >= cap:
            continue
        selected.append(d)
        counts[a] += 1
        titles.add(t)
    # popular floor: back-fill from popular candidates if under-filled
    n_pop = sum(1 for d in selected if 'popular' in d.get('sources', []))
    if n_pop < popular_floor:
        for d in merged:
            if n_pop >= popular_floor:
                break
            if 'popular' not in d.get('sources', []):
                continue
            if d in selected:
                continue
            a = normalize_artist(d.get('artist') or '')
            cap = same_cap if (seed_a and a == seed_a) else other_cap
            if counts[a] >= cap:
                continue
            # replace the lowest-scored selected non-popular track
            for j in range(len(selected) - 1, -1, -1):
                if 'popular' not in selected[j].get('sources', []):
                    old = selected[j]
                    counts[normalize_artist(old.get('artist') or '')] -= 1
                    titles.discard(clean(old.get('title') or ''))
                    selected[j] = d
                    counts[a] += 1
                    titles.add(clean(d.get('title') or ''))
                    n_pop += 1
                    break
    # shortfall back-fill: remaining candidates in merged order, relaxing
    # the per-artist cap but keeping the same-artist cap and dup-title rule
    if len(selected) < top_k:
        for d in merged:
            if len(selected) >= top_k:
                break
            if d in selected:
                continue
            t = clean(d.get('title') or '')
            if t in titles:
                continue
            a = normalize_artist(d.get('artist') or '')
            if seed_a and a == seed_a and counts[a] >= same_cap:
                continue
            selected.append(d)
            counts[a] += 1
            titles.add(t)
    if selected:
        mx = max(d['score_raw'] for d in selected)
        for d in selected:
            d['score'] = round(d['score_raw'] * 2.0 / mx, 2) if mx > 0 else 0.0
    return selected


def count_adjacent_repeats(items):
    """Adjacent same-artist pairs (album pass is a no-op: pool records
    carry no album field)."""
    n = 0
    for a, b in zip(items, items[1:]):
        if normalize_artist(a.get('artist') or '') == \
                normalize_artist(b.get('artist') or ''):
            n += 1
    return n


def shuffle_result_block(items):
    """Upstream shuffle_result_block (max_block_len=1): greedy
    arrangement so no two adjacent items share an artist. Deterministic:
    stable order by merged score, first fitting candidate wins."""
    remaining = list(items)
    out = []
    while remaining:
        last_a = normalize_artist(out[-1].get('artist') or '') if out else None
        pick = None
        for i, d in enumerate(remaining):
            if normalize_artist(d.get('artist') or '') != last_a:
                pick = i
                break
        if pick is None:
            pick = 0
        out.append(remaining.pop(pick))
    return out


# ---------------------------------------------------------------- recommend
def recommend(seed_spotify_id, top_k=10, random_seed=616):
    """Full ported recommendation for one seed liked track. Returns the
    final top_k with per-track 'sources', 'score' (rescaled, max 2.0),
    'score_raw', 'seed_bucket' and 'seed_bucket_source'."""
    liked = {t['spotify_id']: t for t in load_liked()}
    seed_rec = liked.get(seed_spotify_id)
    if seed_rec is None:
        raise ValueError(f'unknown seed spotify id: {seed_spotify_id}')
    index = get_index()
    bucket, source = vote_track_bucket(index, seed_rec)
    ch1 = recall_similar(index, seed_rec, top_k=top_k)
    ch2 = recall_popular(index, seed_rec, top_k=top_k)
    ch3 = recall_same_artist(index, seed_rec, top_k=top_k,
                             random_seed=random_seed)
    ch4 = recall_related_artist(index, seed_rec, top_k=top_k)
    ch5 = recall_related_track_stub(index, seed_rec, top_k=top_k)
    # exclusion: never re-recommend Liked Songs or history (by key; the
    # pool builder already excluded them, this is defense in depth)
    from recommend_v2 import load_id_sets

    _, liked_keys, _, hist_keys = load_id_sets()
    excluded = liked_keys | hist_keys

    def _keep(c):
        return (clean(c.get('title') or ''),
                clean((c.get('artist') or '').split(',')[0])) not in excluded

    lists = {'similar': [c for c in ch1 if _keep(c)],
             'popular': [c for c in ch2 if _keep(c)],
             'same_artist': [c for c in ch3 if _keep(c)],
             'related_artist': [c for c in ch4 if _keep(c)],
             'related_track': ch5}
    merged = merge_by_score(lists, top_k=top_k, random_seed=random_seed,
                            seed_artist=seed_rec.get('artist') or '')
    final = shuffle_result_block(merged)
    n_rep = count_adjacent_repeats(final)
    if n_rep:
        log(f'block shuffle: {n_rep} adjacent artist repeats remain for '
            f'seed {seed_spotify_id}')
    for d in final:
        d['seed_bucket'] = bucket
        d['seed_bucket_source'] = source
    return final

# ---------------------------------------------------------------- eval
def build_eval_items():
    """Rebuild the IDENTICAL 311-item eval pool from evaluate.py
    (TEST_N=130, 200 distractors, seed 42), with v2/v6b/v6g scores.
    Verifies the baselines reproduce evaluation_v2.json exactly; raises
    if they do not, so a drifted pool can never silently pass."""
    global _EVAL_ITEMS_CACHE
    if _EVAL_ITEMS_CACHE is not None:
        return _EVAL_ITEMS_CACHE
    from evaluate import (TEST_N, N_DISTRACTORS, SEED, ndcg_at_k,
                          precision_at_k, rank_metrics)
    from scoring import v6_score
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    tracks = [t for t in doc['tracks'] if t['spotify_id'] != 'None']
    model = get_model()
    neg_recs = load_negatives()
    pool = load_pool()
    artist_bucket = {}
    for t in tracks:
        a = (t.get('artist') or '').split(',')[0].strip().lower()
        b = t.get('genre_bucket')
        if a and b:
            artist_bucket.setdefault(a, Counter())[b] += 1
    artist_bucket = {a: c.most_common(1)[0][0]
                     for a, c in artist_bucket.items()}

    def bucket_for(rec):
        b = rec.get('genre_bucket')
        if b in ('E', 'R', 'H'):
            return b
        a = (rec.get('artist') or '').split(',')[0].strip().lower()
        return artist_bucket.get(a)

    test = [t for t in tracks if t['position'] < TEST_N]
    test_lb = [t for t in test if t.get('librosa')]
    neg_lb = [r for r in neg_recs if r.get('librosa')]
    rng = np.random.RandomState(SEED)
    pool_lb = [p for p in pool if p.get('librosa')]
    idx = rng.choice(len(pool_lb), size=min(N_DISTRACTORS, len(pool_lb)),
                     replace=False)
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
    # cross-check: baselines must reproduce evaluation_v2.json exactly
    ref = json.load(open(os.path.join(BASE, 'evaluation_v2.json')))
    got = {
        'v2': rank_metrics([x for x in items], 'v2', higher_better=True),
        'v6_per_bucket': rank_metrics(
            [x for x in items if x['v6b'] is not None], 'v6b',
            higher_better=False),
        'v6_global': rank_metrics(
            [x for x in items if x['v6g'] is not None], 'v6g',
            higher_better=False),
    }
    for m in ('v2', 'v6_per_bucket', 'v6_global'):
        for k in ('P@10', 'NDCG@10', 'mean_rank_heldout_like',
                  'mean_rank_negative'):
            field = 'legacy_topk_hit_ndcg' if k == 'NDCG@10' else k
            if got[m][field] != ref[m][k]:
                raise RuntimeError(
                    f'eval pool drift: baseline {m} {k} recomputed as '
                    f"{got[m][field]}, evaluation_v2.json says {ref[m][k]}")
    _EVAL_ITEMS_CACHE = (items, got)
    return _EVAL_ITEMS_CACHE


def _eval_index():
    global _EVAL_INDEX_CACHE
    if _EVAL_INDEX_CACHE is None:
        items, _ = build_eval_items()
        _EVAL_INDEX_CACHE = build_index([x['rec'] for x in items],
                                        os.path.join(BASE, 'model_v2.npz'))
    return _EVAL_INDEX_CACHE


def eval_seed():
    """Medoid eval seed: the train like (position >= 130, with librosa)
    whose artist is a top-30 liked artist and whose embedding is closest
    to the train-like centroid. A top-30 artist keeps channel 4 live in
    the eval ranking; the medoid keeps channel 1 representative."""
    liked = load_liked()
    train = [t for t in liked
             if t['position'] is not None and t['position'] >= 130
             and t.get('librosa') and t['spotify_id'] != 'None']
    model = get_model()
    emb = embed_records(model, train)
    centroid = emb.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    from collections import Counter as _C
    top_artists = {a for a, _ in _C(
        normalize_artist(t.get('artist') or '') for t in liked
    ).most_common(30)}
    best, best_cos = None, -2.0
    for t, e in zip(train, emb):
        if normalize_artist(t.get('artist') or '') not in top_artists:
            continue
        c = float(e @ centroid)
        if c > best_cos:
            best, best_cos = t, c
    return best


def rank_eval_pool(seed_spotify_id=None, channels=None, top_k=10):
    """Rank the identical 311-item eval pool with the ported ranker.
    The query is one seed track (default: the medoid eval seed). Ranking
    uses the pre-cap merged score (the v2/v6 eval precedent ranks by raw
    score, without MMR/diversity assembly); items in no channel score
    0.0. Returns the full ranked item list with 'port' set."""
    items, _ = build_eval_items()
    index = _eval_index()
    liked = {t['spotify_id']: t for t in load_liked()}
    if seed_spotify_id is None:
        seed_rec = eval_seed()
    else:
        seed_rec = liked.get(seed_spotify_id)
        if seed_rec is None:
            raise ValueError(f'unknown seed spotify id: {seed_spotify_id}')
    want = channels or ('similar', 'popular', 'same_artist',
                        'related_artist')
    lists = {}
    if 'similar' in want:
        lists['similar'] = recall_similar(index, seed_rec, top_k=top_k)
    if 'popular' in want:
        lists['popular'] = recall_popular(index, seed_rec, top_k=top_k)
    if 'same_artist' in want:
        lists['same_artist'] = recall_same_artist(index, seed_rec,
                                                  top_k=top_k)
    if 'related_artist' in want:
        lists['related_artist'] = recall_related_artist(index, seed_rec,
                                                        top_k=top_k)
    if 'related_track' in want:
        lists['related_track'] = recall_related_track_stub(
            index, seed_rec, top_k=top_k)
    merged = _merge_scores(lists)
    by_key = {d['key']: d for d in merged}
    # Acoustic cosine to the seed, used ONLY as a deterministic tie-break
    # for items outside every channel (merged score 0.0). Without it the
    # ~180 unscored items would rank in arbitrary title order and the
    # mean-rank legs would be tie-break artifacts, not measurements.
    se = _seed_emb(index, seed_rec)
    cosines = index['embeddings'] @ se
    ranked = []
    for pos_i, x in enumerate(items):
        k = cand_key(x['rec'])
        m = by_key.get(k)
        x = dict(x)
        x['port'] = float(m['score_raw']) if m else 0.0
        x['port_sources'] = list(m['sources']) if m else []
        x['port_cosine'] = float(cosines[pos_i])
        ranked.append(x)
    # rank by merged score desc, cosine tie-break, then title/artist
    ranked.sort(key=lambda x: (-x['port'], -x['port_cosine'],
                               x['rec'].get('title', ''),
                               x['rec'].get('artist', '')))
    return ranked


def evaluate_port(write=True):
    """Full section-(d) evaluation. Returns the results dict and, when
    write=True, writes data/evaluation_v3_embeat_replay.json."""
    from evaluate import rank_metrics
    t_start = time.perf_counter()
    items, baselines = build_eval_items()
    seed_rec = eval_seed()
    ranked = rank_eval_pool(seed_rec['spotify_id'])
    port = rank_metrics(ranked, 'port', higher_better=True)
    port['neg_top10'] = sum(
        1 for x in ranked[:10] if x['kind'] == 'negative')
    # six-way ablation on the identical pool
    ablation = {}
    for name, ch in [
            ('full_port', ('similar', 'popular', 'same_artist',
                           'related_artist')),
            ('minus_channel1', ('popular', 'same_artist',
                                'related_artist')),
            ('minus_channel2', ('similar', 'same_artist',
                                'related_artist')),
            ('minus_channel3', ('similar', 'popular', 'related_artist')),
            ('minus_channel4', ('similar', 'popular', 'same_artist')),
            ('channel1_only', ('similar',))]:
        r = rank_eval_pool(seed_rec['spotify_id'], channels=ch)
        m = rank_metrics(r, 'port', higher_better=True)
        m['neg_top10'] = sum(1 for x in r[:10] if x['kind'] == 'negative')
        ablation[name] = m
    # mean over the 20 fixed eval seeds (secondary number)
    per_seed = []
    for sid in EVAL_SEEDS:
        r = rank_eval_pool(sid)
        m = rank_metrics(r, 'port', higher_better=True)
        m['neg_top10'] = sum(1 for x in r[:10] if x['kind'] == 'negative')
        per_seed.append(m)
    mean20 = {
        'P@10': round(float(np.mean([m['P@10'] for m in per_seed])), 3),
        'NDCG@10': round(float(np.mean([m['NDCG@10'] for m in per_seed])), 3),
        'mean_rank_heldout_like': round(
            float(np.mean([m['mean_rank_heldout_like']
                           for m in per_seed])), 1),
        'mean_rank_negative': round(
            float(np.mean([m['mean_rank_negative'] for m in per_seed])), 1),
        'mean_neg_top10': round(
            float(np.mean([m['neg_top10'] for m in per_seed])), 2),
        'n_seeds': len(per_seed),
    }
    # per-channel contribution + diversity + timing over the 20 seeds
    contrib = {c: 0 for c in ('similar', 'popular', 'same_artist',
                              'related_artist', 'related_track')}
    distinct_artists, same_artist_max = [], 0
    times = []
    for sid in EVAL_SEEDS:
        t0 = time.perf_counter()
        rec = recommend(sid, top_k=10, random_seed=RANDOM_SEED)
        times.append(time.perf_counter() - t0)
        for c in contrib:
            if any(c in r.get('sources', []) for r in rec):
                contrib[c] += 1
        arts = [normalize_artist(r.get('artist') or '') for r in rec]
        distinct_artists.append(len(set(arts)))
        seed_a = normalize_artist(
            {t['spotify_id']: t for t in load_liked()}[sid].get('artist')
            or '')
        same_artist_max = max(
            same_artist_max, sum(1 for a in arts if a == seed_a))
    times.sort()
    contribution = {c: {'n_seeds': contrib[c], 'of_20': EVAL_SEEDS.__len__(),
                        'rate': round(contrib[c] / len(EVAL_SEEDS), 2)}
                    for c in contrib}
    diversity = {
        'median_distinct_artists': int(np.median(distinct_artists)),
        'min_distinct_artists': int(min(distinct_artists)),
        'max_same_artist_in_top10': same_artist_max,
    }
    # RSS: /usr/bin/time -v on a fresh 10-seed run (spec T-15 method).
    # /usr/bin/time does not exist on this VM, so fall back to a fresh
    # subprocess reporting its own ru_maxrss (same quantity, same scope).
    rss_mb, rss_note = None, ''
    code = ('import sys, resource; sys.path.insert(0, %r); '
            'import embeat_port as E; '
            '[E.recommend(s, top_k=10, random_seed=616) for s in %r]; '
            'print("RSS-MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0)'
            % (BASE, EVAL_SEEDS[:10]))
    try:
        r = subprocess.run(['/usr/bin/time', '-v', sys.executable, '-c', code],
                           capture_output=True, text=True, timeout=600)
        for line in (r.stderr or '').splitlines():
            if 'Maximum resident set size' in line:
                rss_mb = round(int(line.split(':')[1].strip()) / 1024.0, 1)
    except FileNotFoundError:
        rss_note = '/usr/bin/time not present on this VM; '
    except Exception as e:
        rss_note = f'/usr/bin/time -v run failed: {e}; '
    if rss_mb is None:
        try:
            r = subprocess.run([sys.executable, '-c', code],
                               capture_output=True, text=True, timeout=600)
            for line in (r.stdout or '').splitlines():
                if line.startswith('RSS-MB'):
                    rss_mb = round(float(line.split()[1]), 1)
            rss_note += 'fresh-subprocess ru_maxrss'
        except Exception as e:
            rss_mb = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
            rss_note += f'subprocess RSS failed ({e}); in-process ru_maxrss'
    git_hash = None
    try:
        g = subprocess.run(['git', '-C', BASE, 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, timeout=30)
        git_hash = g.stdout.strip() or None
    except Exception:
        git_hash = None
    index = get_index()
    results = {
        'method': 'embeat_port (v3, architecture-only port; channel 5 stub)',
        'random_seed': RANDOM_SEED,
        'git_hash': git_hash,
        'git_note': 'local-only work; branch pipeline-v3 @ b7f3ef0a per '
                    'brief (no git repo in this workspace dir)' if not git_hash
                    else None,
        'pool': {
            'n_ranked': 311,
            'test_likes_with_librosa': 99,
            'test_likes_total': 129,
            'negatives_with_librosa': 12,
            'negatives_total': 13,
            'negatives_excluded_no_librosa': ['6aJn7Cst74cj4lNIiPRgav'],
            'distractors': 200,
            'pool_total': 590,
            'note': 'Identical 311-item pool to evaluation_v2.json; '
                    'baselines recomputed here match it exactly.',
        },
        'baselines': baselines,
        'eval_seed': {
            'spotify_id': seed_rec['spotify_id'],
            'title': seed_rec.get('title'),
            'artist': seed_rec.get('artist'),
            'genre_bucket': seed_rec.get('genre_bucket'),
            'note': 'medoid train like (position>=130) among top-30 '
                    'artists; keeps channel 4 live in the eval ranking',
        },
        'ported_medoid': port,
        'ported_mean_20seeds': mean20,
        'ablation': ablation,
        'contribution_rates_20seeds': contribution,
        'diversity_20seeds': diversity,
        'runtime': {
            'median_recommend_s': round(times[len(times) // 2], 4),
            'max_recommend_s': round(max(times), 4),
            'n_seeds_timed': len(times),
            'peak_rss_mb_10seed_run': rss_mb,
            'rss_note': rss_note,
            'eval_wall_s': round(time.perf_counter() - t_start, 1),
        },
        'pool_bucket_assignment': index['bucket_counts'],
        'notes': [
            'Channel 2 popularity proxy is the DEGRADED v2-score order '
            'within bucket (Deezer rank not stored on disk).',
            'Pool buckets: artist map 112/590 direct, via_artist fallback '
            '478/590, unknown 0/590.',
            'O-bucket tracks are excluded from genre-filtered channel 1 '
            '(treated as unknown-genre per spec b.5); reachable via '
            'channels 2-4 and merge back-fill.',
            'NDCG@10 uses the eligible pool ideal. Frozen v2/v3 files used '
            'the conditional statistic now named legacy_topk_hit_ndcg.',
            'Zero-score tail: items outside every channel score 0.0 '
            '(~180/311); they are tie-broken by acoustic cosine to the '
            'seed (a real signal), not title order, so mean-rank legs are '
            'measurements rather than tie-break artifacts. P@10/NDCG@10 '
            'are unaffected (top 10 all score > 0).',
            'Channel-4 conflict: contribution rate 8/20 (40%) is below '
            'the d.2.3 50% cut line, but removing channel 4 costs 0.10 '
            'P@10 (ablation minus_channel4), so per the b.4 ablation rule '
            'it is NOT redundant. Channel 4 is KEPT; the conflict is '
            'flagged for the coordinator.',
            'Spec d.2.1 ">=17 additional held-out likes" is wrong: +0.05 '
            'P@10 is +0.5 held-out likes in a top 10.',
            'Spec d.2.2 neg-rank leg is unsatisfiable vs v2 (292.9+20 > '
            '311); evaluated vs best-P@10 baseline v6-global (178.2).',
            'Final-score rescale implemented as linear max->2.0; the '
            'spec-quoted formula is self-contradictory (maps max to '
            '~0.55).',
        ],
    }
    if write:
        from evaluate import write_replay
        write_replay(BASE, 'evaluation_v3_embeat', results)
        log('wrote data/evaluation_v3_embeat_replay.json')
    return results


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'recommend':
        for r in recommend(sys.argv[2], top_k=10, random_seed=RANDOM_SEED):
            print(f"  {r['score']:.2f} [{','.join(r['sources'])}] "
                  f"{r['title']} - {r['artist']} "
                  f"(bucket {r.get('seed_bucket')}/{r.get('seed_bucket_source')})")
    else:
        res = evaluate_port(write=True)
        print(json.dumps({'ported_medoid': res['ported_medoid'],
                          'ported_mean_20seeds': res['ported_mean_20seeds'],
                          'ablation': {k: {'P@10': v['P@10'],
                                           'NDCG@10': v['NDCG@10']}
                                       for k, v in res['ablation'].items()},
                          'contribution': res['contribution_rates_20seeds'],
                          'diversity': res['diversity_20seeds'],
                          'runtime': res['runtime']}, indent=1))
