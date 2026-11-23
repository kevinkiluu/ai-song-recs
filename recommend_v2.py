#!/usr/bin/env python3
"""Candidate generation + recommendations v2.

Two modes:
  build-pool : build a candidate pool from Deezer related artists of
               Kevin's most-liked artists, extract deep librosa features
               + ReccoBeats (by ISRC) for every pool track, save
               data/pool_features.json. No ranking happens here.
  recommend  : score the pool with the v2 model, MMR-diversify, verify
               the top 100 by Spotify track ID and recording key (cleaned
               title + first artist) against Liked Songs and the
               recommendation history, rescore, and pick the final 10.
               Writes recommendations_v2.json. Never re-recommends.

Pool dedup is by Deezer ID + (title, artist) strings; finalists are
additionally verified by Spotify track ID and recording key, which is
the dedup rule that governs the final 10.

Run:
  ~/workspace/audio-env/bin/python recommend_v2.py build-pool
  ~/workspace/audio-env/bin/python recommend_v2.py recommend
"""
import json, os, subprocess, sys, time, urllib.parse, urllib.request
from collections import Counter

import numpy as np

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
PREV = os.path.join(BASE, 'previews')
sys.path.insert(0, BASE)
import features_v2 as F
import pipeline as P
from scoring import V2Model, feat_vector, v6_score

LIKED_FILE = os.path.expanduser('~/workspace/spotify-liked-songs.txt')
HIST_FILE = os.path.expanduser('~/workspace/spotify-recommended-tracks.txt')
TOP_ARTISTS = 30
RELATED_PER_ARTIST = 8
TOP_PER_RELATED = 4
POOL_CAP = 800
MMR_LAMBDA = 0.7
# How many of the top-scored pool tracks MMR-selects for the verify step.
# Was 30; v7/v8/v9 consumed exactly that window, so the verify loop could
# leave 0 clean picks (Nov 22, 2026). 100 keeps the same ranking and
# just looks further down the same list.
VERIFY_WINDOW = 100


def log(msg):
    print(msg, flush=True)


def deezer_get(path, attempts=4):
    url = 'https://api.deezer.com' + path
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'muse-pipeline/2.0'})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 ** (i + 1))
    log(f'  deezer GET failed {path}: {last}')
    return None


def clean(s):
    return P.clean(s)


def load_id_sets():
    liked_ids, liked_keys = set(), set()
    for line in open(LIKED_FILE):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3:
            liked_ids.add(p[2].replace('spotify:track:', ''))
            liked_keys.add((clean(p[0]), clean(p[1].split(',')[0])))
    hist_ids, hist_keys = set(), set()
    for line in open(HIST_FILE):
        if line.startswith('#'):
            continue
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3:
            hist_ids.add(p[2].replace('spotify:track:', ''))
            hist_keys.add((clean(p[0]), clean(p[1].split(',')[0])))
    return liked_ids, liked_keys, hist_ids, hist_keys


def build_pool():
    liked_ids, liked_keys, hist_ids, hist_keys = load_id_sets()
    artists = Counter()
    for line in open(LIKED_FILE):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 2:
            artists[(p[1].split(',')[0].strip())] += 1
    top = [a for a, _ in artists.most_common(TOP_ARTISTS)]
    log(f'top {len(top)} artists by liked count')
    pool = {}  # deezer_id -> record
    for ai, artist in enumerate(top, 1):
        # artist search via Deezer search/artist endpoint
        d = deezer_get('/search/artist?' + urllib.parse.urlencode({'q': artist, 'limit': 3}))
        if not d or not d.get('data'):
            log(f'  no deezer artist for {artist}')
            continue
        # best match: exact name first
        aid = None
        for a in d['data']:
            if clean(a.get('name', '')) == clean(artist):
                aid = a['id']
                break
        if aid is None:
            aid = d['data'][0]['id']
        rel = deezer_get(f'/artist/{aid}/related?limit={RELATED_PER_ARTIST}')
        if not rel or not rel.get('data'):
            continue
        for r in rel['data']:
            top_tracks = deezer_get(f"/artist/{r['id']}/top?limit={TOP_PER_RELATED}")
            if not top_tracks:
                continue
            for t in top_tracks.get('data', []):
                did = t.get('id')
                if not did or did in pool:
                    continue
                title = t.get('title', '')
                tart = (t.get('artist', {}) or {}).get('name', '')
                key = (clean(title), clean(tart))
                if key in liked_keys or key in hist_keys:
                    continue
                pool[did] = {'deezer_id': did, 'title': title, 'artist': tart,
                             'isrc': t.get('isrc'),
                             'via_artist': artist,
                             'related_artist': r.get('name')}
        log(f'  [{ai}/{len(top)}] {artist}: pool={len(pool)}')
        if len(pool) >= POOL_CAP:
            break
        time.sleep(0.3)
    log(f'pool built: {len(pool)} tracks')
    recs = list(pool.values())
    # ReccoBeats by ISRC (batch 20)
    rb_cache = F.load_json(os.path.join(DATA, 'reccobeats_pool.json'), {})
    isrcs = [r['isrc'] for r in recs if r.get('isrc') and r['isrc'] not in rb_cache]
    log(f'ReccoBeats by ISRC: {len(rb_cache)} cached, {len(isrcs)} to fetch')
    for n in range(0, len(isrcs), 20):
        batch = isrcs[n:n + 20]
        url = 'https://api.reccobeats.com/v1/audio-features?ids=' + ','.join(batch)
        req = urllib.request.Request(url, headers={'Accept': 'application/json',
                                                   'User-Agent': 'muse-pipeline/2.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            for item in d.get('content', []):
                if item.get('isrc'):
                    rb_cache[item['isrc']] = item
            log(f'  rb batch {n // 20 + 1}: {len(d.get("content", []))}/{len(batch)}')
        except Exception as e:
            log(f'  rb batch {n // 20 + 1} FAILED: {e}')
        F.save_json(os.path.join(DATA, 'reccobeats_pool.json'), rb_cache)
        time.sleep(0.3)
    # previews + deep librosa (shared cache with liked tracks)
    for r in recs:
        dest = os.path.join(PREV, f"{r['deezer_id']}.mp3")
        if not os.path.exists(dest):
            d = deezer_get(f"/track/{r['deezer_id']}")
            prev = (d or {}).get('preview')
            if prev:
                F.download_preview(prev, dest)
    deep = F.deep_features({str(r['deezer_id']): {'deezer_id': r['deezer_id']}
                            for r in recs})
    out = []
    for r in recs:
        feats = deep.get(str(r['deezer_id']))
        if isinstance(feats, dict) and 'error' in feats:
            feats = None
        out.append({'spotify_id': None, 'title': r['title'], 'artist': r['artist'],
                    'position': None, 'added_at': None, 'genre_bucket': None,
                    'audio_source': 'deezer_preview_30s' if feats else 'none',
                    'deezer_id': r['deezer_id'], 'isrc': r.get('isrc'),
                    'via_artist': r.get('via_artist'),
                    'related_artist': r.get('related_artist'),
                    'reccobeats': rb_cache.get(r['isrc']) if r.get('isrc') else None,
                    'librosa': feats})
    F.save_json(os.path.join(DATA, 'pool_features.json'), out)
    n_lb = sum(1 for x in out if x['librosa'])
    n_rb = sum(1 for x in out if x['reccobeats'])
    log(f'pool features: {len(out)} tracks, librosa={n_lb}, reccobeats={n_rb}')
    return out


def mmr_select(cands, feats_std, scores, k, lam=MMR_LAMBDA):
    """Maximal marginal relevance over standardized feature vectors."""
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(feats_std)
    selected, remaining = [], list(range(len(cands)))
    while remaining and len(selected) < k:
        best, best_val = None, -1e18
        for i in remaining:
            div = max((sim[i][j] for j in selected), default=0.0)
            val = lam * scores[i] - (1 - lam) * div
            if val > best_val:
                best_val, best = val, i
        selected.append(best)
        remaining.remove(best)
    return selected


def pick_search_hit(title, artist, response):
    """Pick the best Spotify track ID from a search response.

    Prefers an exact title match (cleaned, case-insensitive) over document
    order; among exact matches only one whose first artist agrees is
    returned. When no exact title match exists, falls back to the first
    hit. When exact title matches exist but none agrees on artist, returns
    None instead of attaching a different artist's same-title track (live
    case: "drift (feat. Lames)" returns only other artists' "Drift"
    recordings). The old code returned the first track URI in document
    order, which picked "af1" for a "drift" query in the v9 run.
    Title/artist comparison uses clean(), the same normalisation as the
    dedup keys in load_id_sets()."""
    hits = []

    def walk(o):
        if isinstance(o, dict):
            uri = o.get('spotify_uri', '')
            if isinstance(uri, str) and uri.startswith('spotify:track:'):
                hits.append(o)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(response)
    if not hits:
        return None
    want_title = clean(title)
    want_artist = clean((artist or '').split(',')[0])
    exact = [h for h in hits if clean(h.get('title', '')) == want_title]
    if exact:
        for h in exact:
            if clean((h.get('subtitle', '') or '').split(',')[0]) == want_artist:
                return h['spotify_uri'].replace('spotify:track:', '')
        return None
    return hits[0]['spotify_uri'].replace('spotify:track:', '')


def spotify_search_id(title, artist):
    r = subprocess.run(['/opt/hatch/bin/spotify-api', 'search', '--query', title,
                        '--artist', artist, '--search-type', 'TRACKS'],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    try:
        d = json.loads(r.stdout)
    except Exception:
        return None
    return pick_search_hit(title, artist, d)


def is_already_known(sid, title, artist, liked_ids, liked_keys, hist_ids, hist_keys):
    """True if this candidate duplicates a liked or previously recommended
    recording, by Spotify ID or by recording key (cleaned title + first
    artist). The key check catches the same recording under a different
    Spotify ID, e.g. a re-release or an edited/explicit variant: v9
    re-recommended "af1 (sped up)" and "A Milli" under new IDs because
    only the ID was checked."""
    if sid in liked_ids or sid in hist_ids:
        return True
    key = (clean(title), clean((artist or '').split(',')[0]))
    return key in liked_keys or key in hist_keys


def verify_candidates(cands, search_id, liked_ids, liked_keys, hist_ids, hist_keys):
    """Resolve each candidate to a Spotify ID and drop duplicates.
    search_id(title, artist) -> Spotify track ID or None."""
    verified = []
    for c in cands:
        sid = search_id(c['title'], c['artist'])
        time.sleep(0.2)
        if not sid:
            log(f"  no spotify match, dropping: {c['title']} - {c['artist']}")
            continue
        if is_already_known(sid, c['title'], c['artist'],
                            liked_ids, liked_keys, hist_ids, hist_keys):
            log(f"  already liked/history, dropping: {c['title']} ({sid})")
            continue
        c['spotify_id'] = sid
        verified.append(c)
    return verified


def recommend():
    liked_ids, liked_keys, hist_ids, hist_keys = load_id_sets()
    pool = json.load(open(os.path.join(DATA, 'pool_features.json')))
    pool = [p for p in pool if p.get('librosa')]
    log(f'pool with librosa: {len(pool)}')
    model = V2Model()
    s = model.score_records(pool)
    order = np.argsort(-s['final'])
    # standardized features for MMR
    X, _ = zip(*[feat_vector(p) for p in pool])
    X = np.vstack(X)
    X = np.where(np.isnan(X), model.median, X)
    Z = (X - model.mean) / model.scale
    top100 = order[:100]
    sel100 = mmr_select([pool[i] for i in top100], Z[top100],
                        s['final'][top100], k=VERIFY_WINDOW)
    cands = [pool[top100[i]] for i in sel100]
    log(f'MMR top {VERIFY_WINDOW} selected')
    # verify by Spotify ID and recording key
    verified = verify_candidates(cands, spotify_search_id,
                                 liked_ids, liked_keys, hist_ids, hist_keys)
    log(f'verified by Spotify ID: {len(verified)}')
    if not verified:
        # No clean candidates: report it instead of crashing on the
        # empty rescore below (zip(*) over an empty list raised
        # ValueError in the v10 run, Nov 22, 2026).
        log('no clean candidates')
        return []
    # fill any missing ReccoBeats by Spotify ID
    missing = [c for c in verified if not c.get('reccobeats')]
    if missing:
        rb = F.fetch_reccobeats([c['spotify_id'] for c in missing])
        for c in missing:
            if rb.get(c['spotify_id']):
                c['reccobeats'] = rb[c['spotify_id']]
    # rescore verified set and MMR pick 10
    s2 = model.score_records(verified)
    X2, _ = zip(*[feat_vector(c) for c in verified])
    X2 = np.vstack(X2)
    X2 = np.where(np.isnan(X2), model.median, X2)
    Z2 = (X2 - model.mean) / model.scale
    order2 = np.argsort(-s2['final'])
    top20 = order2[:20]
    sel = mmr_select([verified[i] for i in top20], Z2[top20],
                     s2['final'][top20], k=10)
    final = [verified[top20[i]] for i in sel]
    # artist -> bucket map for v6 buckets
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    artist_bucket = {}
    for t in doc['tracks']:
        a = (t.get('artist') or '').split(',')[0].strip().lower()
        b = t.get('genre_bucket')
        if a and b:
            artist_bucket.setdefault(a, Counter())[b] += 1
    artist_bucket = {a: c.most_common(1)[0][0] for a, c in artist_bucket.items()}
    ver_index = {c['spotify_id']: i for i, c in enumerate(verified)}
    out = []
    for c in final:
        i = ver_index[c['spotify_id']]
        sc = s2['final'][i]
        a = (c.get('artist') or '').split(',')[0].strip().lower()
        bucket = artist_bucket.get(a)
        v6b, used_b = v6_score(c.get('librosa'), bucket)
        v6g, _ = v6_score(c.get('librosa'), None)
        out.append({'title': c['title'], 'artist': c['artist'],
                    'spotify_id': c['spotify_id'], 'deezer_id': c['deezer_id'],
                    'v2_score': round(float(sc), 3),
                    'v6_score_per_bucket': v6b, 'v6_bucket': used_b,
                    'v6_score_global': v6g,
                    'nearest_cluster': int(s2['nearest_cluster'][i]),
                    'audio_source': c.get('audio_source')})
    json.dump(out, open(os.path.join(BASE, 'recommendations_v2.json'), 'w'), indent=1)
    print('\nFINAL 10 (v2 score | v6 per-bucket | v6 global | cluster):')
    for o in out:
        print(f"  {o['v2_score']:+.2f} | {o['v6_score_per_bucket']} ({o['v6_bucket']}) | "
              f"{o['v6_score_global']} | c{o['nearest_cluster']}  "
              f"{o['title']} - {o['artist']}  [{o['spotify_id']}]")
    print('wrote recommendations_v2.json')
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'build-pool':
        build_pool()
    elif mode == 'recommend':
        recommend()
    else:
        print('usage: recommend_v2.py [build-pool|recommend]')
        sys.exit(1)
