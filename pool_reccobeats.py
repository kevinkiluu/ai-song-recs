#!/usr/bin/env python3
"""Fill ReccoBeats features for pool tracks by ISRC.

Why this exists: the Deezer /artist/{id}/top endpoint does not return
ISRCs, so build-pool produced 590 pool tracks with reccobeats=0. The
/track/{id} endpoint DOES return an ISRC, so this pass fetches it per
pool track (cached in data/pool_isrc.json), then batch-fetches
ReccoBeats by ISRC and updates data/pool_features.json in place.

Without this, every pool track's ReccoBeats columns are median-imputed
and the learned LR weights on rb_acousticness / rb_energy contribute
nothing to candidate ranking. With it, candidates get the same feature
treatment as liked tracks.

Run: ~/workspace/audio-env/bin/python pool_reccobeats.py
"""
import json, os, sys, time, urllib.request

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
sys.path.insert(0, BASE)
import features_v2 as F


def deezer_track(did, attempts=4):
    url = f'https://api.deezer.com/track/{did}'
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'muse-pipeline/2.0'})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 ** (i + 1))
    print(f'  track {did} failed: {last}')
    return None


def main():
    pool_path = os.path.join(DATA, 'pool_features.json')
    pool = json.load(open(pool_path))
    isrc_cache = F.load_json(os.path.join(DATA, 'pool_isrc.json'), {})
    todo = [p for p in pool if not p.get('isrc')]
    print(f'pool={len(pool)} tracks missing ISRC: {len(todo)} '
          f'({len(isrc_cache)} cached)')
    for n, p in enumerate(todo):
        did = str(p['deezer_id'])
        if did in isrc_cache:
            p['isrc'] = isrc_cache[did]
            continue
        d = deezer_track(p['deezer_id'])
        if d and d.get('isrc'):
            p['isrc'] = d['isrc']
            isrc_cache[did] = d['isrc']
        if (n + 1) % 50 == 0:
            print(f'  isrc {n + 1}/{len(todo)}')
            F.save_json(os.path.join(DATA, 'pool_isrc.json'), isrc_cache)
        time.sleep(0.15)
    F.save_json(os.path.join(DATA, 'pool_isrc.json'), isrc_cache)
    have = sum(1 for p in pool if p.get('isrc'))
    print(f'ISRCs now present: {have}/{len(pool)}')

    rb_cache = F.load_json(os.path.join(DATA, 'reccobeats_pool.json'), {})
    isrcs = [p['isrc'] for p in pool if p.get('isrc') and p['isrc'] not in rb_cache]
    print(f'ReccoBeats by ISRC: {len(rb_cache)} cached, {len(isrcs)} to fetch')
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
            print(f'  rb batch {n // 20 + 1}: {len(d.get("content", []))}/{len(batch)}')
        except Exception as e:
            print(f'  rb batch {n // 20 + 1} FAILED: {e}')
        F.save_json(os.path.join(DATA, 'reccobeats_pool.json'), rb_cache)
        time.sleep(0.3)
    for p in pool:
        if p.get('isrc') and rb_cache.get(p['isrc']):
            p['reccobeats'] = rb_cache[p['isrc']]
    F.save_json(pool_path, pool)
    n_rb = sum(1 for p in pool if p.get('reccobeats'))
    n_lb = sum(1 for p in pool if p.get('librosa'))
    print(f'pool features updated: {len(pool)} tracks, librosa={n_lb}, '
          f'reccobeats={n_rb}')


if __name__ == '__main__':
    main()
