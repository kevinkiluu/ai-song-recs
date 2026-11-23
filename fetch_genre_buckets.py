#!/usr/bin/env python3
"""Re-fetch the genre-playlist membership (E/R/H/O) from Spotify and
save it to data/genre_buckets.json (track ID -> bucket).

The mapping used to live in /tmp/genre_playlists.json, which a runtime
restart wiped; features_v2.py then silently rebuilt features_v2.json
with every genre_bucket=None. The mapping now lives in the workspace
data dir so it survives restarts, and features_v2.py fails loudly if
it is missing.

Playlists (Kevin's genre playlists):
  E spotify:playlist:0jxb8R4WEj7vjcUqBQnJR4 (electronic/EDM)
  R spotify:playlist:3rVMZMYGS1PfXWZfdALy04 (Alt Rock/Rock/Metal)
  H spotify:playlist:5v70YEYQBvLVe0YTPRAzSc (hip hop/trap/rap)
  O spotify:playlist:3YslYmllQeOKnTUUmGgMSt (Other)

Run: ~/workspace/audio-env/bin/python fetch_genre_buckets.py
"""
import json, os, subprocess, sys

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
PLAYLISTS = {
    'E': 'spotify:playlist:0jxb8R4WEj7vjcUqBQnJR4',
    'R': 'spotify:playlist:3rVMZMYGS1PfXWZfdALy04',
    'H': 'spotify:playlist:5v70YEYQBvLVe0YTPRAzSc',
    'O': 'spotify:playlist:3YslYmllQeOKnTUUmGgMSt',
}


def run_cli(args):
    r = subprocess.run(['/opt/hatch/bin/spotify-api'] + args,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f'spotify-api {args} failed: {r.stderr[:300]}')
    return json.loads(r.stdout)


def collect_ids(uri):
    ids = {}
    visited = set()
    d = run_cli(['experience', '--id', uri])
    while True:
        # Two response shapes: experience returns sections[].items,
        # next-page returns top-level items.
        items = list(d.get('items', []))
        next_url = d.get('next')
        for s in d.get('sections', []):
            items.extend(s.get('items', []))
            if s.get('next'):
                next_url = s['next']
        for it in items:
            u = it.get('spotify_uri', '')
            if u.startswith('spotify:track:'):
                tid = u.replace('spotify:track:', '')
                ids[tid] = None
        if not next_url:
            return list(ids)
        if next_url in visited:
            raise RuntimeError('Spotify genre playlist pagination repeated a cursor')
        visited.add(next_url)
        d = run_cli(['next-page', '--url', next_url])


def main():
    mapping = {}
    for bucket, uri in PLAYLISTS.items():
        ids = collect_ids(uri)
        print(f'{bucket}: {len(ids)} tracks')
        for tid in ids:
            mapping.setdefault(tid, []).append(bucket)
    # a track in two playlists keeps both labels; the features file
    # records the first (E/R/H/O priority is arbitrary, noted in the file)
    out = {tid: buckets[0] for tid, buckets in mapping.items()}
    multi = {tid: b for tid, b in mapping.items() if len(b) > 1}
    print(f'total mapped: {len(out)} tracks, {len(multi)} in >1 playlist')
    json.dump(out, open(os.path.join(DATA, 'genre_buckets.json'), 'w'))
    json.dump(multi, open(os.path.join(DATA, 'genre_buckets_multi.json'), 'w'))
    print('wrote data/genre_buckets.json')


if __name__ == '__main__':
    main()
