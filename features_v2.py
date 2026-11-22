#!/usr/bin/env python3
"""Feature extraction v2 for the Spotify recommendation pipeline.

Replaces the v1 approach (5 scalar means over a 30s Deezer preview) with:
  - ReccoBeats Spotify-style audio features for every liked track
    (danceability, energy, valence, acousticness, instrumentalness,
    speechiness, liveness, loudness, tempo, key, mode). Spotify's own
    audio-features endpoint was removed 2024-11-27; ReccoBeats is the
    approved substitute and needs no API key.
  - Deep librosa features per 30s Deezer preview: MFCCs (mean+std),
    chroma histogram, tonnetz, spectral contrast, rolloff, bandwidth,
    flatness, harmonic/percussive balance, onset strength, and per-segment
    (3 x 10s) brightness/energy deltas as a structure proxy.
  - Genre bucket (E/R/H/O) from Kevin's own genre playlists.

Audio source is recorded per track. No full-length audio exists in the
workspace, so every track is preview-only; that limit is recorded here
and in the evaluation report, not silently approximated.

Idempotent: Deezer matches, ReccoBeats responses and librosa results are
cached under data/ and reused on reruns.

Run: ~/workspace/audio-env/bin/python features_v2.py
Outputs: data/reccobeats.json, data/deezer_matches.json,
          data/librosa_deep.json, features_v2.json
"""
import json, os, sys, time, urllib.parse, urllib.request

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
PREV = os.path.join(BASE, 'previews')
os.makedirs(DATA, exist_ok=True)
os.makedirs(PREV, exist_ok=True)
LOG = os.path.join(BASE, 'features_v2.log')
sys.path.insert(0, BASE)
import pipeline as P  # reuse deezer_search / pick / estimate_key

LIKED_FILE = os.path.expanduser('~/workspace/spotify-liked-songs.txt')
RECCO = 'https://api.reccobeats.com/v1/audio-features?ids='


def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- liked list
def load_liked():
    tracks = []
    for pos, line in enumerate(open(LIKED_FILE)):
        p = line.rstrip('\n').split('\t')
        if len(p) < 3:
            continue
        tracks.append({'position': pos, 'title': p[0], 'artist': p[1],
                       'spotify_id': p[2].replace('spotify:track:', '')})
    return tracks


def load_buckets():
    """Genre bucket per Spotify ID from Kevin's genre playlists.

    The mapping lives in data/genre_buckets.json (track ID -> bucket),
    written by fetch_genre_buckets.py. It used to live in
    /tmp/genre_playlists.json; a runtime restart wiped /tmp and a
    re-run silently rebuilt features_v2.json with every genre_bucket
    None, which broke the evaluation's genre sanity check. This now
    fails loudly instead of silently producing bucket-less features.
    """
    path = os.path.join(DATA, 'genre_buckets.json')
    if not os.path.exists(path):
        raise SystemExit(
            'GENRE MAPPING MISSING: run fetch_genre_buckets.py first '
            f'(expected {path})')
    buckets = load_json(path, {})
    log(f'genre buckets: {len(buckets)} labeled')
    if len(buckets) < 500:
        raise SystemExit(
            f'GENRE MAPPING SUSPECT: only {len(buckets)} tracks labeled '
            '(expected ~635); re-run fetch_genre_buckets.py')
    return buckets


# ---------------------------------------------------------------- ReccoBeats
def fetch_reccobeats(ids):
    cache = load_json(os.path.join(DATA, 'reccobeats.json'), {})
    missing = [i for i in ids if i not in cache and i != 'None']
    log(f'ReccoBeats: {len(cache)} cached, {len(missing)} to fetch')
    for n in range(0, len(missing), 20):
        batch = missing[n:n + 20]
        url = RECCO + ','.join(batch)
        req = urllib.request.Request(
            url, headers={'Accept': 'application/json',
                          'User-Agent': 'muse-pipeline/2.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            for item in d.get('content', []):
                href = item.get('href', '')
                sid = href.rsplit('/', 1)[-1].split('?')[0] if href else None
                if sid:
                    cache[sid] = item
            log(f'  batch {n // 20 + 1}: {len(d.get("content", []))}/{len(batch)} returned')
        except Exception as e:
            log(f'  batch {n // 20 + 1} FAILED: {e}')
        save_json(os.path.join(DATA, 'reccobeats.json'), cache)
        time.sleep(0.3)
    return cache


# ---------------------------------------------------------------- Deezer
def deezer_search_retry(title, artist, attempts=4):
    """Deezer search with exponential backoff. Deezer rate-limits with
    transient 403s; a failed search must never be recorded as 'no match'."""
    last_err = None
    for i in range(attempts):
        try:
            return P.deezer_search(title, artist)
        except Exception as e:
            last_err = e
            time.sleep(2 ** (i + 1))
    log(f'  deezer search FAILED after {attempts} tries: {title}: {last_err}')
    return None  # None = search failed (do not cache as no-match)


def download_preview(url, dest, attempts=3):
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'muse-pipeline/2.0'})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, 'wb') as f:
                f.write(r.read())
            return True
        except Exception as e:
            if i == attempts - 1:
                log(f'  download fail {dest}: {e}')
                return False
            time.sleep(2 ** (i + 1))
    return False


def match_deezer(tracks, durations):
    cache = load_json(os.path.join(DATA, 'deezer_matches.json'), {})
    done = 0
    for t in tracks:
        sid = t['spotify_id']
        if sid in cache or sid == 'None':
            continue
        title, artist = t['title'], (t.get('artist') or '').split(',')[0].strip()
        dur_s = (durations.get(sid) or 0) / 1000.0
        cands = deezer_search_retry(title, artist)
        if cands is None:
            # transient failure: do NOT cache; a later run retries this track
            continue
        m = P.pick(cands, title, dur_s) if cands and dur_s else None
        if not m and cands:
            ct = P.clean(title)
            for c in cands:
                if P.clean(c.get('title', '')) == ct:
                    m = c
                    break
        if m and m.get('preview'):
            cache[sid] = {'deezer_id': m['id'], 'deezer_title': m.get('title'),
                          'isrc': m.get('isrc'),
                          'duration_delta_s': round(abs((m.get('duration') or 0) - dur_s), 1)}
            dest = os.path.join(PREV, f"{m['id']}.mp3")
            if not os.path.exists(dest):
                if not download_preview(m['preview'], dest):
                    cache[sid]['download_error'] = 'preview download failed'
        else:
            cache[sid] = {'deezer_id': None, 'reason': 'no match' if not m else 'no preview'}
        done += 1
        if done % 25 == 0:
            save_json(os.path.join(DATA, 'deezer_matches.json'), cache)
            log(f'  deezer matched {done} new tracks')
        time.sleep(0.5)
    save_json(os.path.join(DATA, 'deezer_matches.json'), cache)
    return cache


# ---------------------------------------------------------------- librosa deep
def analyze_deep(path):
    """Deep librosa features for one 30s preview.

    Performance note (measured Nov 21, 2026 on this VM): the naive
    version took ~17s/track because every feature recomputed its own
    STFT and chroma_cqt + two HPSS passes dominated. This version
    computes one STFT (sr=16000) and derives everything from it:
    chroma_stft instead of chroma_cqt (0.2s vs 4.2s), tonnetz from
    chroma instead of the harmonic component, and no HPSS pass at all
    (the vocal/instrumental signal is covered by ReccoBeats
    instrumentalness/speechiness). ~3.5-4s/track.
    """
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) < sr * 2:
        return None
    S = np.abs(librosa.stft(y))
    S_db = librosa.power_to_db(S)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    onset = librosa.onset.onset_strength(S=S_db, sr=sr)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    key = P.estimate_key(chroma.mean(axis=1))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    mfcc = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=13)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    # per-segment (3 x 10s) brightness/energy: structure proxy for a 30s clip
    seg = []
    nseg = 3
    seg_len = len(y) // nseg
    for i in range(nseg):
        s = y[i * seg_len:(i + 1) * seg_len]
        if len(s) < sr:
            continue
        c = librosa.feature.spectral_centroid(y=s, sr=sr)[0]
        r = librosa.feature.rms(y=s)[0]
        seg.append({'centroid_mean': round(float(c.mean()), 1),
                    'rms_mean': round(float(r.mean()), 5)})
    chroma_hist = chroma.mean(axis=1)
    chroma_hist = chroma_hist / (chroma_hist.sum() + 1e-9)
    return {
        'bpm': round(float(np.atleast_1d(tempo)[0]), 1),
        'key': key,
        'onset_strength_mean': round(float(onset.mean()), 4),
        'brightness_mean': round(float(cent.mean()), 1),
        'brightness_std': round(float(cent.std()), 1),
        'rolloff_mean': round(float(roll.mean()), 1),
        'bandwidth_mean': round(float(bw.mean()), 1),
        'flatness_mean': round(float(flat.mean()), 4),
        'zcr_mean': round(float(zcr.mean()), 4),
        'zcr_std': round(float(zcr.std()), 4),
        'rms_mean': round(float(rms.mean()), 5),
        'rms_std': round(float(rms.std()), 5),
        'loudness_db': round(float(20 * np.log10(rms.mean() + 1e-9)), 2),
        'contrast_mean': [round(float(v), 2) for v in contrast.mean(axis=1)],
        'mfcc_mean': [round(float(v), 3) for v in mfcc.mean(axis=1)],
        'mfcc_std': [round(float(v), 3) for v in mfcc.std(axis=1)],
        'chroma_hist': [round(float(v), 4) for v in chroma_hist],
        'tonnetz_mean': [round(float(v), 3) for v in tonnetz.mean(axis=1)],
        'hpss_harmonic_ratio': None,  # dropped: HPSS too slow on this VM;
        # vocal/instrumental signal comes from ReccoBeats instead
        'segments': seg,
        'seg_brightness_delta': round(max(s['centroid_mean'] for s in seg) - min(s['centroid_mean'] for s in seg), 1) if len(seg) > 1 else 0.0,
        'seg_energy_delta': round(max(s['rms_mean'] for s in seg) - min(s['rms_mean'] for s in seg), 5) if len(seg) > 1 else 0.0,
    }


def deep_features(matches):
    cache = load_json(os.path.join(DATA, 'librosa_deep.json'), {})
    todo = [(sid, m['deezer_id']) for sid, m in matches.items()
            if m.get('deezer_id') and str(m['deezer_id']) not in cache]
    log(f'librosa deep: {len(cache)} cached, {len(todo)} to analyze')
    for n, (sid, did) in enumerate(todo, 1):
        dest = os.path.join(PREV, f'{did}.mp3')
        if not os.path.exists(dest):
            cache[str(did)] = {'error': 'preview file missing'}
            continue
        try:
            feats = analyze_deep(dest)
        except Exception as e:
            feats = None
            log(f'  analysis fail deezer {did}: {e}')
        cache[str(did)] = feats if feats else {'error': 'analysis failed'}
        if n % 25 == 0:
            log(f'  analyzed {n}/{len(todo)}')
            save_json(os.path.join(DATA, 'librosa_deep.json'), cache)
    save_json(os.path.join(DATA, 'librosa_deep.json'), cache)
    return cache


# ---------------------------------------------------------------- assemble
def main():
    tracks = load_liked()
    log(f'liked tracks: {len(tracks)}')
    buckets = load_buckets()
    durations = load_json('/tmp/liked_durations.json', {})
    ids = [t['spotify_id'] for t in tracks]
    rb = fetch_reccobeats(ids)
    matches = match_deezer(tracks, durations)
    deep = deep_features(matches)
    out_tracks, unmatched = [], []
    n_rb = n_deep = 0
    for t in tracks:
        sid = t['spotify_id']
        m = matches.get(sid, {})
        did = m.get('deezer_id')
        feats = deep.get(str(did)) if did else None
        if isinstance(feats, dict) and 'error' in feats:
            feats = None
        rec = {
            'spotify_id': sid, 'title': t['title'], 'artist': t['artist'],
            'position': t['position'],  # 0 = most recently liked (list order)
            'added_at': None,  # Spotify connector exposes no added_at field
            'added_at_proxy': 'list_position',
            'genre_bucket': buckets.get(sid),
            'audio_source': 'deezer_preview_30s' if did and feats else 'none',
            'deezer_id': did,
            'reccobeats': rb.get(sid),
            'librosa': feats,
        }
        if rb.get(sid):
            n_rb += 1
        if feats:
            n_deep += 1
        if not did:
            unmatched.append({'spotify_id': sid, 'title': t['title'],
                              'reason': m.get('reason', 'no match')})
        out_tracks.append(rec)
    doc = {
        'version': 2,
        'generated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'counts': {'total': len(out_tracks), 'reccobeats': n_rb,
                   'librosa_deep': n_deep, 'unmatched_deezer': len(unmatched)},
        'notes': [
            'added_at is not exposed by the Spotify connector; position in '
            'the Liked Songs list (0 = most recent, verified against the '
            'live library order) is the time proxy.',
            'No full-length audio exists in the workspace, and Kevin '
            'confirmed Nov 21, 2026 he owns no full-length audio files '
            'anywhere (no local music library). All audio analysis is on '
            '30s Deezer previews. Full-track structure, Demucs stems and '
            'per-section analysis are not possible from previews and were '
            'not approximated; the only path to full-track analysis is him '
            'acquiring the files.',
            'ReccoBeats is the Spotify audio-features substitute (Spotify '
            'removed its audio-features endpoint 2024-11-27).',
        ],
        'tracks': out_tracks,
        'unmatched': unmatched,
    }
    save_json(os.path.join(BASE, 'features_v2.json'), doc)
    log(f"DONE: total={len(out_tracks)} reccobeats={n_rb} librosa={n_deep} unmatched={len(unmatched)}")


if __name__ == '__main__':
    main()
