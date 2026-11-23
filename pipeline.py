#!/usr/bin/env python3
"""Deezer preview -> local audio analysis pipeline.
Matches Spotify liked tracks to Deezer, downloads 30s previews,
computes BPM, key, brightness, energy, harshness with librosa.
"""
import json, os, sys, time, urllib.parse, urllib.request
import numpy as np

from scoring import clean

BASE = os.path.expanduser('~/workspace/audio-pipeline')
PREV = os.path.join(BASE, 'previews')
os.makedirs(PREV, exist_ok=True)
LOG = os.path.join(BASE, 'pipeline.log')

def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

def deezer_search(title, artist, *, raise_on_error=False):
    q = f"{artist} {title}"
    url = 'https://api.deezer.com/search/track?' + urllib.parse.urlencode({'q': q, 'limit': 5})
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            response = json.load(r)
            if not isinstance(response.get('data'), list):
                raise ValueError(f"invalid Deezer search response: {response.get('error')}")
            return response['data']
    except Exception as e:
        if raise_on_error:
            raise
        log(f'  search error: {e}')
        return []

def pick(cands, title, dur_s):
    ct = clean(title)
    if not ct:
        return None
    for c in cands:
        dt = clean(c.get('title', ''))
        dd = abs((c.get('duration') or 0) - dur_s)
        if dd <= 8 and dt and (ct == dt or ct in dt or dt in ct):
            return c
    return None

MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def estimate_key(chroma_mean):
    best, bk = -2, 'unknown'
    for i in range(12):
        for prof, mode in ((MAJ, 'major'), (MIN, 'minor')):
            corr = np.corrcoef(chroma_mean, np.roll(prof, i))[0, 1]
            if corr > best:
                best, bk = corr, f"{NOTES[i]} {mode}"
    return bk

def analyze(path):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    if len(y) < sr * 2:
        return None
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = estimate_key(chroma.mean(axis=1))
    cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    return {
        'bpm': round(float(np.atleast_1d(tempo)[0]), 1),
        'key': key,
        'brightness_hz': round(cent, 1),
        'energy_rms': round(rms, 5),
        'harshness_zcr': round(float(zcr), 4),
    }

def main():
    tracks = json.load(open('/tmp/recent110.json'))[:100]
    log(f'pipeline start: {len(tracks)} tracks')
    results, unmatched = [], []
    for n, t in enumerate(tracks, 1):
        title, artist = t['title'], (t.get('artist') or '').split(',')[0].strip()
        dur_s = (t.get('duration_ms') or 0) / 1000.0
        log(f'[{n}/{len(tracks)}] {title} - {artist}')
        cands = deezer_search(title, artist)
        m = pick(cands, title, dur_s) if dur_s else None
        if not m or not m.get('preview'):
            unmatched.append({'title': title, 'artist': t.get('artist'), 'reason': 'no match' if not m else 'no preview'})
            log('  -> unmatched')
            time.sleep(0.2)
            continue
        did = m['id']
        dest = os.path.join(PREV, f'{did}.mp3')
        if not os.path.exists(dest):
            try:
                urllib.request.urlretrieve(m['preview'], dest)
            except Exception as e:
                unmatched.append({'title': title, 'artist': t.get('artist'), 'reason': f'download: {e}'})
                log(f'  -> download failed: {e}')
                continue
        try:
            feats = analyze(dest)
        except Exception as e:
            unmatched.append({'title': title, 'artist': t.get('artist'), 'reason': f'analysis: {e}'})
            log(f'  -> analysis failed: {e}')
            continue
        if not feats:
            unmatched.append({'title': title, 'artist': t.get('artist'), 'reason': 'clip too short'})
            continue
        results.append({
            'spotify_title': title, 'spotify_artist': t.get('artist'),
            'deezer_id': did, 'deezer_title': m.get('title'),
            'duration_delta_s': round(abs((m.get('duration') or 0) - dur_s), 1),
            **feats,
        })
        log(f"  -> {feats['bpm']} BPM, {feats['key']}")
        time.sleep(0.2)

    bpms = [r['bpm'] for r in results]
    summary = {
        'tracks_attempted': len(tracks),
        'tracks_analyzed': len(results),
        'tracks_unmatched': len(unmatched),
        'bpm_mean': round(float(np.mean(bpms)), 1) if bpms else None,
        'bpm_median': round(float(np.median(bpms)), 1) if bpms else None,
        'brightness_mean_hz': round(float(np.mean([r['brightness_hz'] for r in results])), 1) if results else None,
        'energy_mean_rms': round(float(np.mean([r['energy_rms'] for r in results])), 5) if results else None,
    }
    json.dump({'summary': summary, 'tracks': results, 'unmatched': unmatched},
              open(os.path.join(BASE, 'analysis.json'), 'w'), indent=1)
    log('SUMMARY: ' + json.dumps(summary))
    log('pipeline done')

if __name__ == '__main__':
    main()
