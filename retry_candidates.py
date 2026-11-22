#!/usr/bin/env python3
"""Retry candidates that failed Deezer lookup in recommend.py (transient 403s).

Same scoring as recommend.py. Run with the audio venv:
~/workspace/audio-env/bin/python retry_candidates.py
"""
import sys, json, os, time
sys.path.insert(0, '/home/hatch/workspace/audio-pipeline')
import pipeline as P

PROFILE = {
 'E': {'bpm_med': 123.0, 'bpm_p25': 112.3, 'bpm_p75': 136.0, 'energy': 0.320, 'bright': 2668},
 'R': {'bpm_med': 123.0, 'bpm_p25': 103.4, 'bpm_p75': 161.5, 'energy': 0.322, 'bright': 2655},
 'H': {'bpm_med': 136.0, 'bpm_p25': 117.5, 'bpm_p75': 161.5, 'energy': 0.341, 'bright': 2897},
}
RETRY = [("fantasy","ericdoa",'E'), ("minnesota is a place that exists","glaive",'E'),
         ("Taka","RL Grime",'E'), ("Stuck in My Head","Sleep Theory",'R'),
         ("Hula Hoops","DPR LIVE",'H')]

def score(f, b):
    pr = PROFILE[b]
    iqr = max(pr['bpm_p75'] - pr['bpm_p25'], 1)
    return round(abs(f['bpm']-pr['bpm_med'])/iqr + abs(f['energy_rms']-pr['energy'])/0.10
                 + abs(f['brightness_hz']-pr['bright'])/500.0
                 + (-0.5 if 'minor' in f['key'] else 0.0), 2)

out = []
for title, artist, bucket in RETRY:
    time.sleep(3)
    cands = P.deezer_search(title, artist)
    m = None
    if cands:
        ct = P.clean(title)
        for c in cands:
            if P.clean(c.get('title','')) == ct or ct in P.clean(c.get('title','')):
                m = c; break
    if not m or not m.get('preview'):
        print(f"STILL NO MATCH: {title} - {artist}"); continue
    dest = os.path.join(P.PREV, f"{m['id']}.mp3")
    if not os.path.exists(dest):
        import urllib.request
        urllib.request.urlretrieve(m['preview'], dest)
    feats = P.analyze(dest)
    if not feats:
        print(f"TOO SHORT: {title} - {artist}"); continue
    s = score(feats, bucket)
    out.append({'title': title, 'artist': artist, 'bucket': bucket,
                'deezer_title': m.get('title'), 'score': s, **feats})
    print(f"[{bucket}] {title} - {artist}: {feats['bpm']} BPM, {feats['key']}, "
          f"energy {feats['energy_rms']}, bright {feats['brightness_hz']} -> score {s}")
json.dump(out, open('/tmp/retry_scores.json','w'), indent=1)
