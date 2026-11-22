#!/usr/bin/env python3
"""Score recommendation candidates against Kevin's liked-track audio profile.

Uses the Deezer preview + librosa analysis from pipeline.py. Each candidate
is matched on Deezer, its 30s preview is analyzed (BPM, key, brightness,
energy, harshness), and it is scored by distance from the per-bucket
(E/R/H) profile medians computed from analysis.json. Lower score = closer fit.

Excludes anything already in Liked Songs (~/workspace/spotify-liked-songs.txt)
or previously recommended (~/workspace/spotify-recommended-tracks.txt).

Run with the audio venv: ~/workspace/audio-env/bin/python recommend.py
"""
import sys, json, os, re, time
sys.path.insert(0, '/home/hatch/workspace/audio-pipeline')
import pipeline as P

# Per-bucket profile from analysis.json joined with classification.json (Sep 12, 2026)
PROFILE = {
 'E': {'bpm_med': 123.0, 'bpm_p25': 112.3, 'bpm_p75': 136.0, 'energy': 0.320, 'bright': 2668},
 'R': {'bpm_med': 123.0, 'bpm_p25': 103.4, 'bpm_p75': 161.5, 'energy': 0.322, 'bright': 2655},
 'H': {'bpm_med': 136.0, 'bpm_p25': 117.5, 'bpm_p75': 161.5, 'energy': 0.341, 'bright': 2897},
}

CANDIDATES = [
 # (title, artist, bucket)
 ("Debut","KATSEYE",'E'), ("Contingency Song","Jane Remover",'E'),
 ("fantasy","ericdoa",'E'), ("minnesota is a place that exists","glaive",'E'),
 ("crush","2hollis",'E'), ("How Sweet","NewJeans",'E'), ("Taka","RL Grime",'E'),
 ("Stuck in My Head","Sleep Theory",'R'), ("Jaded","Spiritbox",'R'),
 ("Stitch","Wage War",'R'), ("Just Pretend","Bad Omens",'R'),
 ("New Way Out","Poppy",'R'), ("Savages","Catch Your Breath",'R'),
 ("Hula Hoops","DPR LIVE",'H'), ("Alone","Night Lovell",'H'),
 ("Cha Cha","Freddie Dredd",'H'), ("Akina Speed Star","Xavier Wulf",'H'),
 ("Prove It","21 Savage",'H'), ("Murder In My Mind","Kordhell",'H'),
 ("Paris","$uicideboy$",'H'),
]

def clean(s):
    s = (s or '').lower()
    s = re.sub(r'\(feat[^)]*\)', '', s)
    s = re.sub(r'\[.*?\]', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

liked = set()
for line in open('/home/hatch/workspace/spotify-liked-songs.txt'):
    p = line.rstrip('\n').split('\t')
    if len(p) >= 2: liked.add((clean(p[0]), clean(p[1].split(',')[0])))
hist = set()
for line in open('/home/hatch/workspace/spotify-recommended-tracks.txt'):
    if line.startswith('#'): continue
    p = line.rstrip('\n').split('\t')
    if len(p) >= 2: hist.add((clean(p[0]), clean(p[1].split(',')[0])))

def score(feats, bucket):
    pr = PROFILE[bucket]
    iqr = max(pr['bpm_p75'] - pr['bpm_p25'], 1)
    bpm_z = abs(feats['bpm'] - pr['bpm_med']) / iqr
    e_z = abs(feats['energy_rms'] - pr['energy']) / 0.10
    b_z = abs(feats['brightness_hz'] - pr['bright']) / 500.0
    key_bonus = -0.5 if 'minor' in feats['key'] else 0.0
    return round(bpm_z + e_z + b_z + key_bonus, 2)

results = []
for title, artist, bucket in CANDIDATES:
    key = (clean(title), clean(artist))
    if key in liked:
        print(f"SKIP liked: {title} - {artist}"); continue
    if key in hist:
        print(f"SKIP already recommended: {title} - {artist}"); continue
    cands = P.deezer_search(title, artist)
    m = P.pick(cands, title, 0) if cands else None
    # pick() needs duration; retry with a wide duration by using first candidate title match
    if not m and cands:
        ct = clean(title)
        for c in cands:
            if clean(c.get('title','')) == ct or ct in clean(c.get('title','')):
                m = c; break
    if not m or not m.get('preview'):
        print(f"NO DEEZER MATCH: {title} - {artist}"); continue
    did = m['id']
    dest = os.path.join(P.PREV, f'{did}.mp3')
    if not os.path.exists(dest):
        try:
            import urllib.request
            urllib.request.urlretrieve(m['preview'], dest)
        except Exception as e:
            print(f"DOWNLOAD FAIL: {title} - {artist}: {e}"); continue
    try:
        feats = P.analyze(dest)
    except Exception as e:
        print(f"ANALYSIS FAIL: {title} - {artist}: {e}"); continue
    if not feats:
        print(f"TOO SHORT: {title} - {artist}"); continue
    s = score(feats, bucket)
    results.append({'title': title, 'artist': artist, 'bucket': bucket,
                    'deezer_title': m.get('title'), 'score': s, **feats})
    print(f"[{bucket}] {title} - {artist}: {feats['bpm']} BPM, {feats['key']}, "
          f"energy {feats['energy_rms']}, bright {feats['brightness_hz']} -> score {s}")
    time.sleep(0.2)

json.dump(results, open('/tmp/candidate_scores.json','w'), indent=1)
print("\nRANKED BY BUCKET:")
for b in 'ERH':
    rs = sorted([r for r in results if r['bucket']==b], key=lambda r: r['score'])
    for r in rs:
        print(f"  [{b}] {r['score']:.2f}  {r['title']} - {r['artist']}  ({r['bpm']} BPM, {r['key']})")
