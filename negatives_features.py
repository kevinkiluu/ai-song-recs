#!/usr/bin/env python3
"""Feature extraction for the confirmed negative tracks.

The negatives are tracks Kevin heard and rejected:
  - 9 v5 playlist tracks he dropped (kept only "Fallout" by Sleep Theory)
  - 4 v6 playlist tracks he thumbs-downed on second listen, all from
    the H (hip-hop/trap) bucket: "prove it" (21 Savage, Summer Walker),
    "Akina Speed Star" (Xavier Wulf), "Alone" (Night Lovell),
    "Cha Cha" (Freddie Dredd). v6 keep rate is 6/10, not 10/10.
Total: 13 confirmed negatives. The 6 kept v6 tracks are NOT negatives.

Each negative gets the same feature record shape as features_v2.json
tracks: ReccoBeats panel + deep librosa features from its Deezer 30s
preview. Output: data/negatives_features.json (used by model_v2.py).

Run: ~/workspace/audio-env/bin/python negatives_features.py
"""
import json, os, sys

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
sys.path.insert(0, BASE)
import features_v2 as F

NEGATIVES = [
    # v5 drops (Nov 21, 2026)
    ('6aJn7Cst74cj4lNIiPRgav', 'Touch', 'KATSEYE'),
    ('6HacgXCExkzS552ILfJTXu', 'Census Designated', 'Jane Remover'),
    ('1GfBLbAhZUWdseuDqhocmn', 'Rumble', 'Skrillex, Fred again.., Flowdan'),
    ('5v7oj9PosKdE39x9uwDs6C', 'dancinwithsomebawdy', 'ericdoa'),
    ('4bkPAARkDov1lV05emEF7a', 'Soft Spine', 'Spiritbox'),
    ('2jv1ovpDpTJEPIlth5Wn07', 'TOMBSTONE', 'Wage War'),
    ('7zDu3l7DDlfSlt20YMKkWo', 'Martini Blue', 'DPR LIVE'),
    ('2y57H6nA3vic5t8awwaS69', 'Dark Light', 'Night Lovell'),
    ('2slLXFcITuhf0oSp2wVJLv', 'Opaul', 'Freddie Dredd'),
    # v6 thumbs-down on second listen (Nov 21, 2026), all H bucket
    ('30efCAopZCFcoi43sSToNr', 'prove it', '21 Savage, Summer Walker'),
    ('0FmrNFYyLLRtIS7XWJPLLk', 'Akina Speed Star', 'Xavier Wulf'),
    ('32gSPav2xqFylAeCjJCxWi', 'Alone', 'Night Lovell'),
    ('1vvBUZseJ7fwYPX1NedOLd', 'Cha Cha', 'Freddie Dredd'),
]


def main():
    tracks = [{'spotify_id': sid, 'title': t, 'artist': a}
              for sid, t, a in NEGATIVES]
    rb = F.fetch_reccobeats([t['spotify_id'] for t in tracks])
    matches = F.match_deezer(tracks, {})
    deep = F.deep_features({sid: m for sid, m in matches.items()
                            if sid in {t['spotify_id'] for t in tracks}})
    out = []
    for t in tracks:
        sid = t['spotify_id']
        m = matches.get(sid, {})
        did = m.get('deezer_id')
        feats = deep.get(str(did)) if did else None
        if isinstance(feats, dict) and 'error' in feats:
            feats = None
        out.append({'spotify_id': sid, 'title': t['title'],
                    'artist': t['artist'], 'position': None,
                    'added_at': None, 'genre_bucket': None,
                    'audio_source': 'deezer_preview_30s' if did and feats else 'none',
                    'deezer_id': did, 'reccobeats': rb.get(sid),
                    'librosa': feats})
        print(f"{t['title']} - {t['artist']}: "
              f"reccobeats={'yes' if rb.get(sid) else 'NO'} "
              f"librosa={'yes' if feats else 'NO'}")
    F.save_json(os.path.join(DATA, 'negatives_features.json'), out)
    print(f"wrote {len(out)} negative records")


if __name__ == '__main__':
    main()
