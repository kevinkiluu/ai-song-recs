#!/usr/bin/env python3
"""GTZAN genre sanity check for the v2 feature extractor.

Extracts the same deep librosa features used for Kevin's tracks from a
subset of GTZAN (10 genres x N clips), trains a multinomial logistic
regression on a train split, and reports test accuracy. This validates
that the feature pipeline distinguishes musical genres on an external
benchmark; it does not validate taste.

Requires /tmp/gtzan.tar.gz (1.2GB download from HuggingFace
marsyas/gtzan). If the download never completes, this check is skipped
and the genre-bucket sanity check in evaluate.py is the fallback.

Run: ~/workspace/audio-env/bin/python gtzan_check.py [clips_per_genre=10]
Output: data/gtzan_check.json
"""
import json, os, subprocess, sys, tarfile

import numpy as np

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
sys.path.insert(0, BASE)
import features_v2 as F
from scoring import feat_vector

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
          'metal', 'pop', 'reggae', 'rock']


def main():
    per_genre = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tar_path = '/home/hatch/workspace/tmp-gtzan/gtzan.tar.gz'
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1_200_000_000:
        print('GTZAN tarball missing or incomplete; skipping GTZAN check')
        json.dump({'status': 'skipped',
                   'reason': 'download incomplete or missing'},
                  open(os.path.join(DATA, 'gtzan_check.json'), 'w'))
        return
    work = '/home/hatch/workspace/tmp-gtzan/extracted'
    os.makedirs(work, exist_ok=True)
    # extract only the needed subset
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers()
                   if m.name.startswith('genres/') and m.name.endswith('.wav')
                   and '/._' not in m.name]
        by_genre = {}
        for m in members:
            g = m.name.split('/')[1]
            by_genre.setdefault(g, []).append(m)
        wanted = []
        for g in GENRES:
            wanted += sorted(by_genre.get(g, []), key=lambda m: m.name)[:per_genre]
        print(f'extracting {len(wanted)} clips')
        tf.extractall(work, members=wanted)
    X, y = [], []
    failed = 0
    for m in wanted:
        g = m.name.split('/')[1]
        path = os.path.join(work, m.name)
        try:
            feats = F.analyze_deep(path)
        except Exception as e:
            print(f'  analysis fail {m.name}: {e}')
            failed += 1
            continue
        if not feats:
            failed += 1
            continue
        rec = {'reccobeats': None, 'librosa': feats}
        v, _ = feat_vector(rec)
        X.append(v)
        y.append(g)
    X = np.vstack(X)
    y = np.array(y)
    print(f'features: {X.shape}, failed: {failed}')
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y))
    n_test = max(1, int(0.3 * len(y)))
    te, tr = idx[:n_test], idx[n_test:]
    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(X[tr])
    Xte = imp.transform(X[te])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(Xtr), y[tr])
    acc = float(clf.score(sc.transform(Xte), y[te]))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y[te], clf.predict(sc.transform(Xte)),
                          labels=GENRES).tolist()
    print(f'GTZAN accuracy (test {len(te)} clips): {acc:.3f}')
    out = {'status': 'done', 'clips_per_genre': per_genre,
           'n_train': int(len(tr)), 'n_test': int(len(te)),
           'accuracy': round(acc, 3), 'genres': GENRES,
           'confusion': cm, 'failed_clips': failed}
    json.dump(out, open(os.path.join(DATA, 'gtzan_check.json'), 'w'), indent=1)
    print('wrote data/gtzan_check.json')


if __name__ == '__main__':
    main()
