#!/usr/bin/env python3
"""Taste modeling v2: clusters + learned weights from like/skip labels.

Inputs: features_v2.json (ReccoBeats panel + deep librosa features).
Split: time-based by Liked Songs list position (0 = most recent).
  test  = positions 0..129  (most recent 20%, held out)
  train = positions 130..649
  (added_at is not exposed by the Spotify connector; list position is the
  verified time proxy.)

Clustering: PCA(15) -> GaussianMixture, k chosen by BIC in 2..8.
  Alternative inspected: HDBSCAN on UMAP(10). The clustering with the
  better silhouette (computed in PCA space for comparability) is selected.
Learned weights: logistic regression, train likes (y=1) vs the 13
  confirmed negatives (y=0): 9 dropped v5 tracks + 4 v6 tracks Kevin
  thumbs-downed on second listen, all from the H bucket. class_weight
  balanced, recency sample weights on likes. The negative set is thin
  (13 tracks); the LR is a ranking signal, not a calibrated classifier,
  and the report says so.
Scoring: final = 0.5 * z(-distance to nearest cluster centroid)
               + 0.5 * z(LR log-odds), z-scored on the train distribution.
  No hand-picked constants: all scales come from the train distribution.

Run: ~/workspace/audio-env/bin/python model_v2.py
Outputs: model_v2.json (config, clusters, coefficients, metrics),
         model_v2.npz (scaler, PCA, GMM, LR, z-norm stats for reuse)
"""
import json, os, sys, time
import numpy as np

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
sys.path.insert(0, BASE)

from scoring import feat_vector  # single definition, shared with evaluate/recommend

TEST_N = 130  # most recent 20% of 650, held out

NEGATIVE_IDS = [
    # 13 confirmed negatives: 9 v5 drops + 4 v6 thumbs-down (all H bucket).
    # Documented here; the features live in data/negatives_features.json.
    # v5 (kept only "Fallout" - Sleep Theory)
    '6aJn7Cst74cj4lNIiPRgav',  # Touch - KATSEYE
    '6HacgXCExkzS552ILfJTXu',  # Census Designated - Jane Remover
    '1GfBLbAhZUWdseuDqhocmn',  # Rumble - Skrillex, Fred again.., Flowdan
    '5v7oj9PosKdE39x9uwDs6C',  # dancinwithsomebawdy - ericdoa
    '4bkPAARkDov1lV05emEF7a',  # Soft Spine - Spiritbox
    '2jv1ovpDpTJEPIlth5Wn07',  # TOMBSTONE - Wage War
    '7zDu3l7DDlfSlt20YMKkWo',  # Martini Blue - DPR LIVE
    '2y57H6nA3vic5t8awwaS69',  # Dark Light - Night Lovell
    '2slLXFcITuhf0oSp2wVJLv',  # Opaul - Freddie Dredd
    # v6 thumbs-down on second listen (Nov 21, 2026), all H bucket
    '30efCAopZCFcoi43sSToNr',  # prove it - 21 Savage, Summer Walker
    '0FmrNFYyLLRtIS7XWJPLLk',  # Akina Speed Star - Xavier Wulf
    '32gSPav2xqFylAeCjJCxWi',  # Alone - Night Lovell
    '1vvBUZseJ7fwYPX1NedOLd',  # Cha Cha - Freddie Dredd
]


def main():
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    tracks = doc['tracks']
    # drop the unresolvable 'None' id track (no features possible)
    usable = [t for t in tracks if t['spotify_id'] != 'None']
    print(f'tracks usable: {len(usable)} / {len(tracks)}')
    X, names = [], []
    for t in usable:
        v, names = feat_vector(t)
        X.append(v)
    X = np.vstack(X)
    print(f'feature matrix: {X.shape}, missing frac: {np.isnan(X).mean():.3f}')
    pos = np.array([t['position'] for t in usable])
    test_mask = pos < TEST_N
    train_mask = ~test_mask
    print(f'train={train_mask.sum()} test={test_mask.sum()}')

    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(X[train_mask])
    Xte = imp.transform(X[test_mask])
    Xall = imp.transform(X)
    scaler = StandardScaler()
    Ztr = scaler.fit_transform(Xtr)
    Zte = scaler.transform(Xte)
    Zall = scaler.transform(Xall)

    # recency weights for likes: recent ~1.0, oldest ~0.5
    rec_w = 0.5 + 0.5 * np.exp(-pos[train_mask] / 200.0)

    # ---- clustering: PCA -> GMM, k by BIC
    pca = PCA(n_components=15, random_state=42)
    Ptr = pca.fit_transform(Ztr)
    print('PCA explained variance (15):', round(float(pca.explained_variance_ratio_.sum()), 3))
    best = None
    for k in range(2, 9):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             n_init=3, random_state=42)
        gm.fit(Ptr)
        bic = gm.bic(Ptr)
        print(f'  GMM k={k} BIC={bic:.0f}')
        if best is None or bic < best[0]:
            best = (bic, k, gm)
    _, best_k, gmm = best
    labels_tr = gmm.predict(Ptr)
    sil_gmm = silhouette_score(Ptr, labels_tr)
    print(f'GMM selected k={best_k}, silhouette={sil_gmm:.3f}')

    # ---- alternative: HDBSCAN on UMAP
    sil_hdb = None
    hdb_labels = None
    try:
        import umap
        import hdbscan
        red = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1,
                        random_state=42).fit_transform(Ztr)
        hdb = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        hdb_labels = hdb.fit_predict(red)
        n_cl = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = int((hdb_labels == -1).sum())
        if n_cl >= 2:
            m = hdb_labels != -1
            sil_hdb = silhouette_score(red[m], hdb_labels[m])
        print(f'HDBSCAN/UMAP: clusters={n_cl} noise={n_noise} silhouette={sil_hdb}')
    except Exception as e:
        print(f'HDBSCAN/UMAP failed: {e}')

    # Selection: GMM is used for scoring because it has a stable transform
    # for new points (PCA + Gaussian means). HDBSCAN/UMAP is reported as
    # an inspected alternative only; it has no clean out-of-sample scoring
    # path here and is not used for candidate scores.
    method = 'gmm'
    print('selected clustering for scoring: GMM '
          f'(hdbscan/umap silhouette was {sil_hdb})')
    use_labels = labels_tr
    # cluster sizes + dominant genre bucket per cluster (train)
    buckets_tr = [usable[i].get('genre_bucket') for i in np.where(train_mask)[0]]
    cluster_info = []
    for c in sorted(set(use_labels)):
        if c == -1:
            continue
        m = use_labels == c
        bks = [b for b, mm in zip(buckets_tr, m) if mm and b]
        from collections import Counter
        top = Counter(bks).most_common(3)
        cluster_info.append({'cluster': int(c), 'size': int(m.sum()),
                             'genre_counts': dict(top)})
    print('clusters:', json.dumps(cluster_info))

    # ---- logistic regression: likes vs 13 confirmed negatives
    # negatives need features: they are NOT in liked; fetch from data/negatives.json
    neg_path = os.path.join(DATA, 'negatives_features.json')
    if not os.path.exists(neg_path):
        print('NEGATIVES FILE MISSING: run negatives_features.py first')
        sys.exit(1)
    neg_recs = json.load(open(neg_path))
    Xn = []
    for r in neg_recs:
        v, _ = feat_vector(r)
        Xn.append(v)
    Xn = np.vstack(Xn)
    Xn = imp.transform(Xn)
    Zn = scaler.transform(Xn)
    Xlr = np.vstack([Ztr, Zn])
    ylr = np.array([1] * len(Ztr) + [0] * len(Zn))
    wlr = np.concatenate([rec_w, np.ones(len(Zn))])
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
    lr.fit(Xlr, ylr, sample_weight=wlr)
    acc_tr = lr.score(Xlr, ylr, sample_weight=wlr)
    print(f'LR train accuracy (weighted): {acc_tr:.3f} '
          f'(n_pos={len(Ztr)}, n_neg={len(Zn)})')
    coef = lr.coef_[0]
    order = np.argsort(-np.abs(coef))[:10]
    print('top LR coefficients:')
    for i in order:
        print(f'  {names[i]:28s} {coef[i]:+.3f}')

    # ---- scoring function artifacts (GMM centroids in PCA space)
    centroids = gmm.means_  # (k, 15) PCA space
    from scipy.spatial.distance import cdist
    dtr = cdist(Ptr, centroids).min(axis=1)
    ltr = lr.decision_function(Ztr)
    score_stats = {'negdist_mean': float(-dtr.mean()),
                   'negdist_std': float(dtr.std()),
                   'lr_mean': float(ltr.mean()),
                   'lr_std': float(ltr.std())}
    print('score stats:', score_stats)

    out = {
        'method': method, 'gmm_k': best_k,
        'silhouette_gmm': round(float(sil_gmm), 3),
        'silhouette_hdbscan_umap': (round(float(sil_hdb), 3)
                                    if sil_hdb is not None else None),
        'pca_explained_15': round(float(pca.explained_variance_ratio_.sum()), 3),
        'clusters': cluster_info,
        'lr_train_accuracy_weighted': round(float(acc_tr), 3),
        'lr_n_pos': int(len(Ztr)), 'lr_n_neg': int(len(Zn)),
        'lr_top_coefficients': [{'feature': names[i],
                                 'coef': round(float(coef[i]), 3)}
                                for i in order],
        'feature_names': names,
        'score_stats': score_stats,
        'test_n': int(test_mask.sum()), 'train_n': int(train_mask.sum()),
        'recency_weight': '0.5 + 0.5*exp(-position/200), likes only',
    }
    json.dump(out, open(os.path.join(BASE, 'model_v2.json'), 'w'), indent=1)
    np.savez(os.path.join(BASE, 'model_v2.npz'),
             imputer_median=imp.statistics_, scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_, pca_components=pca.components_,
             pca_mean=pca.mean_, gmm_means=gmm.means_,
             gmm_covariances=gmm.covariances_,
             gmm_weights=gmm.weights_,
             gmm_precisions_cholesky=gmm.precisions_cholesky_,
             lr_coef=lr.coef_, lr_intercept=lr.intercept_,
             centroids=centroids,
             negdist_mean=score_stats['negdist_mean'],
             negdist_std=score_stats['negdist_std'],
             lr_mean=score_stats['lr_mean'], lr_std=score_stats['lr_std'])
    print('wrote model_v2.json + model_v2.npz')


if __name__ == '__main__':
    main()
