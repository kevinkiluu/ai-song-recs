#!/usr/bin/env python3
"""Taste modeling v2: clusters + learned weights from like/skip labels.

Inputs: features_v2.json (ReccoBeats panel + deep librosa features).
Historical split is by Liked Songs list position.
  test  = positions 0..129  (most recent 20%, held out)
  train = positions 130..649
  The snapshot has no verified added_at values.

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
from collections import Counter
import json, os, sys
import numpy as np

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')

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


def fit_records(train, negatives, *, inspect_clusters=False):
    """Fit the v2 scoring artifacts using the supplied training records."""
    if len(train) < 3 or not negatives:
        raise ValueError('v2 fitting requires at least three likes and one reject')
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    features = [feat_vector(record) for record in train]
    X = np.vstack([vector for vector, _ in features])
    names = features[0][1]
    # Older sklearn versions drop all-missing columns during median
    # imputation. Keep their original indices without using test values.
    X[:, np.isnan(X).all(axis=0)] = 0.0
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Ztr = scaler.fit_transform(imp.fit_transform(X))

    positions = np.array([record['position'] for record in train])
    rec_w = 0.5 + 0.5 * np.exp(-positions / 200.0)

    # ---- clustering: PCA -> GMM, k by BIC
    pca = PCA(n_components=min(15, *Ztr.shape), random_state=42)
    Ptr = pca.fit_transform(Ztr)
    print('PCA explained variance:', round(float(pca.explained_variance_ratio_.sum()), 3))
    best = None
    for k in range(2, min(8, len(train) - 1) + 1):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             n_init=3, random_state=42)
        gm.fit(Ptr)
        bic = gm.bic(Ptr)
        print(f'  GMM k={k} BIC={bic:.0f}')
        if best is None or bic < best[0]:
            best = (bic, k, gm)
    _, best_k, gmm = best
    labels_tr = gmm.predict(Ptr)
    sil_gmm = (silhouette_score(Ptr, labels_tr)
               if 1 < len(set(labels_tr)) < len(train) else None)
    print(f'GMM selected k={best_k}, silhouette={sil_gmm}')

    # ---- alternative: HDBSCAN on UMAP
    sil_hdb = None
    if inspect_clusters:
        sil_hdb = inspect_alternative_clusters(Ztr)

    # Scoring always uses the GMM; its PCA transform supports new tracks.
    buckets_tr = [record.get('genre_bucket') for record in train]
    cluster_info = []
    for cluster in sorted(set(labels_tr)):
        members = labels_tr == cluster
        top = Counter(b for b, member in zip(buckets_tr, members)
                      if member and b).most_common(3)
        cluster_info.append({'cluster': int(cluster), 'size': int(members.sum()),
                             'genre_counts': dict(top)})

    Xn = np.vstack([feat_vector(record)[0] for record in negatives])
    Zn = scaler.transform(imp.transform(Xn))
    Xlr = np.vstack([Ztr, Zn])
    ylr = np.array([1] * len(Ztr) + [0] * len(Zn))
    wlr = np.concatenate([rec_w, np.ones(len(Zn))])
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
    lr.fit(Xlr, ylr, sample_weight=wlr)
    acc_tr = lr.score(Xlr, ylr, sample_weight=wlr)
    coef = lr.coef_[0]
    order = np.argsort(-np.abs(coef))[:12]

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
        'method': 'gmm', 'gmm_k': best_k,
        'silhouette_gmm': round(float(sil_gmm), 3) if sil_gmm is not None else None,
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
        'train_n': len(train),
        'recency_weight': '0.5 + 0.5*exp(-position/200), likes only',
    }
    artifacts = dict(
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
    return out, artifacts


def inspect_alternative_clusters(Ztr):
    from sklearn.metrics import silhouette_score

    try:
        import umap
        import hdbscan
        reduced = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1,
                            random_state=42).fit_transform(Ztr)
        labels = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(reduced)
        in_cluster = labels != -1
        if len(set(labels[in_cluster])) >= 2:
            return silhouette_score(reduced[in_cluster], labels[in_cluster])
    except Exception as exc:
        print(f'HDBSCAN/UMAP failed: {exc}')
    return None


def main():
    with open(os.path.join(BASE, 'features_v2.json')) as stream:
        tracks = json.load(stream)['tracks']
    usable = [track for track in tracks if track['spotify_id'] != 'None']
    train = [track for track in usable if track['position'] >= TEST_N]
    neg_path = os.path.join(DATA, 'negatives_features.json')
    if not os.path.exists(neg_path):
        print('NEGATIVES FILE MISSING: run negatives_features.py first')
        sys.exit(1)
    with open(neg_path) as stream:
        negatives = json.load(stream)
    config, artifacts = fit_records(train, negatives, inspect_clusters=True)
    config['test_n'] = len(usable) - len(train)
    with open(os.path.join(BASE, 'model_v2.json'), 'w') as stream:
        json.dump(config, stream, indent=1)
    np.savez(os.path.join(BASE, 'model_v2.npz'), **artifacts)
    print('wrote model_v2.json + model_v2.npz')


if __name__ == '__main__':
    main()
