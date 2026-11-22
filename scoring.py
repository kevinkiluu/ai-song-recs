#!/usr/bin/env python3
"""Shared scoring for pipeline v2: feature vectors, model loading, and
the v6 baseline scorer. Used by model_v2.py, evaluate.py and recommend_v2.py
so all three agree on exactly what a score means.

v2 score = 0.5 * z(-distance to nearest GMM cluster centroid, PCA space)
         + 0.5 * z(logistic-regression log-odds)
Higher = more like Kevin's taste. All z-scales come from the train set.
v6 score (baseline, from recommend.py) = |bpm-med|/IQR
         + |energy-mean|/0.10 + |brightness-mean|/500 - 0.5 if minor.
Lower = closer fit. Per-bucket profile when a bucket is known, else the
global profile.
"""
import json, os
import numpy as np

BASE = os.path.expanduser('~/workspace/audio-pipeline')

RB_KEYS = ['danceability', 'energy', 'valence', 'acousticness',
           'instrumentalness', 'speechiness', 'liveness', 'loudness',
           'tempo', 'mode']
LIB_KEYS = ['bpm', 'onset_strength_mean', 'brightness_mean',
            'brightness_std', 'rolloff_mean', 'bandwidth_mean',
            'flatness_mean', 'zcr_mean', 'zcr_std', 'rms_mean', 'rms_std',
            'loudness_db',
            'seg_brightness_delta', 'seg_energy_delta']
# hpss_harmonic_ratio was dropped from analyze_deep (HPSS too slow on
# this VM); it is not a model feature. ReccoBeats instrumentalness /
# speechiness cover the vocal/instrumental signal instead.
LIB_VEC = [('contrast_mean', 7), ('mfcc_mean', 13), ('mfcc_std', 13),
           ('chroma_hist', 12), ('tonnetz_mean', 6)]

# v6 per-bucket profile (recommend.py, Sep 12 2026 analysis)
V6_PROFILE = {
    'E': {'bpm_med': 123.0, 'bpm_p25': 112.3, 'bpm_p75': 136.0,
          'energy': 0.320, 'bright': 2668},
    'R': {'bpm_med': 123.0, 'bpm_p25': 103.4, 'bpm_p75': 161.5,
          'energy': 0.322, 'bright': 2655},
    'H': {'bpm_med': 136.0, 'bpm_p25': 117.5, 'bpm_p75': 161.5,
          'energy': 0.341, 'bright': 2897},
}
V6_GLOBAL = {'bpm_med': 123.0, 'bpm_p25': 103.0, 'bpm_p75': 152.0,
             'energy': 0.320, 'bright': 2766}


def feat_vector(rec):
    """Feature vector + names. Missing values -> np.nan (imputed later)."""
    rb = rec.get('reccobeats') or {}
    lb = rec.get('librosa') or {}
    vals, names = [], []
    for k in RB_KEYS:
        v = rb.get(k)
        vals.append(float(v) if v is not None else np.nan)
        names.append('rb_' + k)
    key = rb.get('key')
    if key is None:
        vals += [np.nan, np.nan]
    else:
        ang = 2 * np.pi * float(key) / 12.0
        vals += [float(np.sin(ang)), float(np.cos(ang))]
    names += ['rb_key_sin', 'rb_key_cos']
    for k in LIB_KEYS:
        v = lb.get(k)
        vals.append(float(v) if v is not None else np.nan)
        names.append('lb_' + k)
    for k, n in LIB_VEC:
        v = lb.get(k)
        if v is None or len(v) != n:
            vals += [np.nan] * n
        else:
            vals += [float(x) for x in v]
        names += [f'lb_{k}_{i}' for i in range(n)]
    lk = (lb.get('key') or '')
    vals.append(1.0 if 'minor' in lk else (0.0 if lk else np.nan))
    names.append('lb_is_minor')
    return np.array(vals, dtype=float), names


def v6_score(librosa_feats, bucket=None):
    """The v6 baseline score. Lower = closer to the profile.
    Returns (score, bucket_used) or (None, None) if librosa is missing."""
    if not librosa_feats:
        return None, None
    pr = V6_PROFILE.get(bucket) if bucket in V6_PROFILE else V6_GLOBAL
    used = bucket if bucket in V6_PROFILE else 'global'
    iqr = max(pr['bpm_p75'] - pr['bpm_p25'], 1)
    bpm_z = abs(librosa_feats['bpm'] - pr['bpm_med']) / iqr
    e_z = abs(librosa_feats['rms_mean'] - pr['energy']) / 0.10
    b_z = abs(librosa_feats['brightness_mean'] - pr['bright']) / 500.0
    key_bonus = -0.5 if 'minor' in (librosa_feats.get('key') or '') else 0.0
    return round(bpm_z + e_z + b_z + key_bonus, 2), used


class V2Model:
    """Loads model_v2.npz/json and scores feature vectors."""

    def __init__(self):
        z = np.load(os.path.join(BASE, 'model_v2.npz'))
        self.median = z['imputer_median']
        self.mean = z['scaler_mean']
        self.scale = z['scaler_scale']
        self.pca_comp = z['pca_components']
        self.pca_mean = z['pca_mean']
        self.centroids = z['centroids']
        self.lr_coef = z['lr_coef'][0]
        self.lr_intercept = float(z['lr_intercept'][0])
        self.negdist_mean = float(z['negdist_mean'])
        self.negdist_std = float(z['negdist_std'])
        self.lr_mean = float(z['lr_mean'])
        self.lr_std = float(z['lr_std'])
        cfg = json.load(open(os.path.join(BASE, 'model_v2.json')))
        self.gmm_k = cfg['gmm_k']
        self.clusters = cfg['clusters']

    def score_matrix(self, X_raw):
        """X_raw: (n, d) with NaNs. Returns dict of score components."""
        X = np.where(np.isnan(X_raw), self.median, X_raw)
        Z = (X - self.mean) / self.scale
        P = (Z - self.pca_mean) @ self.pca_comp.T
        # distance to nearest centroid in PCA space
        d = np.sqrt(((P[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2))
        dmin = d.min(axis=1)
        logodds = Z @ self.lr_coef + self.lr_intercept
        z_negdist = (-dmin - self.negdist_mean) / self.negdist_std
        z_lr = (logodds - self.lr_mean) / self.lr_std
        final = 0.5 * z_negdist + 0.5 * z_lr
        nearest = d.argmin(axis=1)
        return {'final': final, 'z_negdist': z_negdist, 'z_lr': z_lr,
                'dmin': dmin, 'logodds': logodds, 'nearest_cluster': nearest}

    def score_records(self, records):
        X, _ = zip(*[feat_vector(r) for r in records]) if records else ([], [])
        X = np.vstack(X) if records else np.zeros((0, len(self.mean)))
        return self.score_matrix(X)
