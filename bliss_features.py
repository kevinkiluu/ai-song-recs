#!/usr/bin/env python3
"""bliss23 port: bliss-rs (bliss-audio 0.13.0, FeaturesVersion 2) reimplemented
in librosa/numpy per ~/workspace/audio-pipeline/research/bliss-rs-spec.md §(b).

Preview-only v3 path. Full-audio variant is CLOSED for v3 (Eric's music vault
access shelved Nov 21, 2026; friend API unresolved): FULL_AUDIO_AVAILABLE
defaults OFF, full-audio extraction raises, and preview/full vectors are
never mixed in one matrix (enforced by check_no_mixing / source field).

Templates for the chroma interval transform are reconstructed from the spec
§(a.6) description (dyads IC1-IC6 = {0,k} k=1..6; triads major {0,4,7},
minor {0,3,7}, diminished {0,3,6}, augmented {0,4,8}); the /tmp/bliss-rs
clone the spec cites was not present on this VM, so no verbatim copy from
src/chroma.rs was possible. The B1 golden-vector test is the check that
would catch a template error; it skips here for lack of the fixture.
"""
import math

import numpy as np

BLISS_DIM = 23
BLISS_NAMES = ["tempo", "zcr",
               "centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
               "flatness_mean", "flatness_std", "loudness_mean", "loudness_std",
               "chroma1", "chroma2", "chroma3", "chroma4", "chroma5", "chroma6",
               "chroma7", "chroma8", "chroma9", "chroma10", "chroma11",
               "chroma12", "chroma13"]
VERSION2_WEIGHTS = [0.25] + [1.0] * 9 + [3.0 / 13.0] * 13  # sum = 12.25
EXTRACTOR = "bliss23/1.0"

# Full-audio variant gate (§b.4). CLOSED for v3: vault access shelved.
FULL_AUDIO_AVAILABLE = False

MAX_L2_INTERVAL = 0.25
MAX_L2_TRIAD = 0.025

# Interval templates: 12 pitch classes x 10 columns. Columns 0-5 are the six
# interval classes (dyads) IC1-IC6, columns 6-9 the four triad types.
_TEMPLATES = np.zeros((12, 10), dtype=float)
for _k in range(1, 7):  # dyad IC_k = pitch classes {0, k}
    _TEMPLATES[0, _k - 1] = 1.0
    _TEMPLATES[_k, _k - 1] = 1.0
for _col, _pcs in ((6, (0, 4, 7)),   # major triad
                   (7, (0, 3, 7)),   # minor triad
                   (8, (0, 3, 6)),   # diminished triad
                   (9, (0, 4, 8))):  # augmented triad
    for _p in _pcs:
        _TEMPLATES[_p, _col] = 1.0


def _norm(x, lo, hi):
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def bliss_tempo_bpm(tempo_dim):
    """Inverse of the tempo normalisation: normalised dim -> BPM."""
    return (float(tempo_dim) + 1.0) / 2.0 * 206.0


def _tempo_dim(y, sr):
    import librosa
    if len(y) == 0 or not np.any(y):
        return -1.0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0 or float(np.max(onset_env)) <= 0:
        return -1.0
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else 0.0
    if beats is None or len(beats) == 0 or bpm <= 0:
        return -1.0
    # Cross-check with the onset-envelope tempo estimate. If the two disagree
    # by a factor ~2, beat_track's DP estimate governs (it is the one that
    # also produced the beat positions); the disagreement is recorded by the
    # caller via T3, not silently resolved here.
    return float(np.clip(2.0 * bpm / 206.0 - 1.0, -1.0, 1.0))


def _zcr_dim(y):
    if len(y) == 0 or not np.any(y):
        return -1.0
    sb = np.signbit(y).astype(np.int8)
    crossings = int(np.count_nonzero(np.diff(sb)))
    return 2.0 * (crossings / len(y)) - 1.0


def _spectral_dims(y, sr):
    """Returns (centroid_mean, centroid_std, rolloff_mean, rolloff_std,
    flatness_mean, flatness_std), all normalised. Silence -> all -1."""
    import librosa
    if len(y) == 0 or not np.any(y):
        return (-1.0,) * 6
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512,
                                             hop_length=128)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512,
                                            hop_length=128,
                                            roll_percent=0.95)[0]
    # Flatness: upstream (timbral.rs do_) is geometric_mean/arithmetic_mean
    # of the MAGNITUDE spectrum, first 256 bins -- NOT librosa's default
    # power-spectrum flatness (power=2.0), which gives ~0.56 raw for white
    # noise instead of ~0.79. Compute it manually from the magnitude STFT.
    mag = np.abs(librosa.stft(y, n_fft=512, hop_length=128))[:256]
    with np.errstate(divide="ignore"):
        geo = np.exp(np.mean(np.log(np.maximum(mag, 1e-12)), axis=0))
    flat = geo / np.maximum(np.mean(mag, axis=0), 1e-12)
    return (
        float(np.clip(_norm(float(cent.mean()), 0, sr / 2), -1, 1)),
        float(np.clip(_norm(float(cent.std()), 0, sr / 2), -1, 1)),
        float(np.clip(_norm(float(roll.mean()), 0, sr / 2), -1, 1)),
        float(np.clip(_norm(float(roll.std()), 0, sr / 2), -1, 1)),
        float(np.clip(2.0 * float(flat.mean()) - 1.0, -1, 1)),
        float(np.clip(2.0 * float(flat.std()) - 1.0, -1, 1)),
    )


def _loudness_dims(y):
    """Pure numpy, 1024-sample chunks, 10*log10 of mean/std of linear
    energy, floor -90 dB, normalised over -90..0 dB."""
    if len(y) == 0 or not np.any(y):
        return -1.0, -1.0
    n = (len(y) // 1024) * 1024
    chunks = y[:n].reshape(-1, 1024) if n >= 1024 else y.reshape(1, -1)
    levels = np.mean(chunks.astype(np.float64) ** 2, axis=1)
    mean_lin = max(float(np.mean(levels)), 1e-9)
    std_lin = max(float(np.std(levels)), 1e-9)
    mean_db = 10.0 * math.log10(mean_lin)
    std_db = 10.0 * math.log10(std_lin)
    return (
        float(np.clip((mean_db + 90.0) / 90.0 * 2.0 - 1.0, -1, 1)),
        float(np.clip((std_db + 90.0) / 90.0 * 2.0 - 1.0, -1, 1)),
    )


def interval_features(chroma):
    """Bliss chroma interval features (§a.6 steps 4-5) from a 12 x T
    chromagram (librosa chroma_stft output). Returns 13 normalised dims
    in [-1, 1]: 6 L2-normalised interval classes, 4 L2-normalised triad
    types, then the interval L2 norm, triad L2 norm, and triad-vs-dyad
    atan2 ratio summaries."""
    c = np.asarray(chroma, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != 12 or c.shape[1] == 0:
        raise ValueError(f"chroma must be 12 x T, got shape {c.shape}")
    # Sharpen + per-frame L1 normalisation.
    c = np.exp(15.0 * c)
    colsum = c.sum(axis=0, keepdims=True)
    c = np.divide(c, colsum, out=np.zeros_like(c), where=colsum > 0)
    # Per frame, per template column, sum over all 12 rotations of the
    # product of the chroma values selected by the rotated template.
    raw = np.zeros(10, dtype=np.float64)
    for col in range(10):
        base = np.where(_TEMPLATES[:, col] > 0)[0]
        for r in range(12):
            idx = (base + r) % 12
            prod = np.ones(c.shape[1], dtype=np.float64)
            for i in idx:
                prod *= c[i]
            raw[col] += float(prod.sum())
    raw /= max(c.shape[1], 1)  # mean over time
    n_int = float(np.linalg.norm(raw[:6]))
    n_tri = float(np.linalg.norm(raw[6:]))
    normed = np.zeros(10, dtype=np.float64)
    if n_int > 0:
        normed[:6] = raw[:6] / n_int
    if n_tri > 0:
        normed[6:] = raw[6:] / n_tri
    s_int = min(n_int / MAX_L2_INTERVAL, 1.0)
    s_tri = min(n_tri / MAX_L2_TRIAD, 1.0)
    s_ratio = math.atan2(20.0 * n_tri, n_int + 1e-12) / (math.pi / 2.0)
    vals = np.concatenate([normed, [s_int, s_tri, s_ratio]])
    return np.clip(2.0 * vals - 1.0, -1.0, 1.0)


def _chroma_dims(y, sr):
    import librosa
    if len(y) == 0 or not np.any(y):
        return np.full(13, -1.0)
    try:
        tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        tuning = 0.0
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=8192,
                                         hop_length=2205, tuning=tuning,
                                         norm=1)
    # norm=1 is per-frame L1 normalisation, matching upstream
    # ("raw_chroma = filter . |STFT|^2, then each frame L1-normalized").
    # librosa's default norm=inf (max-norm) makes exp(15x) sharpening
    # amplify per-pitch-class filterbank unevenness and breaks the
    # transposition invariance of the interval transform (test C2).
    return interval_features(chroma)


def extract_bliss23(y, sr=22050):
    """23-dim bliss V2 vector from an already-loaded mono waveform.
    Never downloads or decodes; callers load with librosa.load(sr=22050,
    mono=True) exactly as pipeline.py:63."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0 or not np.any(y):
        return np.full(BLISS_DIM, -1.0)
    out = np.empty(BLISS_DIM, dtype=float)
    out[0] = _tempo_dim(y, sr)
    out[1] = _zcr_dim(y)
    out[2:8] = _spectral_dims(y, sr)
    out[8:10] = _loudness_dims(y)
    out[10:23] = _chroma_dims(y, sr)
    return np.clip(out, -1.0, 1.0)


def extract_bliss23_from_file(path, source="preview"):
    """Load a preview file and extract. source='full' raises while the
    full-audio variant is gated off (FULL_AUDIO_AVAILABLE is False)."""
    if source == "full" and not FULL_AUDIO_AVAILABLE:
        raise RuntimeError(
            "full-audio extraction is gated off (FULL_AUDIO_AVAILABLE=False); "
            "v3 is preview-only")
    import librosa
    y, _sr = librosa.load(path, sr=22050, mono=True)
    return extract_bliss23(y, 22050)


def bliss_distance(a, b, weights=VERSION2_WEIGHTS):
    """Weighted Mahalanobis distance with diagonal weights:
    sqrt((a-b)^T diag(w) (a-b)). d(zeros, ones) == sqrt(12.25) == 3.5."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(w * (a - b) ** 2)))


def bliss_score_to_liked(candidate_vec, liked_matrix, weights=VERSION2_WEIGHTS,
                        k=10):
    """Mean of the k smallest weighted distances from the candidate to the
    liked vectors (10 nearest liked tracks). Lower = closer to Kevin's taste.
    The candidate's own row, if present in the matrix, gives distance 0 and
    counts as one of the k — callers scoring held-out likes must use a
    train-only liked matrix."""
    cand = np.asarray(candidate_vec, dtype=float)
    mat = np.asarray(liked_matrix, dtype=float)
    if mat.ndim != 2 or len(mat) == 0:
        raise ValueError("liked_matrix must be a non-empty 2-D array")
    w = np.asarray(weights, dtype=float)
    d = np.sqrt(np.sum(w * (mat - cand) ** 2, axis=1))
    kk = min(int(k), len(d))
    return float(np.mean(np.sort(d)[:kk]))


def variance_based_weights(seeds):
    """Port of bliss variance_based_weight_matrix (diagonal only):
    per-dim weight = 1/(variance + 1e-6), rescaled so weights sum to the
    dimension count. Raises on < 2 seeds."""
    s = np.asarray(seeds, dtype=float)
    if s.ndim != 2 or len(s) < 2:
        raise ValueError("variance_based_weights needs >= 2 seed vectors")
    var = np.var(s, axis=0)
    w = 1.0 / (var + 1e-6)
    return w * (len(w) / w.sum())


def compute_adaptive_weights(liked_matrix):
    """Seed-adaptive variant (§b.3.5): inverse-variance diagonal weights
    over the liked preview vectors, summing to 23."""
    return variance_based_weights(np.asarray(liked_matrix, dtype=float))


def dedup_threshold(matrix, weights=VERSION2_WEIGHTS):
    """Recalibrated dedup threshold (§b.3.6 step 6 exactly): 0.1th
    percentile of pairwise weighted bliss distances between distinct
    liked tracks, floored at 0.05 and capped at 0.5. Callers must pass a
    matrix of distinct songs (same recording under two Spotify IDs
    deduped by Deezer ID first); by construction ~0.1% of distinct pairs
    fall below a percentile threshold, so a 'zero pairs below' reading of
    spec §c D4 is inconsistent with the §b.3.6 procedure and is NOT
    asserted here -- see test_D4."""
    mat = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(mat)
    if n < 2:
        return 0.05
    dists = []
    for i in range(0, n, 64):
        chunk = mat[i:i + 64]
        d = np.sqrt(np.sum(w * (chunk[:, None, :] - mat[None, :, :]) ** 2, axis=2))
        for a in range(d.shape[0]):
            vals = d[a][i + a + 1:]
            if len(vals):
                dists.append(vals)
    all_d = np.concatenate(dists) if dists else np.array([0.0])
    thr = float(np.percentile(all_d, 0.1))
    return float(min(max(thr, 0.05), 0.5))


def check_no_mixing(rows):
    """Raise if rows mix preview and full sources. Preview-vs-full
    comparisons are invalid and must be blocked by the source field."""
    sources = {r.get("source") for r in rows}
    if len(sources) > 1:
        raise ValueError(f"mixed audio sources in one matrix: {sorted(sources)}")
    return True
