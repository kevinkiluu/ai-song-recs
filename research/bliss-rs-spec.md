# Porting spec: bliss-rs → ai-song-recs v3 ("bliss23")

- **Upstream:** bliss-rs (`bliss-audio` crate), version **0.13.0**,
  repo https://github.com/Polochon-street/bliss-rs,
  commit `0de922d` ("Bump ffmpeg version (#136)"), cloned 2026-11-21 to `/tmp/bliss-rs`.
- **Researched:** 2026-11-21. All upstream claims below were read from that clone
  (file:line cited). Nothing in this spec is inferred from memory of bliss.
- **Companion repo (metric learning only):**
  https://github.com/Polochon-street/bliss-metric-learning, cloned 2026-11-21 to `/tmp/bliss-ml`.
- **Target:** v3 = v2 pipeline + the 23-dim bliss feature vector and bliss distance,
  in `~/workspace/audio-pipeline/`. New files only: `bliss_features.py`,
  `tests/test_bliss_features.py`; extend `scoring.py` (feature vector),
  `evaluate.py` (bliss scorer + 13-negative control), `features_v2.py`
  (cache `data/bliss23.json`). Do NOT modify `pipeline.py:analyze()` in place;
  add a parallel extractor. Do NOT touch `~/workspace/spotify-liked-songs.txt`
  or append to `~/workspace/spotify-recommended-tracks.txt`.
- **Decision (binding):** the bliss23 block is **additive**. The existing v2 vector
  (ReccoBeats + librosa deep + GMM/LR in `scoring.py`) stays intact and keeps
  working if bliss extraction fails for a track (NaN + imputer median).
  Bliss contributes (i) 23 new features, (ii) one new scorer (bliss weighted
  distance to the liked-set centroid / to nearest liked neighbours), (iii) one
  new ablation input. It does not replace the v2 scorer in v3.

## 0. Conventions used in this spec

- All bliss features are normalized to **[-1, 1]** by
  `normalize(x) = 2*(x - MIN) / (MAX - MIN) - 1`
  (`src/utils.rs`, trait `Normalize`). Every ported value MUST be stored
  normalized, in the exact index order of §(b) Table 1.
- Canonical audio input for every descriptor is **mono, f32, 22050 Hz**
  (`src/lib.rs`: `CHANNELS = 1`, `SAMPLE_RATE = 22050`;
  `src/song/decoder.rs:54-55,119-120`). This already matches
  `pipeline.py:63` (`librosa.load(path, sr=22050, mono=True)`). No resampling
  change is needed.
- "Preview-only" = Deezer 30 s MP3 preview, the only audio that exists today
  (502 previews in `previews/` at research time). "Full-audio" = a complete
  track file, available only if the friend API lands. Every feature below
  states which variant it belongs to and what the preview approximation loses.
- Numbers are thresholds, not adjectives. A claim without a number in §(d)
  does not count as passing.

---

## (a) How the upstream mechanism works

### (a.1) What a Song is

`Song::analyze(sample_array)` (`src/song/mod.rs`, `analyze_with_options`)
runs **five descriptor families in parallel threads** over the whole decoded
sample array and concatenates their outputs into one `Analysis`:
a `Vec<f32>` of length **23** (FeaturesVersion 2, `NUMBER_FEATURES = 23`;
Version 1 had 20). Feature order is fixed by the concatenation
(`src/song/mod.rs`, end of `analyze_with_options`):

```
[tempo, zcr,
 centroid_mean, centroid_std, rolloff_mean, rolloff_std,
 flatness_mean, flatness_std,
 loudness_mean, loudness_std,
 chroma1 .. chroma13]
 = 1 + 1 + 6 + 2 + 13 = 23
```

`Analysis::distance` panics on version mismatch; V1 and V2 vectors are
**not interchangeable** (`src/song/mod.rs`, `test_analysis_distance_mismatched_versions`).
Port V2 only. V1 differs only in the last 10 chroma values (raw, unnormalized
interval features; V1 chroma7-10 sit between -0.98 and -0.99 and are near-useless
— `AnalysisIndexv1` docs). Do not port V1.

### (a.2) Tempo — 1 dim (index 0)

Source: `src/temporal.rs` (`BPMDesc`), engine `src/aubio.rs` (`Tempo` struct,
a Rust transcription of aubio's `tempo.c` / `beattracking.c`).

- Window 512 samples, hop 256 (`BPMDesc::WINDOW_SIZE = 512`,
  `HOP_SIZE = 256`), at 22050 Hz.
- Per hop: phase vocoder → **SpecFlux** onset detection function (sum of
  positive spectral-magnitude differences, `src/aubio.rs:428-466`) →
  adaptive-threshold peak picker (threshold 0.3, `Tempo::new`) →
  beat tracker over a ~6 s detection-function window
  (`winlen = next_power_of_two(5.8 * samplerate / hop_size)`).
- Each time a beat fires, the current BPM estimate is recorded; the final
  value is the **median** of all recorded BPMs
  (`BPMDesc::get_value`, `quantile_mut(0.5, Midpoint)`).
- Normalization: `MIN = 0, MAX = 206` BPM → `2*bpm/206 - 1`.
  Upstream notes aubio "has trouble to identify tempo > 190 BPM"
  (`src/temporal.rs` doc comment) and the `MAX_VALUE` comment says the true
  cap "should really be 413, needs testing" — the 206 cap is empirical, not
  derived. **No beat = -1.0** (sentinel, not 0).
- **Known upstream gap (do not copy silently):** the beat-tracker's confidence
  is computed but unused. `src/temporal.rs` carries
  `TODO>1.0 use the confidence value to discard this descriptor if confidence
  is too low`, and the V2 weight for tempo is 0.25 instead of 1.0 with the
  comment "beat tracker unreliable on onset-poor music. Pending confidence use."
  (`src/lib.rs`, `VERSION2_WEIGHTS`). Tempo is the least trusted bliss dim
  by its own author's weighting.

### (a.3) Zero-crossing rate — 1 dim (index 1)

Source: `src/timbral.rs` (`ZeroCrossingRateDesc`), helper
`src/utils.rs:number_crossings` (transcribed from Essentia's
`zerocrossingrate.cpp`).

- One global count over the **whole song**: total sign changes / total samples
  (`do_` is called once on the full array in `analyze_with_options`;
  per-chunk calls sum identically). Range 0..1 → normalized `2*zcr - 1`.
- Semantics per upstream docs: percussiveness / speech-vs-instrument
  discriminator. Silence → -1.0.

### (a.4) Spectral centroid / rolloff / flatness — 6 dims (indices 2-7)

Source: `src/timbral.rs` (`SpectralDesc`), engine `src/aubio.rs`
(`PVoc`, `SpecDesc`, `spectral_centroid`, `spectral_rolloff`).

- One shared FFT: phase vocoder, **window 512, hop 128**
  (`WINDOW_SIZE = 512, HOP_SIZE = WINDOW_SIZE / 4`).
- Per frame: centroid = magnitude-weighted mean bin frequency;
  rolloff = bin frequency below which **95%** of spectral energy lies
  (`src/aubio.rs:50`, `cumsum *= 0.95`; aubio's `specdesc_rolloff` transcription);
  flatness = geometric mean / arithmetic mean of the magnitude spectrum
  (first 256 bins of the phase-vocoder output; `geometric_mean` in
  `src/utils.rs` is a chunked f64 implementation that returns 0.0 if any
  8-sample chunk product is 0).
- Each descriptor is summarized as **(mean, std)** over all frames.
  Centroid and rolloff normalize over 0..`SAMPLE_RATE/2` = 0..11025 Hz;
  flatness normalizes over 0..1 (its own formula, `2*x - 1` for both mean
  and std — note: the std is normalized as if it were a value in [0,1],
  so flatness-std is almost always near -1; that is upstream behaviour,
  reproduce it exactly).
- Upstream cross-checks recorded in `src/timbral.rs` tests (against Essentia
  with Hann window): on `data/s16_mono_22_5kHz.flac`, centroid raw mean
  1354.2273 Hz (Essentia 1351), rolloff raw mean 2026.7644 Hz (Essentia
  1979.63), flatness raw mean 0.111949615 (Essentia 0.11197535). The aubio
  phase-vocoder and a plain STFT agree to ~0.1-2.5% on these — this is the
  evidence that a librosa-STFT port can match bliss within a few percent
  rather than exactly.

### (a.5) Loudness — 2 dims (indices 8-9)

Source: `src/misc.rs` (`LoudnessDesc`, `level_lin` transcribed from
aubio `mathutils.c:328-340`).

- Non-overlapping chunks of **1024** samples; per chunk
  `level = mean(x^2)` (mean-square energy, linear domain).
- Summary: `mean_dB = 10*log10(mean(levels))`,
  `std_dB = 10*log10(std(levels))` — note the std is taken in the **linear**
  domain and then converted to dB; do not take std of dB values.
- Floor: any of mean/std below `1e-9` is clamped to `1e-9` (i.e. -90 dB).
  Normalize over -90..0 dB. Silence → (-1, -1); constant ±1.0 signal → (1, -1).
- Upstream is explicit that this is recording-dependent by design
  ("the same song would yield different loudness on different recordings...
  which is exactly what we want"). It is **not** loudness-normalized,
  not perceptual (no K-weighting, no LUFS), and not comparable across
  different masterings/encodings of the same track.

### (a.6) Chroma / interval features — 13 dims (indices 10-22)

Source: `src/chroma.rs` (`ChromaDesc`), `src/utils.rs:stft`.
This is the largest and most distinctive part of bliss; it is **not**
a plain 12-bin chromagram.

Pipeline, in order (`ChromaDesc::do_` then `get_values`):

1. **STFT:** `n_fft = 8192`, hop **2205** samples (= 100 ms at 22050 Hz),
   periodic Hann window, reflect padding (`utils::stft`). Magnitude spectrum.
   Unlike the other descriptors, chroma is documented as performing better
   when given the **full song at once, not streamed**
   (`src/chroma.rs` doc comment on `ChromaDesc` and on `do_`).
2. **Tuning estimation:** `estimate_tuning` → `pip_track` (parabolic
   interpolated peak tracking, fmin 150 Hz, fmax min(4000, sr/2),
   magnitude threshold 0.1 × per-frame max) → `pitch_tuning` (histogram of
   pitch deviations from A440, resolution 0.01, 12 bins/octave). Returns a
   tuning offset in semitones; 0.0 if no pitches found.
3. **Chroma filterbank:** `chroma_filter` is a Rust rewrite of
   librosa's `chroma_stft` filterbank (`src/chroma.rs` comment cites
   `librosa/feature/spectral.py`, `chroma(22050, n_fft=5, n_chroma=12)`;
   README credits librosa directly). Center octave 5.0, octave width 2.0,
   Gaussian weighting, L2-normalized columns, rolled by 3 bins.
   Applied as `raw_chroma = filter · |STFT|^2`, then each frame L1-normalized.
4. **Interval features** (`chroma_interval_features`, after
   AudioLabs Erlangen FMP notebooks, https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html,
   and the paper "Timbre-invariant Audio Features for Style Analysis of
   Classical Music", https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1461.pdf):
   - Sharpen: `chroma = normalize_feature_sequence(exp(15 * chroma))`
     (per-frame L1 normalization after exponentiation).
   - **Templates** (12 rows × 10 columns, `src/chroma.rs`,
     `chroma_interval_features`): each column is a binary pitch-class-set
     pattern; the matrix is rotated through all 12 transpositions and the
     product of the chroma values selected by each rotated template is summed.
     Columns 0-5 are the six **interval classes (dyads)** IC1-IC6
     (semitone, whole tone, minor 3rd, major 3rd, perfect 4th, tritone);
     columns 6-9 are the four **triad types** (major, minor, diminished,
     augmented). Because every template is summed over all 12 rotations,
     **the result is transposition-invariant: it encodes interval content
     and triad quality, not key**. There is no key/mode output anywhere
     in bliss.
   - Mean over time → 10 raw values.
5. **V2 summarization** (`get_values`, the part V1 lacks):
   - L2-normalize IC1-6 as a group and IC7-10 as a group (each divided by
     its own L2 norm; skipped if norm is 0).
   - Append three summary dims: L2 norm of IC1-6 normalized against ceiling
     `MAX_L2_INTERVAL = 0.25` (empirical; theoretical max ≈ 0.62),
     L2 norm of IC7-10 against `MAX_L2_TRIAD = 0.025` (theoretical ≈ 0.074),
     and `atan2(20 * ||triads||, ||intervals|| + 1e-12)` normalized against
     π/2 — i.e. the triad-vs-dyad ratio, bounded by construction.
   - All 13 values pass through the standard [-1, 1] normalization; the two
     L2 norms are additionally clipped at +1 (`.min(1.)`).

### (a.7) Distance — the similarity computation

Source: `src/lib.rs` (`FeaturesVersion::feature_weights`,
`VERSION2_WEIGHTS`), `src/playlist.rs`
(`euclidean_distance`, `cosine_distance`, `mahalanobis_distance`,
`mahalanobis_distance_builder`, `variance_based_weight_matrix`,
`closest_to_songs`, `song_to_song`, `dedup_playlist_custom_distance`,
`closest_album_to_group`), `src/song/mod.rs` (`Analysis::distance`).

- **Default distance (V2): weighted Mahalanobis with a fixed diagonal
  matrix** `M = diag(VERSION2_WEIGHTS)`:
  `d(a,b) = sqrt((a-b)^T M (a-b))`. Exact weights (`src/lib.rs`):
  ```
  tempo 0.25
  zcr, centroid_mean, centroid_std, rolloff_mean, rolloff_std,
  flatness_mean, flatness_std, loudness_mean, loudness_std: 1.0 each (9 dims)
  chroma1..chroma13: 3/13 ≈ 0.2308 each (13 dims)
  ```
  Weight budget: tempo 0.25 + timbral/loudness 9.0 + chroma 3.0 = **12.25**.
  Harmony therefore carries 3.0/12.25 = 24.5% of the distance, tempo 2.0%.
  Sanity identity (upstream test `test_distance_metric_features_version`,
  `src/lib.rs`): distance between the all-0 and all-1 23-vectors is
  `sqrt(12.25) = 3.5` exactly (asserted as `3.4999998` in f32).
  V1 used the identity matrix (plain Euclidean); distance all-0/all-1 = √20 ≈ 4.4721.
- **Playlist construction:**
  - `closest_to_songs(seeds, pool, metric)`: build one metric from the seed
    set, sort the whole pool by distance ascending. This is the mode v3 wants.
  - `song_to_song(seeds, pool, metric)`: greedy chain — repeatedly take the
    pool song closest to the last picked song, refitting the metric each step.
    Upstream warns it "has a tendency to go from one style to the other very
    fast, and it can be slow on big libraries" (`src/playlist.rs`). Port as
    an optional ordering mode only; do not make it the default ranker.
  - `dedup_playlist`: two songs are duplicates if same non-empty
    title+artist, or distance < threshold (default **0.05**, Euclidean).
    0.05 is calibrated to unweighted Euclidean space; it is **not** valid
    in weighted-bliss space and must be recalibrated (§(b.5)).
  - `closest_album_to_group`: averages analyses per album, ranks albums by
    Euclidean distance to the seed group's mean vector. Not applicable here
    (no album grouping in the liked set); do not port.
- **Seed-adaptive weighting** (`variance_based_weight_matrix`, `src/playlist.rs:173`):
  given ≥2 seed vectors, per-dim weight = `1/(variance + 1e-6)`,
  rescaled so weights sum to `n` (the dimension count). Stable dims across
  seeds get high weight, variable dims get ~0. This is upstream's own
  "what do my seeds have in common" weighting and is directly applicable to
  Kevin's 650 liked tracks as seeds. Errors if <2 seeds, empty vectors, or
  mismatched lengths.
- **Extended Isolation Forest** (`ForestOptions`, `src/playlist.rs:230`):
  bliss's documented best metric "for measuring the distance to a set of
  songs" (trait docs, `src/playlist.rs:17-22`); single-song distance falls
  back to Euclidean/cosine. The only concrete instantiation in the repo is
  in a test (`src/playlist.rs:1646`): `n_trees = 1000, sample_size = 200,
  max_tree_depth = None, extension_level = 10`, trained on Mozart piano
  concertos with the expectation that Miles Davis tracks rank last.
  No production default is shipped in `library.rs` (grep: no `ForestOptions`
  outside `playlist.rs`). **scikit-learn has no Extended Isolation Forest**
  (`sklearn.ensemble.IsolationForest` is axis-aligned, not extended) — an
  EIF port is a stretch goal, not a drop-in; see §(b.6).
- **Metric learning** (separate repo, `bliss-metric-learning`, README +
  `learn.py`): a human answers odd-one-out triplets (3 songs, pick the least
  similar; "~200 is optimal, but 30-50 also works"), then `learn.py`
  (scipy `minimize`, sklearn KFold) fits a full Mahalanobis matrix `M`
  (parameterized as `L`, `d_M` with `grad_d`) minimizing triplet violations
  `d_M(A,B) < d_M(A,C)` and `d_M(A,B) < d_M(B,C)`, and stores it for
  `blissify playlist --distance mahalanobis`. Thesis reference:
  https://lelele.io/thesis.pdf, chapter 4 (also cited in `playlist.rs:67`
  for the Euclidean default). The in-stack analogue for v3 is learning the
  diagonal weights from Kevin's 650 likes vs 13 negatives (logistic
  regression already exists in v2); a full-matrix triplet fit is a stretch
  goal requiring Kevin to answer triplets.

### (a.8) Decoding and cost (upstream)

- Decoders (`src/song/decoder.rs`, `ffmpeg.rs`, `symphonia.rs`) decode and
  resample to the canonical mono f32 22050 Hz form before analysis; analysis
  of the same canonical samples is decoder-independent by design.
  FFmpeg is default; Symphonia is pure Rust, "~5-10% slower" overall
  (README: 10k files took ~56 min FFmpeg vs ~1 h 5 min Symphonia on the
  author's machine — **0.34 s/song** reference point, author's hardware
  unknown, not a VM guarantee). No decode step is needed in v3: librosa
  already decodes the previews.
- bliss-rs is **not** installable as a Python feature extractor here without
  a Rust toolchain + ffmpeg dev libraries; the Python bindings
  (`bliss-audio` on PyPI, built from
  https://github.com/Polochon-street/bliss-python) were not verified as
  installed on this VM. The port in §(b) is therefore a reimplementation in
  librosa/numpy, not a call into the Rust library. §(c) test B1 is the check
  that the reimplementation matches upstream numbers.

---

## (b) Exact feature to port, mapped onto our stack

### (b.1) Deliverable

New module `bliss_features.py` exposing:

```python
BLISS_DIM = 23
BLISS_NAMES = ["tempo","zcr",
  "centroid_mean","centroid_std","rolloff_mean","rolloff_std",
  "flatness_mean","flatness_std","loudness_mean","loudness_std",
  "chroma1","chroma2","chroma3","chroma4","chroma5","chroma6",
  "chroma7","chroma8","chroma9","chroma10","chroma11","chroma12","chroma13"]
VERSION2_WEIGHTS = [0.25] + [1.0]*9 + [3.0/13.0]*13   # sum = 12.25

def extract_bliss23(y, sr=22050) -> np.ndarray[23]   # y: mono float32, sr 22050
def bliss_distance(a, b, weights=VERSION2_WEIGHTS) -> float
def bliss_score_to_liked(candidate_vec, liked_matrix, weights=...) -> float
```

`extract_bliss23` takes the already-loaded waveform (same `librosa.load`
call as `pipeline.py:63`); it must not re-download or re-decode.
Cache per-track output in `data/bliss23.json` keyed by Spotify ID with fields
`{spotify_id, source: "preview"|"full", n_samples, bliss23: [...], extractor: "bliss23/1.0"}`.
Preview and full-audio vectors for the same track MUST be stored under
different keys/rows and never mixed in one matrix.

### (b.2) Table 1 — per-feature port

| # | bliss dim | Upstream params | librosa / numpy port | Preview-only? | What the preview loses |
|---|-----------|-----------------|----------------------|---------------|------------------------|
| 0 | tempo | aubio SpecFlux+beattrack, win 512 hop 256, median of per-beat BPM, norm 0..206, -1 if none (§a.2) | `librosa.beat.beat_track(y, sr)` (onset-envelope dynamic programming) + `librosa.beat.tempo(onset_envelope)` median as cross-check; norm `2*bpm/206-1`; -1.0 if onset strength max < 0.05 × median onset or no beats | Yes, with loss | Median over 30 s from ~15-70 beats vs ~100-500 over a full track. Bridge/intro tempo changes invisible. Octave errors (half/double) differ between aubio and librosa DP; expect disagreement on ~10-20% of tracks (threshold in §c test T3). |
| 1 | zcr | global crossings/samples, norm 0..1 (§a.3) | `np.mean(np.abs(np.diff(np.signbit(y))))` computed globally, NOT librosa per-frame mean (librosa `zero_crossing_rate` frame mean differs in the 3rd decimal; global is exact) | Yes, near-lossless | None material: ZCR is a stationary timbral stat; 30 s estimates it to ±0.01 normalized on typical tracks (test Z2). |
| 2-3 | centroid mean/std | PVoc win 512 hop 128, norm 0..11025 (§a.4) | `librosa.feature.spectral_centroid(y, sr, n_fft=512, hop_length=128)`; mean+std; norm `2*x/11025-1`. MUST set n_fft/hop explicitly (librosa defaults are 2048/512 and shift values by >0.05 normalized) | Yes, near-lossless | Brightness varies by section (verse vs chorus); preview mean is a section mean, not a song mean. Std over 30 s underestimates full-song std (test S3 quantifies). |
| 4-5 | rolloff mean/std | PVoc win 512 hop 128, 95% energy, norm 0..11025 (§a.4) | `librosa.feature.spectral_rolloff(y, sr, n_fft=512, hop_length=128, roll_percent=0.95)`; mean+std; same norm | Yes, near-lossless | Same section bias as centroid; rolloff is the noisiest of the three on MP3 previews (test S3). |
| 6-7 | flatness mean/std | PVoc win 512 hop 128, geomean/mean, norm 0..1 with std normalized as a value (§a.4) | `librosa.feature.spectral_flatness(y, n_fft=512, hop_length=128)`; mean+std; `2*x-1` for BOTH (do not z-normalize the std) | Yes, near-lossless | Same as centroid. Flatness-std is near -1 for most tracks by construction; verify it is not constant before letting it into the model (test S4). |
| 8-9 | loudness mean/std | 1024-sample chunks, 10*log10 of mean/std of linear energy, floor -90 dB, norm -90..0 (§a.5) | Pure numpy on `y` (no librosa call): chunk 1024, `level=mean(x^2)`, `10*log10(mean)`, `10*log10(std)`, clamp floor 1e-9, norm `(x+90)/90*2-1` | Yes, but NOT comparable to full-audio | Deezer previews are separately encoded/mastered excerpts; a preview loudness value and a full-track loudness value for the same song differ by an unknown offset. Preview-vs-preview comparisons are valid (same chain); preview-vs-full comparisons are invalid and must be blocked by the `source` field (§b.1). Dynamics (std) also compresses in a 30 s excerpt. |
| 10-22 | chroma1-13 | STFT 8192/2205, tuning est., librosa-style filterbank, exp(15x)+L1, interval templates over 12 rotations, time-mean, V2 L2-norms + 3 summaries (§a.6) | `librosa.feature.chroma_stft(y, sr, n_fft=8192, hop_length=2205, tuning=librosa.estimate_tuning(y, sr))`, then implement `interval_features()` in numpy EXACTLY per §(a.6) steps 4-5 (templates copied verbatim from `src/chroma.rs`; do NOT substitute `librosa.feature.chroma_cqt` — v1's `chroma_cqt` is a different transform and is not bliss chroma) | Yes, with the largest loss | Chroma is a time-mean over the excerpt: a 30 s chorus-only preview yields the chorus's interval profile, not the song's. Upstream explicitly warns chroma wants the full song at once. This is the feature most improved by the full-audio variant (§b.4). Also: key is deliberately discarded (transposition-invariant), so chroma1-13 cannot help key/mode — v2's `chroma_hist`/`tonnetz`/key features remain the key signal. |

### (b.3) Preview-only variant (works today) — the default v3 path

1. Input: Deezer 30 s preview MP3, `librosa.load(sr=22050, mono=True)` — identical
   to `pipeline.py:63`.
2. Extract all 23 dims per Table 1. Tempo uses librosa beat tracking; chroma
   uses `chroma_stft` + numpy interval transform.
3. Store with `source="preview"`. All 650 liked tracks and all candidates get
   preview vectors; ranking compares preview-to-preview only.
4. Scorer: `bliss_score_to_liked` = weighted Mahalanobis (§a.7) from the
   candidate vector to each liked vector, aggregated as **distance to the
   10 nearest liked tracks (mean of the 10 smallest distances)** — not
   distance to the global centroid, because Kevin's taste is multi-modal
   (E/R/H buckets; a centroid sits between clusters and penalizes all of them).
   Lower = closer. Report the score alongside, not instead of, the v2 score;
   v3 ranking uses the §(d) harness to decide the blend, it does not assume one.
5. Seed-adaptive variant (port `variance_based_weight_matrix`): compute
   inverse-variance diagonal weights over the 650 liked preview vectors
   (weights sum to 23), then use them in place of `VERSION2_WEIGHTS` as a
   second scorer. Both scorers are evaluated in §(d); the better one ships.
6. Dedup: recalibrate, do not reuse 0.05. Procedure: compute the weighted
   bliss distance between all pairs of liked tracks that are known distinct
   songs; set the dedup threshold to the 0.1th percentile of those distances,
   floored at 0.05 and capped at 0.5, and report the chosen value. If two
   distinct liked tracks fall below it, the threshold is wrong — raise the
   percentile, do not hand-tune per pair.

### (b.4) Full-audio variant (gated on the friend API)

Gate: **no full-audio code runs until the friend API is available and a
`FULL_AUDIO_AVAILABLE` flag is set in the v3 config.** The gate is a config
flag, not a code branch that silently falls back to previews. If the flag is
off, `extract_bliss23` on a full-length path raises; if on, it runs the
identical function on the full waveform.

Differences from preview-only (same code, different input):

| Item | Preview-only | Full-audio |
|------|-------------|------------|
| Input length | 30 s, 661,500 samples @22050 | full track, typically 180-300 s |
| Tempo | median over ~15-70 beats | median over ~100-500 beats; octave-error rate falls; `-1` sentinel near-vanishes except ambient tracks |
| Chroma | section-biased time-mean | true song time-mean; upstream's non-streamed condition satisfied |
| Spectral/loudness std | underestimated (short window) | true song-level std; loudness std becomes informative |
| Runtime/track (budget, §d) | p50 ≤ 2.0 s | p50 ≤ 8.0 s (chroma STFT at 8192/2205 dominates; ~3-10× preview cost) |
| Storage | `source="preview"` | `source="full"`, separate cache rows; never mix with preview rows in one distance computation |

Full-audio additionally enables two things preview-only cannot do at all:
(i) per-segment (e.g. 6 × 30 s) bliss vectors and their std across segments —
a structure feature bliss itself does not compute; (ii) a direct
preview-vs-full consistency measurement per track (same extractor, two
inputs), which is the empirical answer to "what does the preview lose"
— report median |Δ| per dim over ≥50 tracks once full audio exists.

### (b.5) What is NOT ported (explicitly out of scope for v3)

- The Rust library itself, FFmpeg/Symphonia decoders, and the `bliss-audio`
  Python bindings: no Rust toolchain or ffmpeg dev libraries are verified
  present on this VM. Reimplementation only.
- `closest_album_to_group`: no album grouping in this data.
- `song_to_song` greedy chaining as default ranker: port the function
  (15 lines) as an optional playlist-ordering mode, but default ranking is
  `closest_to_songs` semantics. Upstream's own drift warning applies.
- Full-matrix metric learning (`bliss-metric-learning`): needs Kevin to answer
  30-200 odd-one-out triplets. v3 uses the diagonal-weight analogue instead
  (§b.3.5 + existing v2 logistic regression). Triplet survey is listed as
  an unblocker in §(e), not a v3 deliverable.
- Extended Isolation Forest: sklearn has no EIF. Stretch only, §(b.6).

### (b.6) Stretch goals only (do not block v3 on these)

- **EIF:** implement Extended Isolation Forest in numpy/sklearn-equivalent
  code, or vendor a minimal EIF, with the upstream test parameters
  (`n_trees=1000, sample_size=200, extension_level=10`) as the starting point.
  Acceptance: it must beat the Mahalanobis scorer on §(d) by ≥0.02 P@10 or
  it does not ship. `sklearn.ensemble.IsolationForest` is NOT an EIF and
  must not be presented as one; if it is used as a stand-in, label it
  "axis-aligned IsolationForest (not EIF)" in every output.
- **torch / FAISS / Essentia / Demucs:** out of scope per the task brief.
  Essentia would be the natural cross-check for spectral goldens (§c test B2)
  if it is ever installed; it is not required.

### (b.7) VM fit (1.5 shared CPUs, ~6 GB RAM)

- Per-track memory: waveform 30 s float32 ≈ 2.6 MB; chroma STFT (8192/2205)
  ≈ 301 frames × 4097 bins float64 ≈ 9.9 MB transient; full track 300 s
  ≈ 26 MB waveform + ~100 MB STFT transient. All fit in 6 GB with
  single-track processing. Process tracks **sequentially** (no multiprocessing
  pool; 1.5 CPUs make pool overhead exceed gains) and free the STFT per track.
- Throughput reference: upstream reports ~0.34 s/song (FFmpeg, author's
  machine, 10k files / 56 min). Budget for v3 on this VM: preview p50 ≤ 2.0 s/track
  (650 liked + ~200-500 candidates ⇒ ≤ ~40 min for a full rebuild; cached
  thereafter in `data/bliss23.json`). If p50 exceeds 2.0 s, profile the chroma
  STFT first (it is the known hot spot: 8192-point FFT × 301 frames).
- No new heavy dependencies: librosa + numpy + scipy + scikit-learn only
  (all already in the v2 venv `~/workspace/audio-env/bin/python`).

---

## (c) TDD acceptance tests — write FIRST, watch fail, then implement

Location: `tests/test_bliss_features.py`, run with
`~/workspace/audio-env/bin/python -m pytest tests/test_bliss_features.py -v`.
Every test below must FAIL before `bliss_features.py` exists (import error)
and PASS after. Tolerances are stated per test; they are looser than bliss's
own 1e-5 self-consistency tolerance because librosa-STFT ≠ aubio phase
vocoder — 1e-5 across implementations is unachievable and must not be the bar.
**Reference fixture:** `data/s16_mono_22_5kHz.flac` from the bliss-rs clone
(`/tmp/bliss-rs/data/`); the v3 builder copies it into `tests/fixtures/`
at build time (it is not committed to the repo by this spec; the golden
numbers are reproduced verbatim below so the test does not depend on the
fixture file existing).

### (c.1) Unit / synthetic tests (no network, no fixtures)

- **T1 tempo 60 BPM click:** synthetic 60 s click train, 1 beat/s, 22050 Hz.
  `extract_bliss23` tempo dim == -0.416853 ± 0.02 (upstream
  `test_tempo_artificial`, `src/temporal.rs`). Fails if librosa reports
  120 BPM (octave error) — the test then forces an octave-correction rule
  (prefer the BPM closest to the ReccoBeats tempo when the two disagree by
  a factor ≈2; record the rule, do not silently pick librosa's value).
- **T2 tempo silence:** 10 s of zeros → tempo dim == -1.0 exactly, zcr == -1.0
  exactly, centroid/rolloff/flatness/loudness == (-1,-1) each (upstream
  boundary tests, `src/temporal.rs`, `src/timbral.rs`, `src/misc.rs`).
- **T3 tempo cross-check on liked tracks:** for 30 liked tracks with both
  ReccoBeats tempo and preview audio, Pearson r between bliss-port tempo
  (un-normalized back to BPM) and ReccoBeats tempo ≥ 0.80, and median
  |ΔBPM| ≤ 5 BPM after octave correction (halve/double to nearest). If r < 0.80,
  tempo is demoted: weight 0.25 → 0.0 in the v3 scorer and the demotion is
  reported, not hidden.
- **Z1 ZCR exactness:** on 5 s of alternating ±1 samples, zcr dim ==
  0.9980469 ± 0.001 (upstream `test_zcr_boundaries`). On silence, -1.0.
- **Z2 ZCR preview stability:** for 20 liked tracks, split the 30 s preview
  into two 15 s halves; |zcr(first) - zcr(second)| ≤ 0.05 normalized on
  ≥18/20 tracks. (Quantifies "near-lossless" in Table 1; failure means ZCR
  is section-dependent and its weight must be revisited.)
- **S1 spectral goldens on pure tone:** 10 s sine at 11080 Hz, 22050 Hz.
  centroid == (0.97266, -0.9609926) ± 0.02 each; rolloff ==
  (0.9967681, -0.99615175) ± 0.02 each (upstream
  `test_spectral_centroid_boundaries`, `test_spectral_roll_off_boundaries`).
- **S2 spectral flatness on white noise:** 10 s white noise (seed 0).
  flatness == (0.5785303, -0.9426308) ± 0.05 each (upstream
  `test_spectral_flatness_boundaries`). Failure here means the flatness
  implementation (geometric mean) is wrong, not that noise is noisy.
- **S3 spectral section bias:** for 20 liked tracks, |centroid_mean(first 15 s)
  - centroid_mean(last 15 s)| ≤ 0.10 normalized on ≥15/20 tracks; same for
  rolloff_mean. Report the actual counts; if <15/20 pass, the spec's
  "section bias" warning is upgraded to a measured bias and the full-audio
  variant is marked higher priority.
- **S4 flatness-std non-degeneracy:** over the 650 liked preview vectors,
  flatness_std has std > 0.01 across tracks. If ≤0.01 it is a near-constant
  and must be excluded from the model input (it carries no signal).
- **L1 loudness exactness:** constant +1.0 signal, 1024-sample chunks →
  loudness == (1.0, -1.0) ± 1e-6; silence → (-1.0, -1.0) ± 1e-6
  (upstream `test_loudness_boundaries`). 10*log10(mean(x²)) on a sine of
  amplitude 0.5 → mean dim == 2*(10*log10(0.125)+90)/90-1 ≈ -0.80 ± 0.01.
- **C1 chroma interval transform on synthetic triads:** feed a synthetic
  C-major triad (C4+E4+G4 sines, 5 s) through `interval_features()`:
  chroma7 (major triad) > chroma8 (minor), chroma9 (diminished),
  chroma10 (augmented) by ≥ 0.10 normalized each; and a C-minor triad
  reverses the major/minor order (chroma8 > chroma7 by ≥ 0.10). Uses the
  `data/chroma/*.ogg` fixtures in the bliss clone as the reference set if
  available (Cmaj, Cmin, Cdim, Caug, triads); otherwise the synthetic sines.
- **C2 chroma transposition invariance:** C-major triad vs F#-major triad
  (6 semitones up): max |Δ| over chroma1-13 ≤ 0.05 normalized. This is the
  defining property of bliss chroma (§a.6 step 4); failure means the
  template rotation is implemented wrong. (Key IS lost: also assert that
  the two triads' raw 12-bin chromagrams differ by ≥ 0.5 in at least one bin
  while their chroma1-13 agree — proving the invariance is in the transform,
  not in the input.)
- **C3 chroma output range/shape:** for any input, all 13 chroma dims lie
  in [-1, 1]; chroma11, chroma12 ≤ 1.0 (clipping in `get_values`);
  chroma13 in [-1, 1].
- **D1 distance identity:** `bliss_distance(zeros(23), ones(23)) == 3.5 ± 1e-6`
  and `bliss_distance(a, a) == 0.0` exactly; `bliss_distance` is symmetric
  to 1e-6 on 10 random pairs (upstream `test_distance_metric_features_version`).
- **D2 weights sum:** `sum(VERSION2_WEIGHTS) == 12.25 ± 1e-6`;
  chroma block sums to 3.0, tempo weight is 0.25 (not 1.0).
- **D3 variance weights:** `variance_based_weight_matrix` port on 3 seeds
  from upstream's test (`[1,0,1],[1,100,1],[1,200,1]`) gives
  w[0] > w[1], w[2] > w[1], off-diagonals 0, weights sum to 3; raises on
  <2 seeds. (Upstream `test_variance_based_weight_matrix_*`.)
- **D4 dedup calibration:** threshold from §(b.3.6) is in [0.05, 0.5] and
  zero distinct liked-track pairs fall below it (computed on the 650 liked
  preview vectors).

### (c.2) Golden-file test (cross-implementation check)

- **B1 golden vector:** `tests/fixtures/s16_mono_22_5kHz.flac`
  (copied from `/tmp/bliss-rs/data/s16_mono_22_5kHz.flac`) through
  `extract_bliss23` must equal the upstream V2 golden vector
  (`src/song/mod.rs`, `SONG_AND_EXPECTED_ANALYSIS`) within **±0.05 per dim**
  on ≥20/23 dims, and within ±0.10 on all 23 dims:
  ```
  [0.3846389, -0.849141, -0.75481045, -0.8790748, -0.63258266,
   -0.7258959, -0.7757379, -0.8146726, 0.2716726, 0.25779057,
   -0.34292513, -0.62803423, -0.28095096, 0.08686459, 0.24446082,
   -0.5723257, 0.23292065, 0.19981146, -0.58594406, -0.06784296,
   -0.06000763, -0.58485717, -0.07880378]
  ```
  Per-descriptor sub-tolerances (from upstream's own per-descriptor tests):
  tempo ±0.05 of 0.3846389 (note upstream's own `test_tempo_real` gives
  0.378605 via a different chunking — the ±0.05 band covers both);
  zcr ±0.02 of -0.849141; centroid ±0.02; rolloff ±0.02; flatness ±0.02;
  loudness ±0.02; chroma ±0.05. If B1 fails, the port is wrong in the
  failing descriptor — fix that descriptor, do not widen the tolerance.
  This test is the only check that the port matches upstream numbers at all;
  it must be run and its per-dim deltas reported, not skipped as "close enough".
- **B2 (stretch, only if Essentia is installed):** same fixture through
  Essentia spectral descriptors; centroid raw mean within 2.5% of 1354.2273 Hz
  (upstream's Essentia cross-check, §a.4). Skip with an explicit skip message
  if Essentia is absent; never fake it.

### (c.3) Full-audio variant tests (run only when the friend API flag is on)

- **F1 same-extractor consistency:** for ≥50 tracks with both preview and
  full audio, median |Δ| per dim (preview vs full, same extractor) is
  reported for all 23 dims. Pass bar: median |Δ| ≤ 0.10 normalized on
  ≥18/23 dims. Dims failing the bar are named in the v3 report as
  "preview-unreliable" and down-weighted 50% in the preview-only scorer.
- **F2 tempo stability:** full-audio tempo vs preview tempo agree within
  ±5 BPM (after octave correction) on ≥80% of the 50 tracks; the -1 sentinel
  rate on full audio is ≤2% (vs whatever the preview rate is — report both).
- **F3 chroma stability:** median |Δ| over chroma1-13 (preview vs full)
  ≤ 0.15 normalized; if larger, chroma is confirmed as the most
  preview-sensitive bliss block and the full-audio variant is required for
  any claim about harmony.
- **F4 runtime:** full-audio extraction p50 ≤ 8.0 s/track on this VM over
  the 50 tracks; preview p50 ≤ 2.0 s/track over the same 50 (test P1).
- **F5 no mixing:** a distance matrix built from mixed preview+full rows
  raises an error (the `source` field is enforced); the test constructs the
  mixed input deliberately and asserts the raise.

### (c.4) What "watch it fail" means here

Before writing `bliss_features.py`, create `tests/test_bliss_features.py`
with T1-T3, Z1-Z2, S1-S4, L1, C1-C3, D1-D4, B1 and run it: **all tests fail
with ModuleNotFoundError/ImportError**. Record that run (command + failure
count) in the v3 build log. Implement one descriptor at a time in the order
zcr → loudness → spectral → tempo → chroma → distance, re-running after each.
A test that passes before its descriptor is implemented is a broken test —
investigate, do not celebrate.

---

## (d) Eval-harness metric proving it beats the v6 6/10 baseline

Extend `evaluate.py` (do not replace it). The harness already ranks a pool of
held-out likes + negatives + 200 distractors (seed 42) and reports P@10,
NDCG@10, mean rank of held-out likes, mean score/rank of negatives
(`evaluate.py`, `rank_metrics`). v3 adds one method and one control:

### (d.1) Pool and protocol (identical for v2, v6, and bliss scorers)

- **Held-out likes:** TEST_N = 130 (most recent 20% by list position, as in
  `evaluate.py`). Relevant = 1 iff held-out like.
- **Negatives:** **all 13 confirmed negatives** — the 9 dropped v5 tracks
  already in `negatives_features.json` PLUS the 4 v6 thumbs-down
  ("prove it" — 21 Savage, Summer Walker; "Akina Speed Star" — Xavier Wulf;
  "Alone" — Night Lovell; "Cha Cha" — Freddie Dredd; all hip-hop/trap bucket).
  `evaluate.py` currently loads only the 9 v5 negatives; v3 must load the 4
  v6 tracks the same way (`negatives_features.py` pattern) and assert the
  negative count is 13 before ranking. If any negative lacks a preview,
  report it as excluded and reduce the denominator — do not silently rank 9.
- **Distractors:** 200 pool tracks, seed 42, same sample for every method.
- **Bliss features required:** only items WITH bliss23 vectors are ranked;
  excluded counts per method are reported (same apples-to-apples rule as
  `evaluate.py`). A method that ranks fewer items because of missing previews
  does not get to hide that.
- **Scorers compared on the same pool:** v2 (GMM+LR), v6b (per-bucket),
  v6g (global), bliss-fixed (VERSION2_WEIGHTS, 10-nearest-liked mean distance),
  bliss-adaptive (inverse-variance weights, §b.3.5), bliss+v2 blend
  (z-scored mean of v2 score and -bliss distance; blend weight fitted on
  train only, never on the 130 held-out). Report all six rows.

### (d.2) Pass bars — ALL must hold for the claim "v3 beats v6 6/10"

| # | Metric | Bar | Rationale |
|---|--------|-----|-----------|
| M1 | P@10 (held-out likes) | **≥ 0.70** for the shipping scorer, and ≥ v6b P@10 + 0.10 | 0.70 = 7/10 top-10 are held-out likes, the offline analogue of beating the 6/10 human keep rate by 1 track with n=130 held-out instead of n=10 |
| M2 | NDCG@10 | ≥ v6b NDCG@10 + 0.05 | Ranking quality, not just top-10 hit count |
| M3 | Mean rank of held-out likes | ≤ v6b mean rank − 20 positions | Held-out likes must move up the full ranking, not just into the top 10 |
| M4 | Negatives in top 10 | **≤ 1 of 13** in the top 10 | v6's failure mode was recommending tracks Kevin thumbs-down; a scorer that ranks negatives highly has not beaten v6 regardless of M1-M3 |
| M5 | Mean rank of the 13 negatives | **≥ 200** (pool ≈ 343 items; random ≈ 172) | Negatives pushed to the bottom half on average |
| M6 | Mean rank of the 4 H-bucket negatives | **≥ 200** | All 4 v6 thumbs-down are hip-hop/trap; the H bucket is where v6 failed |
| M7 | Bliss ablation | Removing the 23 bliss dims from the blend drops P@10 by **≥ 0.02** | Proves the bliss block itself (not the blend or retraining) earns its place; a block with ΔP@10 < 0.02 ships as dead weight and is cut |
| M8 | Runtime | Preview extraction p50 ≤ 2.0 s/track; full rebuild of 650 liked + pool ≤ 40 min wall on this VM | §(b.7); a scorer that cannot be rebuilt is not shippable |
| M9 | Human listen gate | Kevin listens to the shipping scorer's top 10 and keeps **≥ 7/10** on second listen | The only direct comparison to the v6 6/10 number. NOTE (statistics): with n=10, P(≥7/10 \| true p=0.6) ≈ 0.38, so M9 alone cannot prove superiority — M1-M6 are the proof, M9 is the sanity check. If M1-M6 pass and M9 gives 6/10, report both and do not claim a win; if M9 gives ≤5/10, the offline metrics are suspect and the scorer does not ship. |

### (d.3) Reporting

Output `evaluation_v3.json` with: per-method P@10, NDCG@10, mean rank
held-out, mean rank/score of all 13 negatives, mean rank of the 4 H negatives,
top-10 kind lists, excluded counts per method, ablation ΔP@10 per bliss block
(tempo / timbral / loudness / chroma dropped one at a time), and the per-dim
preview-vs-full deltas from F1 once available. Keep `evaluation_v2.json`
untouched for comparison.

---

## (e) What it cannot do on our data, and what would unblock it

### (e.1) Cannot do — preview-only (today)

1. **Key/mode.** Bliss chroma is transposition-invariant by construction
   (§a.6 step 4): it cannot distinguish C major from F# major, or major from
   minor except through triad-quality proportions. Do not use chroma1-13 as a
   key signal. The v2 key features (`rb_key_sin/cos`, `lb_is_minor`,
   chroma_hist, tonnetz) remain the only key/mode signal. (Verified: C2 test.)
2. **Full-song harmony and tempo.** Chroma is a time-mean and tempo a median;
   both are computed over a 30 s excerpt of unknown position in the track
   (Deezer does not guarantee which 30 s). A verse-only preview and a
   chorus-only preview of the same song give different chroma/tempo. The
   magnitude is measured by F1/F3, not assumed.
3. **Loudness comparability across sources.** Preview loudness is comparable
   preview-to-preview only. Any claim of the form "this candidate is louder
   than Kevin's liked tracks" is valid only within one source. (§b.2, loudness row.)
4. **Tempo confidence.** Upstream computes no usable confidence (TODO in
   `src/temporal.rs`); the port has none either. Onset-poor music (ambient,
   sparse piano, drone) gets unreliable tempo in both implementations, and
   the only mitigation is the 0.25 weight plus the T3 cross-check against
   ReccoBeats tempo. If T3 fails (r < 0.80), tempo is demoted to weight 0.
5. **Vocals, lyrics, genre, valence, danceability.** Bliss has no MFCCs, no
   embeddings, no vocal detector, no genre classifier. It cannot tell rap
   from sung vocals except weakly via ZCR/speechiness-adjacent spectral
   stats — and the 4 confirmed v6 negatives are all hip-hop/trap, i.e.
   exactly the bucket where a vocal/lyrical signal would matter most.
   ReccoBeats speechiness/instrumentalness remain the vocal signal.
6. **Exact numeric equivalence to Rust bliss.** librosa STFT ≠ aubio phase
   vocoder; librosa beat tracking ≠ aubio SpecFlux beattracking. B1's
   ±0.05 tolerance is the honesty bound: the port is a reimplementation
   that agrees with upstream to ~0.05 normalized per dim, not a bit-exact
   clone. Any v3 claim that depends on finer precision than ±0.05 per dim
   is unsupported.
7. **Taste, not audio.** Bliss distance measures audio similarity. Kevin's
   keep/drop decisions also involve artist familiarity, lyrics, nostalgia,
   and context the audio cannot see. The 13 negatives are the only taste
   supervision in the harness; 13 labels cannot fit a full metric matrix
   (see §e.2.3).

### (e.2) What would unblock each limit

| Limit | Unblocker | Owner / status at research time |
|-------|-----------|---------------------------------|
| Full-song chroma/tempo/loudness-std | Friend API supplying full songs; flip `FULL_AUDIO_AVAILABLE` and run §(c.3) F1-F5 | Friend API sign-in pending, not available |
| Preview-vs-full loss quantified | Same as above + F1 over ≥50 tracks | Blocked on friend API |
| Tempo confidence | Implement onset-strength-based confidence (e.g. tempogram peak prominence) in the port; upstream has no reference implementation, so this is new work, not a port | Doable in v3.1; not required for v3 |
| Key/mode from audio | Nothing in bliss; keep v2 key features. A dedicated key estimator (e.g. librosa piptrack + Krumhansl-Schmuckler on the raw 12-bin chromagram before the interval transform) is a separate feature, not a bliss port | Separate spec, not this one |
| Vocal/lyrical signal for the H bucket | ReccoBeats speechiness (already in v2) + the 13 negatives as LR supervision; longer term, a vocal-presence feature | Already partially covered by v2 |
| Learned (not fixed) distance weights | Kevin answers 30-200 odd-one-out triplets (bliss-metric-learning README: "~200 is optimal, but 30-50 also works"); fit diagonal or full Mahalanobis by triplet loss (`learn.py`: scipy minimize) | Needs Kevin's time; the v3 analogue (inverse-variance weights + existing LR) ships first |
| Bit-exact bliss numbers | Install Rust toolchain + ffmpeg dev libraries on the VM, build bliss-rs, run `examples/analyze.rs` on the same previews to produce golden vectors; or install `bliss-audio` Python bindings | Not verified present on this VM; would make B1 a ±1e-5 test instead of ±0.05 |
| More negative supervision | More thumbs-down labels from Kevin's listening (each v3 playlist round adds labels) | Ongoing; 13 today |

### (e.3) Risks to state in the v3 report

- **Preview position unknown:** if Deezer previews are systematically the
  chorus (or systematically the intro), all preview-derived chroma/tempo
  features share that bias and the bias is invisible within preview-only
  data. F1 is the only detection; until it runs, treat preview chroma as
  "chorus-biased until proven otherwise" in any user-facing summary.
- **Weight transfer:** VERSION2_WEIGHTS were tuned by the bliss author on
  their own library (the Mozart-vs-Miles test in `playlist.rs:1646` is the
  only in-repo evidence). They are a starting point for Kevin's taste, not
  a proven optimum. The §(d) ablation (M7) and the adaptive weights (§b.3.5)
  are the checks; if fixed weights underperform adaptive weights by ≥0.03
  P@10, ship adaptive.
- **Small-negative statistics:** M4-M6 rest on 13 negatives (4 in H).
  One negative moving 50 ranks changes M5 materially. Report per-negative
  ranks in `evaluation_v3.json` so a single-track swing is visible.

---

## Sources

- bliss-rs repo, commit `0de922d`, https://github.com/Polochon-street/bliss-rs
  - `README.md` — overview, decoders, metric learning, 10k-file timing
    (56 min FFmpeg / ~65 min Symphonia), thesis link https://lelele.io/thesis.pdf
  - `Cargo.toml` — version 0.13.0, crate `bliss-audio`
  - `src/lib.rs` — `VERSION2_WEIGHTS`, `FeaturesVersion`, distance-metric
    tests (`3.4999998` for all-0 vs all-1 V2, `4.47213595` V1)
  - `src/song/mod.rs` — `Song::analyze_with_options` (5 threaded descriptors,
    concatenation order), `AnalysisIndex` (23 dims), golden V2 vector for
    `data/s16_mono_22_5kHz.flac`, V1 golden vector
  - `src/temporal.rs` — `BPMDesc` (win 512/hop 256, median BPM, norm 0..206,
    -1 sentinel, confidence TODO); tests: 60 BPM → -0.416853, ~192 BPM → 0.86
  - `src/timbral.rs` — `SpectralDesc` (PVoc win 512/hop 128, centroid/rolloff
    95%/flatness, mean+std), `ZeroCrossingRateDesc`; Essentia cross-check
    numbers in test comments; tone/noise goldens
  - `src/misc.rs` — `LoudnessDesc` (1024 chunks, 10*log10, -90 dB floor)
  - `src/chroma.rs` — `ChromaDesc` (STFT 8192/2205, tuning via pip_track,
    librosa-style filterbank, exp(15x), interval templates, V2 L2-norms +
    3 summaries, ceilings 0.25/0.025, atan2 ratio); cites AudioLabs Erlangen
    FMP notebooks and the ICMC-SMC-2014 interval-features paper
  - `src/utils.rs` — `Normalize` (2*(x-min)/(max-min)-1), `stft` (Hann,
    reflect pad), `geometric_mean`, `number_crossings` (Essentia)
  - `src/aubio.rs` — `Tempo` (SpecFlux, PeakPicker threshold 0.3, BeatTracking,
    ~6 s window), `spectral_rolloff` (95%), `PVoc`/`SpecDesc`
  - `src/playlist.rs` — euclidean/cosine/Mahalanobis distances,
    `variance_based_weight_matrix`, `closest_to_songs`, `song_to_song`,
    `dedup_playlist` (default threshold 0.05), `closest_album_to_group`,
    EIF `ForestOptions` test instantiation (n_trees 1000, sample_size 200,
    extension_level 10)
  - `src/song/decoder.rs`, `src/song/decoder/ffmpeg.rs`,
    `src/song/decoder/symphonia.rs` — canonical mono f32 22050 Hz input
- bliss-metric-learning, https://github.com/Polochon-street/bliss-metric-learning
  - `README.md` — triplet survey (30-50 good, ~200 optimal), Mahalanobis
    learning, `learn.py` (scipy minimize, triplet loss)
  - `learn.py` — `d_metric`, triplet loading from sqlite
- Local (read-only, for mapping): `~/workspace/audio-pipeline/pipeline.py`
  (librosa.load sr=22050, beat_track, chroma_cqt), `features_v2.py`
  (ReccoBeats + deep librosa, `data/` caches), `scoring.py`
  (`feat_vector`, `V2Model`, v6 profile), `evaluate.py` (P@10/NDCG@10 harness,
  TEST_N=130, 200 distractors seed 42, 9 v5 negatives)

## What was NOT verified

- Whether a Rust toolchain / cargo / ffmpeg dev libraries exist on this VM
  (not checked; the port assumes they are absent and reimplements in Python).
- Whether the `bliss-audio` PyPI bindings install or run here (not attempted).
- The exact Deezer preview window position (intro vs chorus) for any track.
- Upstream's 0.34 s/song timing on this VM's hardware (author's machine, unknown specs).
- That librosa's `beat_track` on Kevin's actual previews meets T3 (r ≥ 0.80)
  — that is a test to run, not a fact asserted here.
