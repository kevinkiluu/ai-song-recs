# Embeat porting spec — v3 (ai-song-recs)

- **Upstream:** Embeat, item-to-item music recommendation (Spotify track ID / ISRC in, similar songs out)
- **Date researched:** 2026-11-22. Upstream commit `e3a32a64003f3ae1c7118ac3bbc576e56bcec9c7` (31 commits, created 2026-06-26, 550 stars at time of reading).
- **Spec target:** v3 build = v2 pipeline + the ported features below. This spec is architecture-only.
- **Local state at spec time:** v2 build agent still running. `features_v2.json`, `model_v2.json`, `model_v2.npz`, `evaluation_v2.json`, `data/pool_features.json`, `data/negatives_features.json` do NOT exist yet on disk (checked 2026-11-22). All v2 interface claims below are read from v2 *source* (`model_v2.py`, `scoring.py`, `evaluate.py`, `recommend_v2.py`, `features_v2.py`), not from v2 outputs. If a v2 interface changes before the v3 build, the build must re-ground against the landed files and record the delta in the build log.
- **Baseline to beat:** v6 playlist keep rate 6/10 on second listen (2026-11-21). 13 confirmed negatives: 9 dropped v5 tracks + 4 v6 thumbs-down (`prove it` — 21 Savage, Summer Walker; `Akina Speed Star` — Xavier Wulf; `Alone` — Night Lovell; `Cha Cha` — Freddie Dredd), all hip-hop/trap bucket.

## 0. Repo identity correction (verified)

The task brief named the repo `github.com/Embeat-dev/Embeat`. **That path is wrong.**

- `git clone https://github.com/Embeat-dev/Embeat.git` fails: `fatal: could not read Username for 'https://github.com'` (auth prompt for a nonexistent/private path). `git ls-remote` on the same URL fails identically.
- The real repo is **`github.com/gdstudio-org/Embeat`** (public, main branch). Cloned successfully to `/tmp/Embeat` on 2026-11-22. All file paths below are relative to that clone unless stated otherwise.

## 0.1 License boundary — read before building

Upstream `README.md` §License (lines 408–413) states a split license:

| Scope | License |
|---|---|
| Code, Model Weights | MIT |
| Datasets, Database | CC-BY-NC 4.0 |

The repo's single `LICENSE` file on disk contains the full **CC BY-NC 4.0** text (408 lines); no MIT license text file exists in the repo. The MIT claim rests only on the README table.

**Rules for the v3 build:**

1. **Port ARCHITECTURE ONLY.** Re-implement the mechanisms described in §(a) in our own code against our own data. Do not copy Embeat code files, do not download the 45M-track dataset (`huggingface.co/datasets/GD-Studio/embeat_45m_spotify_tracks`), do not download the Qdrant databases (Google Drive / Baidu links in the README), and do not download `spotify_playlists.txt` or any playlist corpus.
2. **Do NOT vendor `checkpoints/EmbeatMLP/model.pt`** (1.6 MB, present in the clone) until the MIT-vs-CC-BY-NC contradiction is resolved with the upstream author or legal review. This spec's port does not need the weights: it replaces the trained encoder with a PCA embedding (§b.1). If a future build wants the weights, it must first record which license the weights are actually under.
3. Attribute in code: `Portions of the retrieval architecture are adapted from Embeat (github.com/gdstudio-org/Embeat), code MIT per its README; no Embeat data used.`
4. Every dataset-derived artifact in v3 (liked tracks, Deezer previews, ReccoBeats features, genre-playlist labels) is ours or a public API response, not Embeat data. The CC BY-NC 4.0 datasets stay untouched.

## (a) How the upstream mechanism works

Sources are the cloned repo at commit `e3a32a6` plus the GitHub-rendered README read 2026-11-22. Every claim cites its file.

### (a.1) Two-stage system, two models

`README.md` §System Architecture + `infer/Embeat.py` (`EmbeatDatabase.search_entry`, lines 1223–1361):

- **Offline:** train two models, encode all tracks to 64-dim vectors, index in Qdrant.
- **Online:** given one seed track, run five recall channels in parallel, merge by score, output top-K (default `top_k=20`).
- **Scale claimed upstream:** 45M tracks, 1.88M playlists; 30–200 ms per query on a 2 GB+ VPS (`README.md` Introduction). These latency/scale numbers are upstream claims on upstream hardware, not measurements on our VM.

### (a.2) EmbeatMLP — acoustic encoder

`train/model.py` (`EmbeatMLP`, `EmbeatMLPConfig`), `train/dataset.py`, `train/loss.py`, `train/sampler.py`, `train/train.py`, `infer/infer.py`.

- **Input.** Discrete: key (vocab 13: -1..11 → 0..12), mode (vocab 3), time signature (vocab 6: 3..7 → 1..5, 0 = other), tempo (vocab 12: `tempo/10`, halved if ≥170 BPM, bucketed 1..11, 0 = unknown). Continuous: 7 Spotify audio features, each centered at 0.5 (`build_dense_vector`, `train/dataset.py`): danceability, energy, valence, acousticness, liveness, speechiness, instrumentalness. No waveform is used; the encoder consumes Spotify-computed features only.
- **Architecture.** Dual tower: discrete embeddings (key 8-d, mode 4-d, time-signature 4-d, tempo 4-d → concat 20-d) through Linear 20→64→64 with BatchNorm1d + PReLU; acoustic tower Linear 7→64→64, same activations. Fusion concat 128-d → backbone Linear 128→256→256→64 (BatchNorm1d + PReLU, dropout 0.0). Output is L2-normalized 64-d (`functional.normalize`, `train/model.py` `encode`).
- **Training.** Masked InfoNCE, temperature τ = 0.07 in `MaskedInfoNCEConfig` (`train/loss.py:23`) but τ = 0.05 is the CLI default in `train/train.py:65` — the two defaults disagree; the shipped checkpoint's τ is not recorded in the repo. Batch size 4096, `max_steps` default 200 (`train/train.py`), README claims convergence in ~330 steps. AdamW, lr 1e-4, weight decay 1e-3. Requires PyTorch ≥2.6 and CUDA ≥12 for training (`README.md` Requirements).
- **Positive-pair rule** (`PairSamplerConfig`, `train/sampler.py:23–103`) — this is the most portable idea in the repo: 99% of positives are sampled from the **same album** as the anchor, 1% from same-genre near-acoustic neighbours. Hard constraints on a positive: same mode (relative major/minor allowed), key close (circle-of-fifths distance ≤2, chromatic distance ≤1), tempo-index difference ≤1, same time signature. Dense caps: |Δvalence| ≤ 0.20, |Δenergy| ≤ 0.20, |Δdanceability| ≤ 0.30, |Δacousticness| ≤ 0.30, |Δliveness| ≤ 0.20, |Δspeechiness| ≤ 0.05, weighted L2 ≤ 0.30 (weights: valence/energy/danceability/speechiness 1.0, acousticness/liveness 0.5). Negatives are masked to exclude same-album and same-ISRC pairs. Popularity bias on positives: weight ∝ popularity^0.5, clipped to [0.10, 1.00]; ranking bonuses +0.08 same album, +0.04 same artist, +0.02 same genre. Speechiness ≥ 0.20 (raw) is excluded globally to drop talk-only audio.
- **Inference.** `infer/infer.py`: `infer(sample_a, sample_b)` loads the checkpoint, encodes both, returns cosine rescaled to [0, 1] as `0.5 * (cos + 1)`. Example in README: two Jay Chou tracks score 0.6944.

### (a.3) Track2Vec — playlist collaborative filtering

`train/train_track2vec.py`, `infer/Embeat.py` `search_related_track_record` (lines 867–902) and `batch_most_similar` (839–866).

- gensim `Word2Vec`, skip-gram (`sg=1`), playlists as sentences, tracks as words. Hyperparameters: `vector_size=64`, `min_count=10`, `window=100`, `negative=20`, `sample=1e-3`, `ns_exponent=-0.5`, `seed=616`, `epochs=60`, `workers=cpu_count`.
- Training data: 1.88M Spotify playlists; vocabulary 1.09M tracks (`README.md` Model Details).
- Retrieval: `batch_most_similar` does exact cosine over the full vocab matrix in batches of 1024 (NumPy matmul, `argpartition` top-N). For a seed, all ISRC-linked variants of the seed are pooled as seeds, per-track best scores are kept, and candidates below `min_related_track_score = 0.75` are dropped (`EMBEAT_MIN_RELATED_TRACK_SCORE`, `.env.example`). If no ISRC seed exists, falls back to `wv.most_similar(seed_id)`.

### (a.4) Genre and artist-relation data (EveryNoise)

`data/everynoise_genre_map.py`, `data/everynoise_related_artist.py`, `data/everynoise_unk_artist_genre.py`, `data/refine_datasets_genre.py`, `infer/EmbeatUtils.py`.

- 6,291 micro-genre tags covering >2M artists (`README.md`), scraped from EveryNoise (`everynoise.com/everynoise1d.html`, `engenremap-###.html`). Each genre has a 2-D map coordinate (`norm_x`, `norm_y`); each artist maps to genre indices (`engenremap.json`).
- Related-artist graph: scraped per artist from `everynoise.com/api/canon/<artist_id>` (`data/everynoise_related_artist.py`), stored as `related_artist_idxs` on the track payload. Artists missing from the map get a patch file `artist_genre_idx_patch.json`.
- **Track-genre estimation** (`get_track_genre_idx_from_related_track`, lines 615–652; `get_track_genre_idx_from_similar`, lines 653–685): the seed's own artist genre is only a fallback. The track genre is voted from the related-track candidates (or similar candidates) with weight `1 / sqrt(rank + 1)` per vote; the top genre overrides the artist genre only if it outscores the runner-up by `track_genre_score_ratio = 2.0`. This is a deliberate design choice: a track's genre is inferred from what it co-occurs with / sounds like, not from its artist's label.

### (a.5) Qdrant vector search and indexing

`infer/hf_to_qdrant.py`, `infer/Embeat.py` `build_qdrant_index` (163–230), `get_collection_version` (233–253), `.env.example`, `requirements.txt`.

- Collection `spotify_tracks`, vector size 64, distance **cosine**, datatype **uint8 by default** (quantized `round((x + 1) * 127.5)`, `quantize_to_uint8`, lines 276–285) or float32. HNSW `m=8`, `ef_construct=200`. Notably the indexer default is `hnsw_indexing_threshold = 0` with the code comment `0 = no HNSW indexing` — i.e. even upstream, the shipped index leans on brute-force-friendly settings at import time, not on HNSW.
- Payload whitelist (`PAYLOAD_FIELD_WHITELIST`, lines 24–35): `track_id, track_name, popularity, artist_name, artist_idx, artist_genres, artist_genre_idx, related_artist_idxs, album_name, isrc`. Upsert batch 512, embed batch 4096.
- Payload indexes built at startup: `artist_genre_idx` (int), `artist_idx` (int), `popularity` (float), `isrc` (keyword), plus optional full-text `track_name`/`artist_name` indexes for name search. Low-RAM mode moves payload/vectors on disk.
- Query paths used by the recall channels: `client.query_points` (vector search with payload filter), `client.scroll` (filtered scan ordered by `popularity` DESC), `client.retrieve` (by point id). Qdrant ≥1.18 required (`README.md`, `requirements.txt` pins `qdrant-client>=1.18.0,<1.19.0`).
- Collection versions v1/v2/v3 are detected by payload schema (`artist_idx` absent → v1; `isrc` absent → v2; else v3). ISRC dedup is skipped on v1/v2.

### (a.6) The five recall channels and the merge

`infer/Embeat.py` `search_entry` (1223–1361) orchestrates; per-channel candidate limits are functions of `top_k`:

| Channel | Upstream function | Candidate limit (top_k=20 upstream; top_k=10 in v3) | Mechanism |
|---|---|---|---|
| 1. Acoustic similarity (`similar`) | `search_vector_similar_record` (686–720) | `min(top_k*10, 512)` → 200 upstream / **100 in v3** | `query_points` with seed EmbeatMLP vector, filtered to `artist_genre_idx ∈ seed genres`; drop records with empty `artist_genres` |
| 2. Same-genre popular (`popular`) | `search_genre_popular_record` (722–779) | `min(top_k*2, 128)` → 40 / **20** | `scroll` ordered by `popularity` DESC within seed genre(s), then re-ranked by query-vector cosine over that id set |
| 3. Same artist (`same_artist`) | `search_artist_popular_record` (781–807) | `min(top_k*2, 128)` → 40 / **20** | `scroll` where `artist_idx == seed`, ordered by `popularity` DESC, shuffled before filtering |
| 4. Related artists (`related_artist`) | `search_related_artist_record` (809–837) | `min(top_k*2, 128)` → 40 / **20** | `query_points` filtered to `artist_idx ∈ related_artist_idxs` (top 5 if ≥10 related, else all), results re-sorted by `popularity` DESC |
| 5. Playlist CF (`related_track`) | `search_related_track_record` (867–902) | `min(top_k*5, 128)` → 100 / **50** | Track2Vec `most_similar`, ISRC-pooled seeds, score ≥ 0.75 |

Channel order in `search_entry`: (5) runs first because its candidates vote the seed's *track* genre (§a.4); then (1) uses the voted track genre as its filter; then (2)–(4); then merge.

**Per-channel filters** (`filter_similar_candidates` 904–975, `filter_others_candidates` 977–1024): drop the seed itself; drop same (title, artist); drop duplicate titles; drop titles starting with the seed title + space; drop any title containing `remix`; drop candidates with `popularity < min(seed popularity, 0.1)`; channel 1 additionally requires the candidate's `artist_genre_idx` to equal the voted track genre and drops unknown-genre (`artist_genre_idx == 0`) candidates; same-artist count in channel 1 is capped at `max(1, int(top_k * 0.15))`; near-tie cosine scores (Δ < 1e-5) are broken in favour of higher popularity.

**Merge** (`merge_result_by_score`, 1090–1221): each channel's list is first ISRC-deduped (`filter_by_isrc`, 1026–1049: one record per ISRC, highest popularity wins; records without ISRC are dropped when `is_filter_no_isrc=1`). Then each channel's rank position is linearly remapped into a score range — defaults from `.env.example` / `__init__` (lines 50–51):

| Channel | Weight range (rank 0 → last) |
|---|---|
| similar | 1.7 → 1.0 |
| popular | 1.0 → 0.8 |
| same_artist | 1.9 → 1.0 |
| related_artist | 1.8 → 1.0 |
| related_track | 2.0 → 1.5 |

A track appearing in multiple channels **sums** its per-channel scores and accumulates the channel names in `sources`. Final assembly enforces: same-artist count ≤ `max(1, int(top_k * U(0.15, 0.20)))` where the ratio is drawn fresh per query from `same_artist_ratio_range=[0.15, 0.20]`; any other single artist ≤ `max(1, int(top_k * 0.15))`; duplicate track names excluded; a `popular`-channel floor of `int(top_k * popular_ratio)` = `int(top_k * 0.1)` tracks is back-filled from popular candidates if the merge under-fills it; if the list is still short, remaining candidates are appended in order. Final scores are rescaled so the max is 2.0, then stored as `round(score / 100, 2)` after `int(min(score, 2.0) * (100 / max_score)) / 100`. The list is finally shuffled in blocks of 1 by `album_name` then `artist_name` (`shuffle_result_block`, 1051–1088) so no two adjacent items share an album or artist. `EmbeatDatabase` default `random_seed=0` means a fresh random draw per call; the eval harness pins `random_seed=616` (`eval/pipeline.py:28`).

### (a.7) Upstream evaluation (what its numbers do and do not prove)

`eval/pipeline.py`, `eval/config.py`, `README.md` §LLM Blind Evaluation.

- 157 seed tracks sampled from 6,291 EveryNoise micro-genres, filtered to tracks present on both Embeat and Netease Cloud Music; 5 recommendations per side, metadata standardized via MusicBrainz, order shuffled, judged blind by 4 LLMs. Reported wins (Embeat / Netease / tie / not rated): Claude Opus 5: 122/14/1/20; Kimi K3: 117/23/2/15; GPT 5.6 Sol: 109/20/0/28; Gemini 3.1 Pro: 90/5/0/62. The README's "84~95% win rate" is wins ÷ (wins + losses) over *rated* pairs only (Claude 122/136 = 89.7%, Gemini 90/95 = 94.7%); it excludes ties and the up-to-62 unrated items per judge. Diversity claim: 4.88 distinct artists per 5-list vs Netease 3.57.
- **Not portable to v3:** the harness needs the Netease API, OpenRouter keys, MusicBrainz normalization, and a 45M-track index. Its verdict is also an LLM preference judgment, not a listener keep-rate. v3 must use the harness in §(d) instead. The one portable lesson is methodological: fix the seed (`random_seed=616` in their eval), blind the comparison, and report unrated/missing items instead of dropping them silently.

## (b) Exact feature to port, mapped onto our stack

### (b.0) What "port" means here

Port the **retrieval architecture**: five-channel recall over a candidate pool, per-channel rank→score remap with summation across channels, ISRC/title dedup, same-artist and per-artist caps, popular-channel floor, and genre voting. Port **none** of the trained weights, datasets, or the Qdrant server.

### (b.1) Channel 1 — acoustic similarity (PORT, re-implemented encoder)

- **Upstream:** 64-d L2-normalized EmbeatMLP embedding, cosine, genre-filtered.
- **Our replacement — no torch, no training.** The v2 feature vector (`scoring.py` `feat_vector`) is 78-d: 10 ReccoBeats (`danceability, energy, valence, acousticness, instrumentalness, speechiness, liveness, loudness, tempo, mode`), key as (sin, cos) of `2π·key/12`, 14 librosa scalars, 51 vector components (`contrast_mean` 7, `mfcc_mean` 13, `mfcc_std` 13, `chroma_hist` 12, `tonnetz_mean` 6), 1 `is_minor` flag. Reuse the v2 fitted transforms already in `model_v2.npz` (`V2Model` in `scoring.py`): median-impute → standardize → **PCA(15)** (fitted on the 650 likes, `model_v2.py`) → L2-normalize each row. That 15-d unit vector is our acoustic embedding. Cosine similarity is then a dot product.
- **Why not train EmbeatMLP:** it needs 10M-track pair sampling (99% same-album positives — we have no album ids for liked tracks, and only 650 tracks total), batch 4096, CUDA, and Spotify audio features for millions of tracks. On 1.5 shared CPUs it is infeasible and the pair statistics would be meaningless at n=650. **CUT: EmbeatMLP training.** The pair rule survives as a *validation* signal instead: same-artist / same-bucket pairs must score closer than random pairs (test T-03 in §c).
- **Candidate limit:** `min(top_k*10, 512)` = 100 at top_k=10, over the pool (≤800), filtered to the seed's voted bucket (§b.5). If the bucket filter yields <10 candidates, fall back to unfiltered acoustic neighbours and log the fallback count (Embeat has no such fallback; at 45M it never needs one — at n≤800 it does).
- **Cost:** 800×15 float32 ≈ 48 KB. Brute-force cosine over the whole pool per seed is ~12k multiply-adds — sub-millisecond. No index needed.

### (b.2) Channel 2 — same-genre popular (PORT, degraded popularity)

- **Upstream:** popularity-DESC scroll within genre, re-ranked by query-vector cosine.
- **Our popularity proxy:** no Spotify popularity exists in our data (ReccoBeats audio-features has no popularity field; nothing in `features_v2.py` / `recommend_v2.py` reads one). Use Deezer `rank` from the Deezer search payload for each pool track, min-max normalized within its bucket to [0, 1]. **Build-time verification required:** log the raw Deezer payload keys for 5 pool tracks and confirm `rank` is present and varies; if it is absent, channel 2 degrades to "top-N by v2 score within the seed bucket" and the build log must say so in one line. Do not invent a popularity signal from audio features.
- **Candidate limit:** `min(top_k*2, 128)` = 20 at top_k=10. Filter: same bucket as seed (voted track bucket, §b.5); then order by the proxy, take 20, re-rank by acoustic cosine to the seed.
- **Minimum-popularity gate:** port as `popularity_proxy ≥ min(seed proxy, 0.1)` on the normalized scale, mirroring `min(query_popularity, min_popularity=0.1)`. If the seed has no proxy value, the gate is 0.0 (no filtering).

### (b.3) Channel 3 — same artist (PORT, exact)

- **Upstream:** `artist_idx` equality + popularity DESC.
- **Ours:** normalized artist-string equality (lowercase, strip `feat.`/`&` collaborators the way `pipeline.py` `clean()` does, first artist only). Pool tracks by the seed artist, ordered by popularity proxy DESC, limit 20, shuffled with the fixed seed before filtering (upstream shuffles; we pin `random_seed=616` so the shuffle is reproducible).
- **Cap:** the merge caps same-artist tracks in the final 10 at `max(1, int(10 * U(0.15, 0.20)))` = 1 or 2. This is consistent with the v6 keeps (6 kept tracks, 6 distinct artists).

### (b.4) Channel 4 — related artists (PORT, reusing pool provenance)

- **Upstream:** EveryNoise related-artist graph (top 5 of ≥10) + vector cosine, re-sorted by popularity.
- **Ours:** Deezer related artists. `recommend_v2.py` `build-pool` already builds the pool from Deezer related artists of the top 30 liked artists (`TOP_ARTISTS=30`, `RELATED_PER_ARTIST=8`, `TOP_PER_RELATED=4`, `POOL_CAP=800`). Record per pool track its provenance: `(seed_artist, related_artist)` pair. Channel 4 = pool tracks whose artist is a Deezer-related artist of the *seed track's* artist, ranked by acoustic cosine, limit 20, re-sorted by popularity proxy DESC (mirroring upstream's post-search popularity sort).
- **Circularity note (must be stated in the build log):** because the pool itself is built from related artists, channel 4 is a *filter over the pool*, not an independent fetch. Its marginal value is measured by the ablation test in §(d): if removing channel 4 changes P@10 by <0.01, the channel is redundant with pool construction and is dropped, not kept for symmetry.

### (b.5) Genre voting — track bucket estimation (PORT)

- **Upstream:** 6,291 micro-genres; track genre voted from related-track/similar candidates with `1/sqrt(rank+1)` weights, override only if top > 2.0× runner-up.
- **Ours:** 4 buckets only — E/R/H/O from Kevin's genre playlists (`/tmp/genre_playlists.json`: E 227, R 84, H 310, O 52 entries; first-label-wins dedup in `features_v2.py` `load_buckets`). Pool-track buckets: (1) inherit from the liked-artist bucket map when the pool artist is a liked artist; (2) else map Deezer artist genres to E/R/H/O by a fixed keyword table written in the build (e.g. hip-hop/trap/rap → H; rock/metal → R; electronic/pop → E; else O); (3) unmapped → bucket `O` and treated as unknown-genre: excluded from the genre-filtered channel 1, exactly as upstream drops `artist_genre_idx == 0` (`filter_similar_candidates`).
- **Vote:** for a seed, take channel-5 candidates if any (none in v3 — stub returns []), else channel-1 candidates; weight each candidate's bucket by `1/sqrt(rank+1)`; the top bucket becomes the seed's track bucket only if its score > 2.0× the runner-up (`track_genre_score_ratio=2.0`); otherwise fall back to the seed artist's bucket. Record `seed_bucket_source ∈ {vote, artist, fallback_O}` per query in the output JSON.
- **Cut:** the EveryNoise 2-D genre coordinates (`norm_x`, `norm_y`) and the 6,291-label taxonomy. Nothing at our scale can use them; the 4-bucket vote is the whole port.

### (b.6) Channel 5 — playlist collaborative filtering (CUT, stubbed)

- **Upstream:** Track2Vec Word2Vec skip-gram, 1.88M playlists, 1.09M-track vocab, min score 0.75.
- **Our data:** zero playlist co-occurrence. `~/workspace/spotify-liked-songs.txt` has no dates and no playlist membership; the 4 genre playlists are label sources, not co-occurrence (each track sits in ≤1 bucket, so within-bucket "co-occurrence" is a constant and carries no signal). Training Word2Vec on 4 playlists / 650 tracks would memorize bucket labels, not learn co-occurrence.
- **Port decision: STUB.** `search_related_track_record` returns `[]` and logs `channel5=stub:no-playlist-corpus` on every call. The merge runs over 4 channels; the related-track weight range [2.0, 1.5] is simply unused. The stub must be a named function with a test asserting it returns [] (§c, T-11), so a future corpus can replace it without touching the merge.
- **What unblocks it (§e):** a playlist corpus of ≥10,000 playlists containing ≥1 liked track each (Spotify public playlists, or the friend's API if it exposes playlist membership), with `min_count=10` and the upstream hyperparameters otherwise unchanged. Until then, any claim of "collaborative filtering" in v3 output is false and must not appear.

### (b.7) Qdrant replacement (CUT the server; keep the semantics)

- **No Qdrant.** Upstream needs it for 45M vectors; our universe is ≤800 pool tracks + 650 likes. The replacement is `sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='cosine')` over the 15-d L2-normalized embeddings (exact search, no approximation error, no server, no 6333 port, no `qdrant-client` dependency — test T-01 asserts the import is absent).
- **Payload indexes become dicts:** `artist → [track ids]`, `bucket → [track ids]`, plus a NumPy popularity-proxy array. Channel 2/3 "scrolls" become `argsort` on the proxy within a bucket/artist slice. This preserves upstream query semantics (filtered vector search; popularity-ordered scan) at our scale.
- **Memory budget:** embeddings 800×15 float32 ≈ 48 KB; full 78-d feature matrix 800×78 float32 ≈ 250 KB; pool JSON on disk < 5 MB. Peak RSS for the whole recommend run must stay ≤ 2 GB (test T-15); measured, not assumed.
- **What is lost:** HNSW approximate search (irrelevant below ~100k vectors — and upstream itself defaults `hnsw_indexing_threshold=0`), persistence across runs (rebuild the index from `pool_features.json` each run; rebuild time is <1 s at this scale), and the `isrc` keyword index (replaced by an in-memory ISRC dict, §b.8).

### (b.8) Merge, dedup, and caps (PORT VERBATIM, with two corrections)

Port `merge_result_by_score` (lines 1090–1221) exactly, including:

- Per-channel rank→score linear remap with the upstream default ranges: similar [1.7, 1.0], popular [1.0, 0.8], same_artist [1.9, 1.0], related_artist [1.8, 1.0]; related_track [2.0, 1.5] unused while stubbed. Multi-channel appearance sums scores; `sources` lists every contributing channel.
- Final assembly caps: same-artist ≤ `max(1, int(10 * U(0.15, 0.20)))`; any other single artist ≤ `max(1, int(10 * 0.15))` = 1; popular-channel floor `int(10 * 0.1)` = 1 track back-filled if under-filled; shortfall back-fill from remaining candidates in merged-score order.
- Final score rescale: max merged score → 2.0, stored as `round(score/100, 2)` after the upstream integer rescale. Keep the upstream rescale arithmetic, not a fresh normalization, so scores stay comparable to the spec.
- **Determinism correction:** upstream default `random_seed=0` re-rolls every call. v3 pins `random_seed=616` (the value upstream's own eval uses) for the same-artist-ratio draw, the popular shuffle, and all tie-breaking. Two runs must produce byte-identical top-10 order (test T-13).
- **ISRC dedup (PORT, with fallback):** upstream `filter_by_isrc` keeps the highest-popularity record per ISRC and drops no-ISRC records. Our Deezer matches *do* carry ISRC (`features_v2.py` `match_deezer` stores `m.get('isrc')` in `data/deezer_matches.json`), so ISRC dedup is portable for Deezer-matched tracks. **Correction:** set the v3 equivalent of `is_filter_no_isrc` to **0** — never drop a track for missing ISRC. No-ISRC tracks dedup by normalized `(title, artist)` instead. Rationale: upstream can afford to drop no-ISRC records at 45M; at n≤800 each drop is 0.125% of the universe.
- **Remix filter (DO NOT PORT VERBATIM):** upstream drops any candidate whose title contains `remix`. Kevin's liked list contains **13 tracks with "remix" in the title (13/650 = 2.0%)**, e.g. `Click (Knock2 Remix)` — ISOxo, Knock2 and `Zombie (Zombic & Felix Schorn Remix)`. A verbatim port would exclude a class he demonstrably likes. v3 drops only a remix *of the seed track itself* (same normalized title stem + same artist); other remixes stay eligible. Test T-10 asserts a non-seed remix can appear in the top 10.
- **Other filters ported as-is:** drop seed; drop same (title, artist); drop duplicate titles; drop titles starting with seed title + space; drop `popularity_proxy < min(seed proxy, 0.1)`; channel-1 unknown-bucket exclusion (§b.5).
- **Final block shuffle:** port `shuffle_result_block` (album, then artist, `max_block_len=1`). Album is usually absent for pool tracks; when absent, the album pass is a no-op and the artist pass still applies. Log how many adjacent artist repeats remain after the shuffle (must be 0).

### (b.9) What gets cut (summary)

| Upstream piece | Verdict | Reason |
|---|---|---|
| EmbeatMLP training (InfoNCE, batch 4096, CUDA) | CUT | Needs 10M-track pairs, album ids, CUDA; n=650 |
| EmbeatMLP weights (`model.pt`) | NOT VENDORED | License contradiction (§0.1); not needed |
| Track2Vec training + 1.88M playlists | CUT → stub | No playlist corpus |
| Qdrant server + HNSW | CUT → sklearn brute force | n≤800; exact search is cheaper |
| EveryNoise 6,291 genres + related-artist graph | CUT → 4 buckets + Deezer related | No EveryNoise data; Deezer graph already in pool builder |
| Name search (`-s`/`-a` modes, full-text indexes) | CUT | v3 queries are always by seed Spotify ID from liked songs |
| LLM-judge eval vs Netease | CUT → §(d) harness | Needs Netease + OpenRouter + 45M index |
| `remix` title filter (verbatim) | MODIFIED | 13/650 liked tracks are remixes |
| ISRC-missing drop | MODIFIED | Never drop for missing ISRC at n≤800 |

### (b.10) v3 module shape (for the build agent)

New module `embeat_port.py` in `~/workspace/audio-pipeline/` (build agent's file, not this spec's): `build_index(pool_features, model_npz)` → embeddings + dicts; `recall_similar / recall_popular / recall_same_artist / recall_related_artist / recall_related_track_stub` → per-channel candidate lists with `sources` tags; `vote_track_bucket` ; `merge_by_score` (upstream arithmetic); `recommend(seed_spotify_id, top_k=10, random_seed=616)` → final 10 with per-track `sources`, `score`, `seed_bucket_source`. It consumes `scoring.feat_vector` / `V2Model` and `recommend_v2.py`'s pool — it must not reimplement feature extraction or scoring. Candidate generation stays in `recommend_v2.py build-pool`; this module only retrieves and merges.

## (c) TDD acceptance tests — written FIRST

These tests are **specified here, not created now**: this spec agent is read-only outside its own spec file, so `tests/test_embeat_port.py` is to be written by the v3 build agent **before** any implementation, run against the v2-only code (expected: the port tests fail), then run again after the port (expected: all pass). The "watch it fail" run must be logged with the failing test ids before implementation starts; a test that passes before implementation is void and must be rewritten to actually discriminate.

Conventions: `top_k=10`, `random_seed=616` everywhere; pool fixture = the real `data/pool_features.json` once it lands, else a synthetic 800-track fixture with the same schema (78-d features, bucket, artist, Deezer rank, ISRC-or-None). Seeds for tests: 5 fixed liked-track IDs spanning buckets E/R/H (IDs recorded in the test file, not invented here).

- **T-01 Qdrant is absent.** `grep -r qdrant_client ~/workspace/audio-pipeline/*.py` returns 0 matches; `import embeat_port` succeeds with no `qdrant_client` import in `sys.modules`.
- **T-02 Embedding contract.** For all 650 liked tracks, the acoustic embedding has dim 15 and L2 norm 1.0 ± 1e-6. Cosine(seed, seed) = 1.0 ± 1e-9.
- **T-03 Pair-rule sanity (EmbeatMLP's rule, not its weights).** Mean cosine(same-artist pairs) > mean cosine(random pairs) by ≥ 0.10, and mean cosine(same-bucket pairs) > mean cosine(cross-bucket pairs) by ≥ 0.05, computed over 200 sampled pairs each with seed 616. Fails if the PCA embedding carries no artist/genre structure.
- **T-04 Channel-1 limits and filter.** `recall_similar(seed, top_k=10)` returns ≤100 candidates; 0 candidates have unknown bucket `O` when the seed bucket is voted (not fallback); the seed itself is absent.
- **T-05 Channel-2 mechanism.** `recall_popular` returns ≤20 candidates, all in the seed bucket, ordered by popularity proxy before the cosine re-rank (assert on the pre-rerank list), and the re-ranked list's top item has the highest acoustic cosine among the 20.
- **T-06 Channel-3 mechanism.** `recall_same_artist` returns only tracks whose normalized artist equals the seed artist, ≤20, ordered by popularity proxy DESC.
- **T-07 Channel-4 mechanism.** `recall_related_artist` returns only tracks whose artist is in the seed artist's Deezer-related set, ≤20; returns [] when the seed artist has no related artists in the pool.
- **T-08 Merge arithmetic.** Synthetic input: track X ranked #0 in `similar` and #0 in `same_artist`; track Y ranked #0 in `similar` only. After `merge_by_score`, score(X) = 1.7 + 1.9 = 3.6 before rescale, score(Y) = 1.7, and X outranks Y. Also assert a rank-last `similar` item scores exactly 1.0 pre-rescale (remap endpoint).
- **T-09 Caps.** Over 20 fixed seeds, no final top-10 contains >2 tracks by the seed artist, and no non-seed artist appears >1 time. Final top-10 contains ≥1 `popular`-sourced track on every seed where channel 2 is non-empty.
- **T-10 Remix correction.** A pool track whose title contains "remix" and whose normalized title stem differs from the seed's is eligible and can appear in a top-10; a remix of the seed itself (same stem + same artist) never appears.
- **T-11 Channel-5 stub.** `recall_related_track_stub(seed)` returns `[]` and the call logs `channel5=stub:no-playlist-corpus`. The merge still returns 10 tracks.
- **T-12 Negative control (unit level).** In the eval pool (§d), all 13 negative Spotify IDs rank below position 10 under the ported ranker, and 0 of the 13 appear in any final top-10 across the 5 fixed seeds.
- **T-13 Determinism.** Two `recommend()` calls with `random_seed=616` on the same seed return byte-identical ordered top-10 (track IDs and scores).
- **T-14 Exclusion.** Final top-10 ∩ liked-song IDs = ∅ and ∩ the 60 history IDs in `spotify-recommended-tracks.txt` = ∅, for every test seed.
- **T-15 Budget.** `recommend()` over the full ≤800 pool completes in ≤500 ms per seed (median of 20 seeds) and a full 10-seed run peaks at ≤2 GB RSS (measured with `/usr/bin/time -v`).
- **T-16 Eval beats baseline (integration, runs last).** §(d) thresholds; fails on v2-only ranking, passes after the port. This is the test that proves the feature, not the plumbing.

## (d) Eval-harness metric — proof it beats v6 6/10

### (d.1) Harness (extend, do not replace)

Extend the existing `evaluate.py` (v2 build agent's file): it already defines the ranking task — pool = 130 held-out liked tracks (most recent 20% by list position, `TEST_N=130`) + negatives + 200 distractor tracks sampled from the candidate pool with seed 42; relevance = 1 iff held-out like; metrics `P@10`, `NDCG@10` (its `ndcg_at_k` / `precision_at_k`), mean rank of held-out likes, mean rank/score of negatives. Two changes are mandatory:

1. **Negatives = all 13**, not the 9 v5-only set currently in `evaluate.py`. The 4 v6 thumbs-down are the discriminating cases (all H bucket — the bucket where v6 failed 4/4 on second listen).
2. **Add a per-channel ablation**: P@10/NDCG@10 with each of channels 1–4 removed in turn, and with the merge replaced by channel-1-only ranking. Report all six numbers (full port, minus-1, minus-2, minus-3, minus-4, channel-1-only) on the identical pool.

The harness must rank the **same pool** for v2, v6b (per-bucket v6 scorer), v6g (global v6 scorer), and the ported ranker, using identical `feat_vector` inputs — the existing `evaluate.py` already does this for v2/v6b/v6g; the port is the fourth method.

### (d.2) Pass thresholds (all must hold; numbers, no adjectives)

Let `P_port`, `N_port` be the ported ranker's P@10 and NDCG@10; `P_best`, `N_best` the best of {v2, v6b, v6g} on the same pool; `neg_top10` the count of the 13 negatives inside the ported top-10 over the whole eval pool; `neg_top10_best` the same count for the best baseline.

1. **Ranking win:** `P_port ≥ P_best + 0.05` **and** `N_port ≥ N_best + 0.05` (absolute points, same pool, same seed 42). A 0.05 absolute gain on a 343-item pool (130 + 13 + 200) is ≥ 17 additional held-out likes displaced into the top 10 across the ranked list — it cannot be produced by reordering ties.
2. **Negative control:** `neg_top10 = 0` **and** `neg_top10 ≤ neg_top10_best - 1`. Mean rank of the 13 negatives under the ported ranker is ≥ 20 positions worse (larger) than under the best baseline. If any of the 4 v6 H-bucket thumbs-down enters the top 10, the port FAILS regardless of P@10.
3. **Ablation (no dead channels):** removing any single channel 1–4 must not *improve* P@10 by >0.02; if it does, that channel is noise and must be removed from the shipped merge, with the removal recorded. Each of channels 1–4 must contribute ≥1 track to the final top-10 on ≥50% of the 20 eval seeds, else it is dead weight and is cut. Channel 5 must contribute 0 tracks on 100% of seeds (stub assertion).
4. **Diversity:** distinct artists in the final top-10 ≥ 8 on the median seed; same-artist count ≤2 on every seed (already T-09).
5. **Human gate (the number that actually beats 6/10):** a fresh 10-track playlist built by the ported ranker, listened to by Kevin on second listen exactly as v6 was judged, keeps **≥7 of 10** tracks. 6/10 is the baseline; 7/10 is the pass line (+1 track). The offline metrics (1)–(2) gate whether the playlist is built at all; the human keep-rate is the ship decision. Report both numbers side by side; do not claim the offline win proves the human win.

### (d.3) Reporting format

One JSON file, `evaluation_v3_embeat.json`, next to `evaluation_v2.json`, containing: pool composition counts (130/13/200), per-method P@10 / NDCG@10 / mean-rank-heldout / mean-rank-negatives / `neg_top10`, the six ablation numbers, per-channel contribution rates, runtime/RSS, `random_seed=616`, and the git hash of the v3 code. No metric is reported without its denominator and pool size.

## (e) What it cannot do on our data, and what would unblock it

| Limitation | Consequence in v3 | Unblocker |
|---|---|---|
| No playlist co-occurrence data (no playlist corpus; liked file has no playlist membership or dates) | Channel 5 is a stub; no collaborative signal at all. The port is purely content-based + artist-graph. | ≥10,000 playlists each containing ≥1 liked track, then train Track2Vec with upstream hyperparameters (`vector_size=64, min_count=10, window=100, negative=20, epochs=60, seed=616`) and enforce `min_related_track_score=0.75`. Sources: Spotify public playlists or the friend's API if it exposes playlist membership (sign-in pending; not available at spec time). |
| Candidate universe ≤800 pool tracks vs upstream 45M | No long-tail discovery; every recommendation is within 1 Deezer-related hop of a top-30 liked artist. Novel artists outside that hop cannot appear. | Expand pool: more seed artists (top 100, not 30), 2-hop related artists, and/or Deezer chart/genre browsing. Each expansion must re-run §(d) — a bigger pool can lower P@10. |
| No Spotify popularity | Channel 2 runs on Deezer `rank` proxy (unverified at spec time, §b.2) or degrades to v2-score order. Popularity *gate* still works; popularity *ranking* is approximate. | Spotify popularity via the friend's API, or ReccoBeats if it ever exposes it. Verify before trusting. |
| Genre = 4 buckets (E/R/H/O) vs 6,291 micro-genres | Genre filter is coarse: the H bucket alone holds 310/650 liked tracks (47.7%). Channel-1 genre filtering barely constrains H seeds, which is exactly where v6 failed. Mitigation in v3: the track-bucket vote (§b.5) plus the LR negative signal from `model_v2` still apply inside H. | Finer labels: Spotify artist genres via the friend's API, or Deezer genre ids per artist mapped to a 20–50 label taxonomy. EveryNoise scraping is not recommended (scraping ToS risk + 2M-artist scale). |
| No full-length audio (30 s Deezer previews only) | librosa features are preview-only; structure features (3×10 s deltas) cover 30 s, not full songs. EmbeatMLP inputs are unaffected (it uses Spotify features, not audio), but any future waveform encoder is capped at previews. | Friend's full-song API (sign-in pending). If it lands, re-extract librosa on full audio and re-run §(d) before claiming any gain. |
| No album IDs for liked tracks | The 99%-same-album positive rule (§a.2) cannot be used for training or for pair validation; only same-artist / same-bucket pair checks (T-03) apply. | Album metadata per liked track (Spotify API or friend's API). |
| No ISRC for some tracks; no track durations for pool tracks in some paths | ISRC dedup falls back to (title, artist); duration-based Deezer matching already handles liked tracks. | ISRC from Deezer match payload (already stored for liked tracks); record ISRC at pool-build time for pool tracks. |
| torch / CUDA training infeasible on this VM (1.5 CPU, ~6 GB) | No neural encoder training in v3. The PCA-15 embedding is linear; it cannot learn the key/mode/tempo interactions EmbeatMLP's towers learn. | A GPU machine or a hosted training run; even then, training needs ≥100k tracks with Spotify features, not 650. |
| Upstream eval (LLM judge vs Netease) not portable | No external preference baseline; §(d) offline metrics + Kevin's keep-rate are the only verdicts. | None sought — the §(d) harness is the correct substitute at this scale. |
| License contradiction on weights (§0.1) | `model.pt` not vendored; architecture-only port. | Written clarification from upstream (gdstudio-org) on whether weights are MIT. |

### (e.1) Known verification gaps in this spec (not hidden)

1. The `Embeat-dev/Embeat` path was asserted in the brief; it does not resolve. The correct repo is `gdstudio-org/Embeat`. If the brief's path was a different fork, it was not found via clone, ls-remote, or web search.
2. README's MIT claim for code/weights vs the CC BY-NC 4.0 `LICENSE` file on disk — unresolved; see §0.1.
3. InfoNCE τ: 0.07 (`train/loss.py`) vs 0.05 CLI default (`train/train.py`). The shipped checkpoint's τ is unrecorded. Immaterial to the port (no training), recorded for accuracy.
4. v2 outputs did not exist at spec time; v2 interface claims rest on v2 source files. The build must re-ground them.
5. Deezer `rank` as popularity proxy is unverified against a live payload (§b.2).
6. Upstream latency (30–200 ms) and scale (45M) figures are README claims, not reproduced here.

## Sources

- Repo (correct path): https://github.com/gdstudio-org/Embeat — commit `e3a32a64003f3ae1c7118ac3bbc576e56bcec9c7`, cloned to `/tmp/Embeat` 2026-11-22.
- `README.md` (Introduction, System Architecture, Model Details, Multi-Channel Recall, License table lines 408–413, LLM Blind Evaluation table).
- `infer/Embeat.py` — `EmbeatDatabase.__init__` (44–84), `search_vector_similar_record` (686–720), `search_genre_popular_record` (722–779), `search_artist_popular_record` (781–807), `search_related_artist_record` (809–837), `batch_most_similar` (839–866), `search_related_track_record` (867–902), `filter_similar_candidates` (904–975), `filter_others_candidates` (977–1024), `filter_by_isrc` (1026–1049), `merge_result_by_score` (1090–1221), `search_entry` (1223–1361).
- `infer/hf_to_qdrant.py` — `PAYLOAD_FIELD_WHITELIST` (24–35), `quantize_to_uint8` (276–285), `ensure_collection` HNSW m=8/ef_construct=200, `hnsw_indexing_threshold=0`, distance cosine, datatype uint8 default.
- `infer/infer.py` — `infer()`, cosine rescaled to [0, 1].
- `train/model.py` — `EmbeatMLP`, `EmbeatMLPConfig` (dual tower, 64-d L2-normalized output).
- `train/dataset.py` — `build_dense_vector` (7 features centered at 0.5), `key_to_idx`, `mode_to_idx`, `tempo_to_idx`, `time_signature_to_idx`.
- `train/sampler.py` — `PairSamplerConfig` (99% same-album positives, dense caps, popularity bias).
- `train/loss.py` — `MaskedInfoNCEConfig` (τ=0.07).
- `train/train.py` — CLI defaults (batch 4096, τ=0.05, max_steps 200).
- `train/train_track2vec.py` — Word2Vec skip-gram hyperparameters.
- `data/everynoise_genre_map.py`, `data/everynoise_related_artist.py`, `data/everynoise_unk_artist_genre.py` — EveryNoise scraping.
- `eval/pipeline.py` — LLM blind A/B harness (`random_seed=616`, `same_artist_ratio_range=[0.0, 0.0]` in eval).
- `.env.example`, `requirements.txt` (`qdrant-client>=1.18.0,<1.19.0`), `LICENSE` (CC BY-NC 4.0 text).
- Local: `~/workspace/audio-pipeline/{pipeline.py, recommend.py, recommend_v2.py, model_v2.py, scoring.py, evaluate.py, features_v2.py, negatives_features.py}`, `~/workspace/spotify-liked-songs.txt` (650 tracks), `~/workspace/spotify-recommended-tracks.txt` (60 entries), `/tmp/genre_playlists.json` (E 227 / R 84 / H 310 / O 52), `~/workspace/audio-pipeline/analysis.json` (93 analyzed, BPM median 123.0, brightness mean 2765.6 Hz, energy RMS mean 0.31854).
