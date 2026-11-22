# implicit (benfred/implicit) — Porting Spec for v3

**Status: DEFERRED — DO NOT IMPLEMENT until the gating conditions in §(b.6) pass.
There is no implementation work to do from this spec today.**

- Spec date: 2026-11-21 (America/Los_Angeles)
- Upstream repo: https://github.com/benfred/implicit
- Upstream version pinned: **0.7.3**, commit `8a95dbe` ("Bump version: 0.7.2 → 0.7.3"),
  cloned to `/tmp/implicit` and read in full on 2026-11-21. Docs site serves 0.7.2
  (https://benfred.github.io/implicit/); the 0.7.3 code read here is authoritative
  where the two differ.
- Target repo: `github.com/kevinkiluu/ai-song-recs`, v3 build
  (v2 pipeline + ported features from these research specs).
- Local data verified 2026-11-21: `~/workspace/spotify-liked-songs.txt` = 650 lines
  (title, artist, `spotify:track:ID`; no added dates);
  `~/workspace/spotify-recommended-tracks.txt` = 61 lines = 60 entries + header.

## 0. Verdict up front (read this before §(b))

1. **Today, with only 650 liked-track IDs, implicit can do nothing useful.**
   A single-user interaction matrix is degenerate for both ALS and BPR (§(a.5)).
   The honest answer is not "nothing until history arrives" — it is stronger:
   **nothing until history arrives AND a multi-user matrix over the same Spotify
   track IDs exists.** Streaming history alone still leaves exactly one user row.
2. **Streaming history is necessary but not sufficient.** It converts Kevin's
   binary likes into weighted per-track play counts for his one row. It does not
   create the cross-user co-occurrence that collaborative filtering learns from.
3. **The only legitimate near-term use of implicit is as a re-ranker / similar-items
   source over in-matrix tracks, blended with the v2 content (audio-profile) scores.**
   It can never score a track that appears in no user's history. Candidate
   generation for fresh tracks stays content-based (ReccoBeats + librosa on Deezer
   30 s previews).
4. **Do not fabricate pseudo-users.** Treating genre buckets, time windows, playlists,
   or individual tracks as "users" manufactures co-occurrence that never happened and
   would present spurious rankings as collaborative filtering. Forbidden in §(e.3).

---

## (a) How the upstream mechanism works

### (a.1) What the library is

`implicit` is Ben Frederickson's "Fast Python Collaborative Filtering for Implicit
Datasets" (MIT licence). It provides CPU (Cython/OpenMP) and GPU (CUDA) implementations
of ALS, BPR, Logistic Matrix Factorization (LMF), and item-item nearest-neighbour
models (Cosine / TF-IDF / BM25). No GPU is available on our VM, so only the CPU
classes are in scope: `implicit.cpu.als.AlternatingLeastSquares`,
`implicit.cpu.bpr.BayesianPersonalizedRanking`.

Sources:

- Repo root README: https://github.com/benfred/implicit/blob/master/README.md
  (cloned copy `/tmp/implicit/README.md`, lines 1–120)
- Docs index: https://benfred.github.io/implicit/
- Version: `/tmp/implicit/pyproject.toml:13` (`version = "0.7.3"`) and
  `/tmp/implicit/setup.cfg:2` (`current_version = 0.7.3`)
- Python requirement: `requires-python = ">=3.9"` (`/tmp/implicit/pyproject.toml`);
  runtime deps `numpy>=1.17.0, scipy>=0.16, tqdm>=4.27, threadpoolctl`
  (`/tmp/implicit/pyproject.toml`, dependencies section)
- Install: `pip install implicit` (prebuilt wheels for x86_64 Linux) or
  `conda install -c conda-forge implicit` (README, Installation section)

### (a.2) ALS — Alternating Least Squares

**Papers.** Hu, Koren & Volinsky, "Collaborative Filtering for Implicit Feedback
Datasets" (http://yifanhu.net/PUB/cf.pdf), plus the conjugate-gradient optimisation
paper cited in the README. The model learns latent factors minimising a weighted
squared error against a binary preference target.

**Hu et al. formulation (paper, not code).** For user `u`, item `i`, with raw
interaction count `r_ui` (e.g. play count):

- Preference: `p_ui = 1` if `r_ui > 0`, else `0`.
- Confidence: `c_ui = 1 + alpha * r_ui` — repeated plays raise confidence, with
  diminishing effect as `alpha` scales the count.

**What the code actually does (differs in one detail that matters).**
`implicit/cpu/als.py:98-200`, `fit(user_items)` docstring:

> "The user_items matrix does double duty here. It defines which items are liked
> by which users (P_ui in the original paper), as well as how much confidence we
> have that the user liked the item (C_ui). The negative items are implicitly
> defined: ... P_ui = 0 and C_ui = 1 for all these items. Negative items can also
> be passed with a higher confidence value by passing a negative value, indicating
> that the user disliked the item."

And `/tmp/implicit/implicit/cpu/als.py:133-134`:

```python
if self.alpha != 1.0:
    Cui = self.alpha * Cui
```

i.e. `implicit` multiplies the **whole input matrix** by `alpha`; it does **not**
compute `1 + alpha * r` internally. To reproduce Hu's formula the v3 loader must
pass precomputed confidences (`1 + alpha * r`) with `alpha=1.0`, or pass BM25
weights (see §(a.4)). Passing raw play counts with `alpha=2.0` yields confidence
`2*r`, not `1 + 2*r`. This is a verified code-level fact, not a doc paraphrase.

**Constructor signature** (`/tmp/implicit/implicit/cpu/als.py:20-97`, docs
https://benfred.github.io/implicit/api/models/cpu/als.html):

| Parameter | Default | Meaning |
|---|---|---|
| `factors` | 100 | Latent-factor dimensionality |
| `regularization` | 0.01 | L2 penalty on factors |
| `alpha` | 1.0 | Scalar multiplier on the input (confidence) matrix |
| `dtype` | `np.float32` | Factor dtype |
| `use_native` | True | Use Cython/OpenMP solver |
| `use_cg` | True | Conjugate-gradient solver (faster); `cg_steps = 3` set in `__init__` (`als.py:88`) |
| `iterations` | 15 | ALS sweeps (user step + item step per sweep) |
| `calculate_training_loss` | False | Log loss per iteration |
| `num_threads` | 0 | 0 = all cores; set explicitly on our VM (§(b.5)) |
| `random_state` | None | Seed for initial factors |

After `fit`, attributes are `user_factors` (`n_users × factors`) and `item_factors`
(`n_items × factors`), both `float32` by default.

**API surface used by the port** (all verified in the cloned source):

- `model.fit(user_items)` — `user_items` is a `scipy.sparse.csr_matrix`, shape
  `(n_users, n_items)`, values = confidence. `implicit/cpu/als.py:98`.
- `model.recommend(userid, user_items, N=10, filter_already_liked_items=True,
  filter_items=None, recalculate_user=False, items=None)` — returns `(itemids, scores)`.
  `items=` restricts scoring to a candidate subset; `filter_items=` excludes ids
  (the two are mutually exclusive). Defined in
  `implicit/cpu/matrix_factorization_base.py:35-95` (shared base) and
  `implicit/recommender_base.py:34-125` (interface docstring).
- `model.similar_items(itemid, N=10, ...)` — items most similar to a seed item, by
  item-factor dot product. `implicit/recommender_base.py:125-170`.
- `model.similar_users(userid, N=10, ...)` — nearest users in factor space.
- `model.recalculate_user(userid, user_items)` — recompute one user's factors from
  their row without refitting (`implicit/cpu/als.py:204`); `partial_fit_users` /
  `partial_fit_items` (`als.py:276`, `als.py:314`) for incremental updates.
- `model.save(path)` / `model.load(path)` — NumPy `.npz` format (`als.py`, `save`).
- **Import path:** use `implicit.cpu.als` / `implicit.cpu.bpr` directly. The
  top-level factory `implicit.als.AlternatingLeastSquares`
  (`/tmp/implicit/implicit/als.py`) switches CPU/GPU on `use_gpu=implicit.gpu.HAS_CUDA`;
  on a GPU-less VM that resolves to CPU, but the explicit CPU import removes the
  branch entirely.

### (a.3) BPR — Bayesian Personalized Ranking

**Paper.** Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit
Feedback" (https://arxiv.org/pdf/1205.2618.pdf). BPR learns factor embeddings by
SGD on a pairwise ranking loss: for each user, an observed item `i` should score
above an unobserved item `j`, maximising `sigmoid(x_u·y_i − x_u·y_j)`.

**Constructor signature** (`/tmp/implicit/implicit/cpu/bpr.pyx:70-125`, docs
https://benfred.github.io/implicit/api/models/cpu/bpr.html):

| Parameter | Default | Meaning |
|---|---|---|
| `factors` | 100 | Latent-factor dimensionality |
| `learning_rate` | 0.01 | SGD step size |
| `regularization` | 0.01 | L2 penalty |
| `iterations` | 100 | Training epochs (SGD passes) |
| `verify_negative_samples` | True | When sampling a negative item `j`, check it was not actually liked; slower but usually better predictions (docstring, `bpr.pyx`) |
| `num_threads` | 0 | 0 = all cores |
| `random_state` | None | Seed |

**Critical limitation, verified in source** (`implicit/cpu/bpr.pyx`, `fit` docstring):

> "BPR ignores the weight value of the matrix right now — it treats non zero
> entries as a binary signal that the user liked the item."

So BPR **cannot use play counts**. If the port uses BPR, the count information
from streaming history is discarded by design; BPR is only a binary-ranking
ablation against ALS. Do not present a BPR run as using play-count confidence.

**Explicit zeroing, verified in source** (`implicit/cpu/bpr.pyx:163-171`):

```python
item_counts = np.bincount(user_items.indices, minlength=items)
self.item_factors[item_counts == 0] = np.zeros(self.factors + 1)
...
self.user_factors[user_counts == 0] = np.zeros(self.factors + 1)
```

Items (and users) with zero interactions get **exactly zero** factor vectors
(the `+1` dimension is a bias term). Their recommendation scores are exactly
`0.0` for every user. This is the code-level anchor for the degeneracy proof
in §(a.5). (LMF does the same: `implicit/cpu/lmf.pyx:147-165`.)

### (a.4) Confidence weighting: BM25 (the library's own recipe)

The library's documented music recipe is the Last.fm tutorial and example:

- Tutorial: https://benfred.github.io/implicit/tutorial_lastfm.html — "we want to
  give repeated plays more confidence ... but have this effect taper off ...
  Likewise we want to direct some of the confidence weight away from popular
  items. To do this we'll use a bm25 weighting scheme."
- Example: `/tmp/implicit/examples/lastfm.py` — `plays = bm25_weight(plays, K1=100,
  B=0.8)` then `user_plays = plays.T.tocsr()` then
  `AlternatingLeastSquares(factors=128, regularization=0.05, alpha=2.0)` for ALS,
  `BayesianPersonalizedRanking(factors=63)` for BPR.
- Implementation: `implicit/nearest_neighbours.py:243-258`, `bm25_weight(X, K1=100,
  B=0.8)`. Formula (from source): per-row length normalisation
  `(1−B) + B·row_sum/mean_row_sum`, IDF `log(N) − log1p(count per column)`,
  weight `x·(K1+1)/(K1·length_norm + x)·idf`.
- **Orientation matters and is not symmetric.** In `lastfm.py`, BM25 is applied to
  the **item × user** matrix (`artist_user_plays`) and then transposed to
  user × item for `fit`. The v3 loader must follow that exact order; applying BM25
  to the user × item matrix is a different weighting, not the documented recipe.
- Sibling weightings in the same file: `tfidf_weight` (`nearest_neighbours.py:222`)
  and `normalize` (`:235`). BM25 is the primary; TF-IDF is an ablation only.

### (a.5) The single-user degeneracy — proof from the source

Claim: **with one user, ALS and BPR produce no ranking signal over unseen items.**
Three independent code facts, each sufficient on its own:

1. **Unseen items score exactly 0.0 (ALS).** The ALS solvers iterate over rows of
   the (transposed) confidence matrix and zero out empty rows:
   `/tmp/implicit/implicit/cpu/_als.pyx:98` (least-squares solver) and `:182`
   (conjugate-gradient solver):
   ```python
   if indptr[u] == indptr[u+1]:
       memset(&X[u, 0], 0, sizeof(floating) * factors)
       continue
   ```
   When solving for item factors, an item no user played is an empty row, so its
   factor vector is set to zero on the first sweep — overwriting the random
   initialisation (`als.py`: factors initialised as `random(...) * 0.01`).
   `recommend()` scores are `user_factor · item_factor` (via `topk` in
   `matrix_factorization_base.py:80-88`), so every never-played item scores
   exactly `0.0`. The top-10 among them is a tie-break artefact, not a ranking.
2. **Unseen items score exactly 0.0 (BPR).** Explicit zeroing, §(a.3):
   `bpr.pyx:163-164`. Same conclusion, by construction rather than by solver.
3. **Liked items are mutually indistinguishable.** With one user, each liked item's
   factor is solved from that single user's vector alone (the item update sees
   exactly one user). All liked-item factors are therefore (up to a per-item scalar)
   collinear with the user vector, so `similar_items` among liked tracks returns
   near-identical scores for every pair, and no cross-item structure exists to
   generalise to unseen items. Collaborative filtering learns from co-occurrence
   **across users**; one user provides zero co-occurrence pairs beyond "liked vs
   not liked".

Corollaries, each falsifiable by the TDD tests in §(c):

- A 1 × 650 binary matrix (the 650 liked IDs, all values 1.0) carries zero
  confidence information: uniform values make `alpha` scaling a no-op on ranking,
  and BPR ignores the values anyway.
- `similar_users` with one user returns nothing meaningful (no other users).
- `recalculate_user` / `partial_fit_users` cannot add signal; they re-derive the
  same single vector.
- The item-item nearest-neighbour models (`CosineRecommender`, `TFIDFRecommender`,
  `BM25Recommender`, `implicit/nearest_neighbours.py`) are degenerate for the same
  reason: with one user, every pair of liked items co-occurs exactly once and every
  unseen item co-occurs zero times, so similarities are 1.0 (liked/liked) or 0.0
  (anything/unseen). They are not a workaround.

This analysis is derived from reading the solver source. It has **not** been
confirmed by executing a fit, because `implicit` is not installed in this VM
(verified: `python3 -c "import implicit"` from `~` → `ModuleNotFoundError`, 2026-11-21).
Test T2 in §(c) exists precisely to confirm it empirically before any v3 code
depends on it; if T2 fails (non-zero score variance over unseen items), this
section must be revised, not the test.

### (a.6) Evaluation helpers shipped by the library

`implicit/evaluation.pyx` (all verified in the cloned source):

- `train_test_split(ratings, train_percentage=0.8, random_state=None)` — random
  holdout (`evaluation.pyx:14`).
- `leave_k_out_split(ratings, K=1, train_only_size=0.0, random_state=None)` —
  holds K interactions per eligible user (users with `> K+1` ratings) in test
  (`evaluation.pyx:141`). Note: it holds out **random** interactions, not the
  most recent ones; the v3 eval uses a **time-based** holdout instead (§(d.2))
  and does not rely on this function for the primary metric.
- `precision_at_k`, `mean_average_precision_at_k`, `ndcg_at_k`, `AUC_at_k`,
  `ranking_metrics_at_k` (all with `K=10` default; `ranking_metrics_at_k` returns
  a dict with `precision`, `map`, `ndcg`, `auc`) — `evaluation.pyx:236-366+`.
  These compute metrics **averaged over all users in the test matrix**. With Kevin
  as 1 of N users, his metrics are diluted by the other N−1 users, so the v3 eval
  must additionally compute **Kevin-only** P@10 / NDCG@10 by calling
  `model.recommend(kevin_uid, ...)` directly (§(d.2)). Do not report the
  library's all-user average as Kevin's result.

### (a.7) Source map (every claim above traces to one of these)

| Claim | Source |
|---|---|
| Library purpose, models, install | `README.md` (repo root); https://benfred.github.io/implicit/ |
| ALS params, `fit` semantics, negative-value dislikes | `/tmp/implicit/implicit/cpu/als.py:20-200` |
| `alpha` multiplies whole matrix | `/tmp/implicit/implicit/cpu/als.py:133-134` |
| ALS zeroing of empty rows | `/tmp/implicit/implicit/cpu/_als.pyx:98`, `:182` |
| BPR params, binary-only weights, zeroing | `/tmp/implicit/implicit/cpu/bpr.pyx:70-171`; https://benfred.github.io/implicit/api/models/cpu/bpr.html |
| `recommend` / `similar_items` / `similar_users` contract | `/tmp/implicit/implicit/recommender_base.py:34-170`; `/tmp/implicit/implicit/cpu/matrix_factorization_base.py:35-95` |
| BM25 recipe + Last.fm params | `/tmp/implicit/examples/lastfm.py`; `/tmp/implicit/implicit/nearest_neighbours.py:222-258`; https://benfred.github.io/implicit/tutorial_lastfm.html |
| Eval functions | `/tmp/implicit/implicit/evaluation.pyx:14-400` |
| CPU/GPU factory switch | `/tmp/implicit/implicit/als.py` (`use_gpu=implicit.gpu.HAS_CUDA`) |
| Papers | Hu et al. http://yifanhu.net/PUB/cf.pdf; Rendle et al. https://arxiv.org/pdf/1205.2618.pdf |
| Version 0.7.3, commit | `/tmp/implicit/pyproject.toml:13`; `git log` → `8a95dbe` |

---

## (b) The exact feature to port — DEFERRED

### (b.1) Status banner

> **DEFERRED until streaming history arrives AND the multi-user gate in §(b.6)
> passes.** No v3 code may import `implicit`, fit a model, or emit a CF score
> before then. The TDD tests in §(c) are written now; the implementation is not.
> If the gate fails, the v3 build ships the v2 content model only, and this spec
> stays deferred.

### (b.2) Feature definition

Port `implicit` ALS (primary) and BPR (binary ablation only) as a **re-ranker and
similar-items source** inside the v3 recommender, blended with the existing v2
audio-profile scores. It is not a standalone recommender and never generates
candidates on its own:

1. **Candidate generation stays content-based.** The v2 pipeline (ReccoBeats +
   librosa on Deezer 30 s previews, `recommend_v2.py` pool) proposes candidates.
   CF only scores candidates that exist in the fitted item universe.
2. **CF scoring.** For a candidate set `C` (Spotify track IDs), map each to its
   internal item id; call
   `model.recommend(kevin_uid, kevin_row, N=len(C), items=<in-matrix ids of C>)`.
   Candidates **not** in the matrix get CF score `None` (missing), never `0.0` —
   `0.0` in implicit means "model scored this at zero" and would be confused with
   a real low score.
3. **Blending.** `final = z(v2_score) + w · z(cf_score)` over candidates where both
   exist, with `w` tuned on the validation split (§(d)); `w = 0` when the gate
   fails or when a candidate has no CF score. `z()` is z-scoring within the
   candidate set for that recommendation round. Report `w` in every eval output.
4. **Similar-items expansion (secondary use).** For each seed liked track,
   `model.similar_items(seed_id, N=20)`; keep only results that are in-matrix,
   not already liked, and not in the 13 confirmed negatives (§(d.1)); feed them
   into the candidate pool for v2 audio scoring. This is the only CF path that can
   surface tracks the content pipeline did not already propose.
5. **Cold start is out of scope for CF.** A track with no plays by any user in the
   matrix has no factor vector and cannot be ranked by CF. It is scored by v2
   alone. Any v3 output that shows a CF score for such a track is a bug (test T9).

### (b.3) Exact input schema the v3 implementation will expect

#### (b.3.1) Raw input — Spotify Extended Streaming History export

- **Location (v3):** `~/workspace/audio-pipeline/data/spotify_extended_history/`
  containing `Streaming_History_Audio_*.json` (current Spotify naming) and/or
  `endsong_*.json` (older naming). Both are JSON arrays of play events. Loader
  must accept either filename pattern and must reject the basic "Account data"
  `StreamingHistory*.json` files (different, poorer schema) with an explicit error.
- **Expected date:** ~2026-11-26, per the task brief. **Not independently verified;**
  Spotify's own stated maximum preparation time is 30 days from request (see §(e)).
  No file exists in the workspace as of 2026-11-21 (test T1 asserts this).
- **Per-event fields used** (Spotify export schema; field semantics verified against
  Spotify's own `ReadMeFirst_ExtendedStreamingHistory.pdf` description as mirrored
  in public write-ups — see §(e.4) for the verification caveat):

  | Field | Type | Use |
  |---|---|---|
  | `ts` | ISO-8601 UTC string, e.g. `2024-03-15T14:23:45Z` | Event time; time-based split; `first_ts` / `last_ts` |
  | `spotify_track_uri` | string `spotify:track:<base62>` or `null` | Item key. `null` rows are dropped (unknown/local tracks) |
  | `ms_played` | int, milliseconds listened | Play qualification (§(b.3.2)) and `total_ms_played` |
  | `master_metadata_track_name` | string or `null` | Audit/display only; never the join key |
  | `master_metadata_album_artist_name` | string or `null` | Audit/display only |
  | `spotify_episode_uri` | string or `null` | Non-null ⇒ podcast episode ⇒ dropped |
  | `reason_start` / `reason_end` | string | Audit only (e.g. `trackdone`, `fwdbtn`) |
  | `skipped` | bool or `null` | Recorded in the audit file; **not** a filter by itself |
  | `incognito_mode` | bool | Recorded; not a filter |

- **Join key is always `spotify_track_uri`, never title/artist.** Title/artist
  strings are ambiguous across remasters, remixes and re-releases.

#### (b.3.2) Play-qualification rule (exact, no discretion)

An event counts as a **play** iff **all** of:

1. `spotify_track_uri` is non-null and starts with `spotify:track:`;
2. `spotify_episode_uri` is null;
3. `ms_played >= 30000` (30 s — a full Deezer-preview length and a conservative
   "actually listened" threshold; the export carries no track duration, so a
   fraction-of-duration rule is impossible from this file alone).

`skipped == true` with `ms_played >= 30000` still counts (the user listened 30 s+
before skipping). `skipped` is retained in the audit file for later analysis; it
must not silently change the count. Any change to the 30 000 ms threshold is a
spec change: re-run the eval and report the threshold alongside the metrics.

#### (b.3.3) Loader outputs (exact schemas)

Loader: `~/workspace/audio-pipeline/history_loader.py` (to be written by v3;
**not** written now). It reads the raw JSON files and writes:

**A. `data/listening_events_clean.csv`** — one row per qualifying play event:

```csv
ts,spotify_track_uri,ms_played,reason_end,skipped,source_file
2024-03-15T14:23:45Z,spotify:track:4u7EnebtmKWzUH433cf5Qv,215000,trackdone,false,Streaming_History_Audio_2024_0.json
```

Columns: `ts` (ISO-8601 UTC), `spotify_track_uri`, `ms_played` (int ≥ 30000),
`reason_end` (string, may be empty), `skipped` (`true`/`false`/empty),
`source_file` (basename). Sorted by `ts` ascending.

**B. `data/user_item_history.csv`** — the aggregated interaction table. This is the
table the matrix builder consumes. One row per (user, track):

```csv
user_id,spotify_track_uri,play_count,total_ms_played,first_ts,last_ts,distinct_days
kevin,spotify:track:4u7EnebtmKWzUH433cf5Qv,17,3124000,2024-03-15T14:23:45Z,2026-11-10T08:01:00Z,12
```

Columns and types:

| Column | Type | Constraint |
|---|---|---|
| `user_id` | string | `kevin` for Kevin (row 0 in the matrix); other users only per §(b.6) |
| `spotify_track_uri` | string | `spotify:track:<base62>`, non-null |
| `play_count` | int | ≥ 1; count of qualifying events for (user, track) |
| `total_ms_played` | int | ≥ `30000 × play_count`; sum of `ms_played` |
| `first_ts` | ISO-8601 UTC | earliest qualifying event |
| `last_ts` | ISO-8601 UTC | latest qualifying event; drives the time-based holdout |
| `distinct_days` | int | ≥ 1; number of distinct UTC dates with a qualifying event |

**C. `data/history_users.csv`** — one row per user in the matrix:

```csv
user_id,source,n_events,n_tracks,first_ts,last_ts
kevin,spotify_extended_history,48213,5211,2019-06-02T00:14:00Z,2026-11-20T23:59:00Z
```

`source` is one of `spotify_extended_history` (Kevin) or the provenance label of
the other-user source (§(b.6)); `n_events` = qualifying plays, `n_tracks` =
distinct tracks. No row may exist here without a consent/provenance note recorded
in `data/history_users_provenance.md` (who, what source, date obtained).

**D. `data/item_id_map.json`** — `{"spotify:track:...": <int item id>, ...}` plus
`{"kevin": 0, ...}` user map, written by the matrix builder. Item ids are dense
`0..n_items−1` in first-seen order; the map file is the only place the int ↔ URI
translation lives.

#### (b.3.4) Matrix construction (exact)

- Build a `scipy.sparse.csr_matrix` of shape `(n_users, n_items)`, `dtype float32`.
- **Primary (ALS) values:** BM25 weights. Follow `examples/lastfm.py` exactly:
  apply `implicit.nearest_neighbours.bm25_weight(X, K1=100, B=0.8)` to the
  **item × user** count matrix, then transpose to user × item CSR for `fit`.
  Do not apply BM25 to the user × item matrix (different weighting, §(a.4)).
- **Ablation (ALS) values:** Hu confidences `1 + alpha * play_count` with
  `alpha=2.0`, `model alpha=1.0` (so the code's whole-matrix multiply is a no-op).
  Report both; ship whichever wins on NDCG@10 (§(d)).
- **BPR values:** binary `1.0` for `play_count ≥ 1` (BPR ignores weights anyway,
  §(a.3)). BPR is a diagnostic ablation, not a shipping candidate, unless it
  beats ALS on NDCG@10 by ≥ 0.05 on the same split — in which case that result
  must be reproduced with a second seed before it is believed.
- Rows for users with zero qualifying plays are dropped before `fit` (they would
  be zeroed by the solver and contribute nothing).
- Columns (items) with zero plays cannot exist by construction (the map is built
  from observed plays only).
- **Minimum history shape (Kevin's row alone):** ≥ 500 distinct tracks with
  `play_count ≥ 1`, ≥ 5 000 qualifying plays total, date span (`last_ts −
  first_ts`) ≥ 90 days. Below any of these, the loader exits non-zero with
  `HISTORY_TOO_THIN` and no model is fit. These are chosen gates, not upstream
  facts; they exist so a 2-week export cannot silently produce a "model".

### (b.4) Hyperparameters (exact grid; selection by §(d), not by default)

| Model | Fixed | Grid searched (validation NDCG@10) |
|---|---|---|
| ALS (primary) | `dtype=np.float32`, `use_native=True`, `use_cg=True`, `iterations=15`, `num_threads=1`, `random_state=42` | `factors ∈ {16, 32, 64}`, `regularization ∈ {0.01, 0.05, 0.10}`, `alpha ∈ {1.0, 2.0}` (BM25 path only; Hu path fixes `alpha=1.0`) |
| BPR (ablation) | `learning_rate=0.01`, `regularization=0.01`, `verify_negative_samples=True`, `num_threads=1`, `random_state=42` | `factors ∈ {16, 32}`, `iterations ∈ {50, 100}` (default 100 kept as the top of the grid; 50 is the small-data concession) |
| Blend | — | `w ∈ {0.0, 0.25, 0.5, 1.0}` on z-scored CF vs v2 (§(b.2.3)) |

Defaults are **not** shipped: the library defaults (`factors=100`, ALS
`regularization=0.01`, BPR `iterations=100`) are tuned for datasets orders of
magnitude larger than any cohort we can plausibly assemble. `factors=100` on a
≤ 100-user cohort is over-parameterised; the grid caps at 64 for that reason.
`random_state=42` is fixed everywhere for reproducibility (test T7).

### (b.5) Stack mapping and VM limits

- **Install (v3 only, after gate passes):** `pip install implicit==0.7.3` into the
  audio-pipeline venv (`~/workspace/audio-env/bin/python`, the interpreter
  `evaluate.py` already uses). Pin `==0.7.3` — the exact version read for this
  spec. Requires `scipy>=0.16`, `numpy>=1.17`; VM already has Python 3.12.3,
  SciPy 1.11.4, NumPy 1.26.4 (verified 2026-11-21), all compatible with
  `requires-python >=3.9`. `implicit` itself is **not** installed in the VM today
  (verified `ModuleNotFoundError`); do not install it during the research phase.
- **CPU only.** No GPU on this VM; import from `implicit.cpu.als` /
  `implicit.cpu.bpr`. Never call `.to_gpu()`. Never install the `implicit-proc=*gpu`
  conda variant or `rmm-cu13`.
- **Threads.** VM: 1.5 shared CPUs, ~6 GB RAM (6,442,450,944 bytes shared-memory
  limit observed on exec). Set `num_threads=1` on every model (the grid's fixed
  value). Set `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in the v3 runner
  environment, per the README's optimal-configuration note — implicit's own
  OpenMP threads and BLAS threads otherwise contend on 1.5 cores.
- **Memory arithmetic (computed, not measured).** Factors are `float32`:
  `n_users × factors × 4` bytes for user factors, `n_items × factors × 4` for item
  factors. Worst plausible cohort in v3 (100 users × 50 000 items, `factors=64`):
  user factors 25 600 bytes, item factors 12 800 000 bytes (~12 MB); the CSR
  matrix itself is bounded by the number of observed (user, item) pairs, at most
  a few hundred thousand nonzeros × ~12 bytes. Total model memory is **< 100 MB**,
  two orders of magnitude under the ~6 GB limit. Even the full Last.fm 360K
  dataset (360 000 users × 64 factors × 4 B ≈ 92 MB user factors) would fit; memory
  is not the constraint. **Time is the constraint to watch:** BPR is SGD over
  epochs (default 100) and is the slowest model; the runtime test T10 caps fit
  time at 60 s wall for the expected cohort. If BPR exceeds it, cut `iterations`
  to 50 and re-test; if it still exceeds it, drop BPR and ship ALS only.
- **No new heavy dependencies.** `implicit` needs only NumPy/SciPy/tqdm/
  threadpoolctl, all already in the stack. No torch, FAISS, Essentia or Demucs
  are required for this port (they remain stretch goals per the brief and are not
  used here).
- **Determinism / persistence.** Save fitted models with `model.save()` (`.npz`)
  to `data/implicit_als_<params>.npz` alongside `data/item_id_map.json`; never
  persist a model without the item map that interprets its rows. Log
  `implicit.__version__`, all hyperparameters, the BM25/Hu choice, the matrix
  shape, nnz, and the gate verdicts into the eval JSON.

### (b.6) Gating conditions (all must be true before any `fit` call)

| # | Gate | Threshold | Why |
|---|---|---|---|
| G1 | History present | `data/spotify_extended_history/` contains ≥ 1 `Streaming_History_Audio_*.json` or `endsong_*.json` that parses as a JSON array | Without it there is no count data at all |
| G2 | Kevin's row is non-thin | §(b.3.4) minima: ≥ 500 distinct tracks, ≥ 5 000 plays, ≥ 90-day span | A thin row cannot support a time-based holdout |
| G3 | **Multi-user matrix exists** | `n_users ≥ 20`, and Kevin shares ≥ 100 items with ≥ 5 other users, and distinct items ≥ 1 000 | §(a.5): with 1 user there is no CF signal. This is the gate the brief's "history arrives" framing omits — history alone does **not** satisfy G3 |
| G4 | Item-space alignment | Every other user's items are Spotify track URIs in the same `spotify:track:` space (not artist names, not Last.fm ids) | Unaligned ids create phantom items that can never match Kevin's tracks |
| G5 | Library available | `import implicit; implicit.__version__ == "0.7.3"` in the audio venv | Pinned version only |

If any gate fails, the v3 runner logs `IMPLICIT_DEFERRED: <gate>` with the measured
numbers and continues with the v2 content model only. A failed gate is a normal
outcome, not an error to work around. **G3 is expected to fail on history-arrival
day** unless a second data source for other users has been secured by then (§(e)).

---

## (c) TDD acceptance tests — written FIRST (specification; code to be added by v3)

The tests below are specified now, before any implementation exists. They live in
`~/workspace/audio-pipeline/tests/test_implicit_cf.py` (new file, **to be created
by the v3 build, not by this research task**) with fixtures under
`~/workspace/audio-pipeline/data/fixtures/`. Run with
`~/workspace/audio-env/bin/python -m pytest tests/test_implicit_cf.py -v`.
No test may be weakened to pass; a failing test blocks the v3 CF code from shipping.

**T0 — library pinned and importable. FAILS TODAY.**
```python
def test_implicit_pinned_version():
    import implicit
    assert implicit.__version__ == "0.7.3"
```
Fails today: `ModuleNotFoundError: No module named 'implicit'` (verified 2026-11-21
from `~`). Passes only after `pip install implicit==0.7.3` in the audio venv.

**T1 — history file exists and parses. FAILS TODAY (the required test).**
```python
def test_history_files_present_and_parse():
    from history_loader import find_history_files, load_events
    files = find_history_files("data/spotify_extended_history")
    assert len(files) >= 1, "no Streaming_History_Audio_*.json / endsong_*.json found"
    events = load_events(files)
    assert len(events) >= 5000
    assert all(e["ms_played"] >= 30000 for e in events)
    assert all(e["spotify_track_uri"].startswith("spotify:track:") for e in events)
    assert all(e["spotify_episode_uri"] is None for e in events)
```
Fails today: the directory does not exist and no export has arrived (expected
~2026-11-26 per the brief). This test must fail until the export lands; a green
T1 before then means the loader is reading the wrong files.

**T2 — single-user matrix is degenerate (characterisation test).**
```python
def test_single_user_matrix_is_degenerate():
    """Proves §(a.5). If this FAILS, the degeneracy analysis is wrong: stop and revise the spec."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from implicit.cpu.als import AlternatingLeastSquares
    # 1 user x 650 liked tracks (binary), plus 200 never-played items as columns
    row = np.zeros(850, dtype=np.float32); row[:650] = 1.0
    m = csr_matrix(row.reshape(1, -1))
    model = AlternatingLeastSquares(factors=32, regularization=0.01, iterations=15,
                                    num_threads=1, random_state=42)
    model.fit(m, show_progress=False)
    ids, scores = model.recommend(0, m, N=200, filter_already_liked_items=True)
    unseen_scores = scores  # all returned ids are unseen (liked are filtered)
    assert np.all(unseen_scores == 0.0), f"expected all-zero unseen scores, got std={unseen_scores.std():.6f}"
    assert np.all(model.item_factors[650:] == 0.0)
```
Expected: **passes** (confirming degeneracy). If it fails, §(a.5) is wrong and the
spec — not the test — gets revised. Companion assertion for BPR: same setup with
`BayesianPersonalizedRanking(factors=32, iterations=5, num_threads=1,
random_state=42)` must also yield all-zero unseen scores.

**T3 — loader refuses a single-user matrix.**
```python
def test_matrix_builder_rejects_single_user():
    import pytest
    from history_loader import build_user_item_matrix  # or the v3 builder name
    with pytest.raises(ValueError, match="n_users=1"):
        build_user_item_matrix("data/fixtures/kevin_only_history.csv")
```
The builder must raise before calling `fit`, with the measured `n_users` in the
message. It must never silently fit a 1-row matrix.

**T4 — play-count aggregation is exact on synthetic history.**
```python
def test_aggregation_counts():
    # fixture: 3 events for track A (40000, 50000, 20000 ms), 1 for track B (31000 ms)
    # 20000 ms event must NOT count (below 30000 ms threshold)
    rows = aggregate("data/fixtures/synthetic_endsong.json")
    a = [r for r in rows if r["spotify_track_uri"] == "spotify:track:AAA"][0]
    assert a["play_count"] == 2 and a["total_ms_played"] == 90000
    assert len(rows) == 2  # A and B only; no row for the sub-threshold-only track
```

**T5 — confidence is monotone and saturating under BM25.**
```python
def test_bm25_confidence_monotone_saturating():
    import numpy as np
    from implicit.nearest_neighbours import bm25_weight
    from scipy.sparse import csr_matrix
    # item x user counts: item rows with counts 1, 10, 100, 1000 for a single user
    X = csr_matrix(np.array([[1.0], [10.0], [100.0], [1000.0]], dtype=np.float32))
    W = bm25_weight(X, K1=100, B=0.8).toarray().ravel()
    assert W[0] < W[1] < W[2] < W[3]
    assert W[3] / W[2] < 10.0, "1000 plays must not weigh 10x 100 plays (saturation)"
```

**T6 — recommendations exclude liked tracks and all 13 confirmed negatives.**
```python
NEGATIVE_IDS = [  # 13 confirmed negatives, Spotify IDs verified from spotify-recommended-tracks.txt
    "6aJn7Cst74cj4lNIiPRgav", "6HacgXCExkzS552ILfJTXu", "1GfBLbAhZUWdseuDqhocmn",
    "5v7oj9PosKdE39x9uwDs6C", "4bkPAARkDov1lV05emEF7a", "2jv1ovpDpTJEPIlth5Wn07",
    "7zDu3l7DDlfSlt20YMKkWo", "2y57H6nA3vic5t8awwaS69", "2slLXFcITuhf0oSp2wVJLv",
    "30efCAopZCFcoi43sSToNr", "0FmrNFYyLLRtIS7XWJPLLk", "32gSPav2xqFylAeCjJCxWi",
    "1vvBUZseJ7fwYPX1NedOLd",
]
def test_recommend_excludes_liked_and_negatives(fitted_model, kevin_row, item_map):
    neg_internal = [item_map[f"spotify:track:{s}"] for s in NEGATIVE_IDS if f"spotify:track:{s}" in item_map]
    ids, _ = fitted_model.recommend(0, kevin_row, N=10, filter_already_liked_items=True,
                                    filter_items=neg_internal)
    assert not set(ids) & set(neg_internal)
    assert len(ids) == 10
```
The 9 v5 IDs are the dropped v5 tracks (v5 had 10 entries; "Fallout" by Sleep
Theory, `3M3oltaiY0Hfl7F7IRhPSA`, was kept and is **not** a negative). The 4 v6
IDs are "prove it" (21 Savage, Summer Walker), "Akina Speed Star" (Xavier Wulf),
"Alone" (Night Lovell), "Cha Cha" (Freddie Dredd). All 13 must be excluded via
`filter_items=`; a negative appearing in a top-10 is a hard failure (§(d)).

**T7 — reproducibility.**
```python
def test_deterministic_top10(fitted_model_fn, kevin_row):
    ids1, s1 = fitted_model_fn().recommend(0, kevin_row, N=10)
    ids2, s2 = fitted_model_fn().recommend(0, kevin_row, N=10)
    assert (ids1 == ids2).all() and (s1 == s2).all()
```
Both fits use `random_state=42`; identical inputs must give identical outputs.

**T8 — offline eval beats v2 on the same pool (the §(d) gates, as a test).**
```python
def test_cf_blend_beats_v2_on_same_pool():
    res = evaluate_cf()  # returns dict with P@10 / NDCG@10 for v2, cf, blend
    assert res["blend"]["P@10"] >= res["v2"]["P@10"] + 0.10
    assert res["blend"]["NDCG@10"] >= res["v2"]["NDCG@10"] + 0.05
    assert res["blend"]["negatives_in_top10"] == 0
    assert res["blend"]["mean_rank_negative"] - res["blend"]["mean_rank_heldout"] >= 50
    assert res["recoverable_fraction"] >= 0.50, "too few held-out likes are recoverable; result is inconclusive"
```
Fails today (no history, no model). It must be run on the §(d) split, not on any
other split.

**T9 — cold-start track is missing, not zero-scored.**
```python
def test_unknown_track_has_no_cf_score():
    score = cf_score_for("spotify:track:0000000000000000000000")  # never in any history
    assert score is None
```
A return of `0.0` fails the test: `0.0` is a real implicit score for in-matrix
items and must never stand in for "unknown".

**T10 — runtime and memory on the VM.**
```python
def test_fit_runtime_and_memory():
    import resource, time
    t0 = time.time()
    fit_model(...)  # expected cohort scale: <= 100 users, <= 50 000 items
    assert time.time() - t0 <= 60.0
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 <= 1024  # MiB
```
60 s wall / 1 024 MiB peak RSS, measured on the 1.5-CPU / ~6 GB VM, `num_threads=1`.

---

## (d) Eval-harness metric — what proves it beats the v6 6/10 baseline

### (d.1) Baselines (do not conflate the two)

| Baseline | Value | What it is | What it is not |
|---|---|---|---|
| v6 human keep rate | **6/10** (second listen; 6 kept, 4 thumbs-down) | Human judgement on 10 recommended tracks | Not a P@10. Never equate the two numbers |
| v2 / v6 offline scores | Whatever `evaluate.py` reports as P@10 / NDCG@10 on its pool | Offline ranking metrics on held-out likes + negatives + distractors | Not a human keep rate |

The existing harness (`~/workspace/audio-pipeline/evaluate.py`, read 2026-11-21)
already defines the pool pattern the CF eval extends: held-out likes
(`TEST_N = 130`, i.e. 20% of 650), the 9 dropped v5 negatives (from
`data/negatives_features.json`), 200 distractors sampled from the candidate pool
with `SEED = 42`, metrics P@10 / NDCG@10 / mean rank of held-out likes /
mean rank & mean score of negatives, plus a genre sanity check. The CF eval
(`evaluate_cf.py`, to be written by v3) reuses that file's `precision_at_k`,
`ndcg_at_k` and `rank_metrics` helpers and pool discipline, with the changes in
§(d.2). It does not replace `evaluate.py`.

### (d.2) CF eval protocol (exact)

1. **Split — time-based, not random.** Train on all qualifying plays except
   Kevin's most recent 20% of distinct played tracks by `last_ts`; those are the
   test set (relevant = 1). `implicit`'s `leave_k_out_split` / `train_test_split`
   hold out random interactions and must not be used for the primary metric —
   random holdout leaks future listening into training. The v2/v6 comparison
   methods are re-run on **this same split and pool**; comparing CF-on-new-split
   against v2-on-old-split is invalid and fails the eval.
2. **Pool.** Test-set held-out likes (relevant = 1) + all **13** confirmed
   negatives (§(c) T6 list; relevant = 0) + 200 distractors (`SEED = 42`,
   same sampling discipline as `evaluate.py`). **Constraint:** every pool item
   must be in the fitted item universe — CF cannot score out-of-matrix tracks.
   Report `n_distractors_in_matrix`; if < 200, top up with additional in-matrix,
   non-liked tracks sampled with the same seed until 200, and report the top-up
   count. Items that cannot be scored by CF are excluded from the CF ranking
   **and** from the v2 ranking for that comparison (apples-to-apples; report the
   excluded counts, as `evaluate.py` already does for missing librosa features).
3. **Scoring.** Kevin-only: `model.recommend(kevin_uid, kevin_train_row, N=10,
   filter_already_liked_items=True, filter_items=<13 negative ids>)` for the
   top-10, and full-pool ranking via `items=<pool ids>` for P@10 / NDCG@10 /
   mean ranks. The library's `ranking_metrics_at_k` all-user average is reported
   for diagnosis only; the gating metrics are Kevin-only.
4. **Recoverable fraction.** Report `recoverable = |held-out likes played by ≥ 1
   other user| / |held-out likes|`. Held-out likes no other user played are
   unrecoverable by CF by construction (§(a.5)) and count as misses in the
   headline metrics — but if `recoverable < 0.50`, the offline result is
   **inconclusive regardless of the metric values** and must not be used to ship.
   Report both the headline metrics and the recoverable-subset metrics.

### (d.3) Pass thresholds (ALL must hold; numbers, no adjectives)

| # | Metric | Threshold |
|---|---|---|
| M1 | P@10 (blend) − P@10 (v2, same pool/split) | ≥ **+0.10** absolute |
| M2 | NDCG@10 (blend) − NDCG@10 (v2, same pool/split) | ≥ **+0.05** absolute |
| M3 | Confirmed negatives in CF/blend top-10 | **0 of 13** |
| M4 | mean_rank(negatives) − mean_rank(held-out likes) | ≥ **50** positions |
| M5 | Recoverable fraction (§(d.2.4)) | ≥ **0.50** |
| M6 | Human gate (after offline pass, separate playlist round) | keep rate ≥ **7/10**, beating the v6 **6/10** by ≥ 1 track |

Also report (no threshold, diagnostic only): P@10 / NDCG@10 for CF-alone,
MAP@10, AUC, per-model training loss if `calculate_training_loss=True`, the
chosen `w`, factors, regularisation, and the BM25-vs-Hu ablation delta. An
offline pass (M1–M5) **never ships on its own** — M6 is the shipping gate, and
a human round that scores 6/10 or worse sends the blend back to tuning even if
M1–M5 passed.

### (d.4) Negative control (explicit)

The 13 confirmed negatives are the control: a CF model that has learned anything
about Kevin's taste must rank all 13 below the held-out likes (M4) and keep all
13 out of the top-10 (M3). Note the control's limit: all 13 are hip-hop/trap
bucket, and 9 of them are v5 drops whose audio-profile scores already separate
them in v2 — so M3/M4 passing is necessary but weak evidence on its own. It
gains force only jointly with M1/M2 (held-out-like recovery). If M1/M2 pass but
M3 fails, the blend is rejected: it is recovering likes while still surfacing
known rejects, which is the exact v5/v6 failure mode.

---

## (e) What it cannot do on our data, and what would unblock it

### (e.1) What it cannot do

1. **It cannot recommend from liked IDs alone.** 650 binary likes = a 1 × 650
   matrix with uniform values. Per §(a.5) the fitted model scores every unseen
   track exactly `0.0` (ALS solver zeroing, `_als.pyx:98,182`; BPR explicit
   zeroing, `bpr.pyx:163-164`). There is no ranking to extract, at any factor
   count or iteration count. Tuning hyperparameters on this matrix is tuning noise.
2. **It cannot use play counts it does not have.** BPR ignores weights entirely
   (§(a.3)); ALS needs counts to build confidence, and the only counts available
   today are "1 = liked, 0 = not in the list", which carry no gradation.
3. **It cannot do cold start.** Any track no user in the matrix has played has no
   factor vector. New releases, deep cuts nobody in the cohort played, and every
   candidate the content pipeline proposes from outside the history universe are
   all unscorable by CF (test T9). CF is a re-ranker over history, not a discovery
   engine.
4. **It cannot explain taste in audio terms.** Latent factors are opaque
   coordinates; `model.explain()` (ALS, `als.py:399`) attributes a recommendation
   to other liked items and their weights, not to tempo, brightness, or any
   audio property. Audio explanations stay with the v2 content model.
5. **It cannot substitute for other users' data with public datasets as-is.**
   The library's own example dataset, Last.fm 360K (`implicit/datasets/lastfm.py`,
   `get_lastfm()`), is **artist-level** play counts with Last.fm artist ids —
   not Spotify track URIs. Its item space does not align with Kevin's (§(b.6) G4),
   so appending it would create a matrix whose "items" are two different things.
   Using it would require an artist→Spotify-track mapping pass that is a separate
   research task with its own error budget; it is not part of this port.

### (e.2) What would unblock it (in order)

| # | Unblocker | Status | Effect |
|---|---|---|---|
| U1 | **Spotify Extended Streaming History export arrives** (expected ~2026-11-26 per the brief; Spotify's stated maximum is 30 days from request) | Not arrived as of 2026-11-21 | Supplies Kevin's per-track play counts → satisfies G1, G2. Still leaves G3 failing |
| U2 | **A multi-user matrix over Spotify track URIs** — ≥ 20 users, Kevin sharing ≥ 100 items with ≥ 5 of them (§(b.6) G3) | Not available; no source identified | The actual CF unblocker. Without it, U1 alone changes nothing for implicit |
| U3 | `pip install implicit==0.7.3` in the audio venv | Not installed (verified) | Satisfies G5; 5-minute task once G1–G4 pass |
| U4 | Human listening round for M6 | Not scheduled | Shipping gate; needs a 10-track playlist from the blend |

**On U2, honestly:** there is currently no identified source of other users'
Spotify-track-level histories. Candidate sources, none secured and none assumed
by this spec: (i) listening histories voluntarily contributed by friends/family,
same URI space, with explicit consent recorded in
`data/history_users_provenance.md` — smallest N, highest alignment; (ii) public
playlist co-occurrence (tracks that appear together in public playlists treated
as implicit co-play) — different semantics from listening history, needs its own
validation that co-occurrence predicts Kevin's keeps, and is a separate spec, not
a silent substitution; (iii) a friend's API supplying full songs (sign-in pending
per the brief) — that supplies **audio**, not other users' histories, so it does
not unblock CF either; it unblocks full-audio content features, which is a
different port. **Spotify's own API does not expose other users' listening
histories**, so there is no API route to U2. If U2 cannot be secured, the correct
outcome is to leave this spec deferred permanently and keep improving the
content-based v2 model — not to simulate users.

### (e.3) Forbidden workarounds (do not do these)

- **Pseudo-users:** genre buckets, time windows (e.g. "Kevin-Jan" vs "Kevin-Feb"),
  playlists, or single tracks treated as users. Fabricates co-occurrence.
- **Liked = 1, everything else = 0, fit anyway:** produces the all-zero ranking
  of §(a.5) and invites reading tie-break order as signal.
- **Last.fm artist ids joined to Spotify tracks by name match:** name matching
  across remasters/remixes is unreliable and the error would be baked into the
  matrix silently. Only a URI-level join (U2) is acceptable.
- **Treating `similar_items` scores among liked tracks as a taste profile:**
  per §(a.5.3) those similarities are ~1.0 for every pair under one user.
- **Reporting the library's all-user average metrics as Kevin's metrics:**
  §(a.6) — Kevin-only metrics are the gating ones.

### (e.4) What was not verified

- **The export has not arrived**, so the raw schema, filename pattern
  (`Streaming_History_Audio_*.json` vs `endsong_*.json`), and field presence
  could not be checked against Kevin's actual file. The schema in §(b.3.1) is
  Spotify's documented export format as described in Spotify's own
  `ReadMeFirst_ExtendedStreamingHistory.pdf` (as mirrored at
  https://github.com/Sawrav-23/Spotify-Streaming-History-Analysis/raw/refs/heads/main/ReadMeFirst_ExtendedStreamingHistory.pdf)
  and corroborated by two independent write-ups
  (https://github.com/spotify-devs/echostats/blob/HEAD/docs/src/content/docs/user-guide/importing-history.mdx,
  https://depth.fm/guides/spotify-extended-streaming-history).
  The loader (T1/T4) must validate the real file on arrival and this spec's
  schema section must be corrected if the real file differs — particularly the
  `skipped` semantics and whether `ms_played` is present on every row.
- **The ~2026-11-26 arrival date is the brief's expectation, not a verified
  fact.** Spotify states preparation can take up to 30 days; some users report
  hours, others the full 30 days. No request confirmation or delivery email was
  available to this research task.
- **Fit behaviour was verified by code reading, not execution** (`implicit` not
  installed; §(a.5) final paragraph). Test T2 is the empirical confirmation step.
- **No other-user data source was evaluated for availability, legality, or
  consent mechanics.** U2 is specified, not sourced.
- **Deezer-preview fallback:** not applicable to this port — CF uses no audio at
  all. The fallback requirement is satisfied structurally: when this feature is
  deferred (the expected state), the v2 content pipeline on Deezer previews
  continues unchanged, and CF contributes no scores. There is no CF variant that
  runs on previews, and none is proposed.
