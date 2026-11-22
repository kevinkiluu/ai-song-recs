# Troi porting spec — ListenBrainz Troi → ai-song-recs v3

- Status: research spec for the v3 build. No code in this file; it specifies code to be written.
- Date: 2026-11-21 (research done 2026-11-22 03:12 UTC). All upstream claims as-of this date.
- Baseline to beat: v6 playlist keep rate **6/10** (second listen, Kevin).
- Confirmed negatives: **13 total** — 9 dropped v5 tracks + 4 v6 thumbs-down, all hip-hop/trap (H) bucket:
  `prove it` (21 Savage, Summer Walker, `spotify:track:30efCAopZCFcoi43sSToNr`),
  `Akina Speed Star` (Xavier Wulf, `spotify:track:0FmrNFYyLLRtIS7XWJPLLk`),
  `Alone` (Night Lovell, `spotify:track:32gSPav2xqFylAeCjJCxWi`),
  `Cha Cha` (Freddie Dredd, `spotify:track:1vvBUZseJ7fwYPX1NedOLd`).
  The 9 dropped v5 tracks are the v5 rows in `~/workspace/spotify-recommended-tracks.txt`
  except `Fallout` (Sleep Theory), the one v5 keep: Touch (KATSEYE), Census Designated
  (Jane Remover), Rumble (Skrillex, Fred again.., Flowdan), dancinwithsomebawdy (ericdoa),
  Soft Spine (Spiritbox), TOMBSTONE (Wage War), Martini Blue (DPR LIVE), Dark Light
  (Night Lovell), Opaul (Freddie Dredd). Verified by reading the history file (60 entries + header).
- Local data verified by reading: `~/workspace/spotify-liked-songs.txt` = 650 liked tracks
  (title, artist, `spotify:track:ID`; NO added dates). `~/workspace/audio-pipeline/analysis.json`:
  93 of 100 sampled liked tracks analyzed; BPM median 123.0, mean 127.6; brightness mean
  2765.6 Hz; energy mean RMS 0.31854. Per-bucket profile in `recommend.py`: E/R/H buckets,
  BPM medians 123/123/136, energy 0.320/0.322/0.341, brightness 2668/2655/2897.
  `evaluate.py` (v2 build agent, already present) implements P@10 / NDCG@10 over a pool of
  130 held-out liked tracks + 9 dropped v5 negatives + 200 distractors (seed 42).
- Stack: ReccoBeats API (Spotify-style audio features) + librosa on Deezer 30 s previews +
  scikit-learn. VM: 1.5 shared CPUs, ~6 GB RAM. No torch / FAISS / Essentia / Demucs in the
  default path (stretch only). Kevin owns no full-length audio files.

## 0. Repo-location correction (verified)

Troi is **not** a directory inside `listenbrainz-server`. It is a separate public repo and
PyPI package:

- Repo: `https://github.com/metabrainz/troi-recommendation-playground`
  (HEAD commit `044398b5b6c6c99fc301f3417e62570b2a408a40`, 874 commits, GPL-2.0, created 2020-05-09).
- Package: `troi` on PyPI, latest release shown `2026.3.3.0` (126.3 kB wheel).
- `listenbrainz-server` consumes Troi as a library: `listenbrainz/webserver/views/explore_api.py`
  does `from troi.patches.lb_radio import LBRadioPatch` and `from troi.patch import Patch`.
- A sparse clone of `listenbrainz-server` to `/tmp` confirmed there is no `troi/` directory in
  that repo. Further git work on that clone was stopped after an execution-risk flag on the
  clone result; all findings below come from the Troi docs and indexed sources, not from
  executing Troi code. Consequence: patch source files (`troi/patches/*.py`) were **not**
  read line-by-line — see §(e) / "Could not verify".

---

## (a) How the upstream mechanism works

### (a.1) Core architecture: patches are pipelines of Elements

Source: `https://troi.readthedocs.io/en/latest/introduction.html`,
`https://troi.readthedocs.io/en/latest/` (index), PyPI `https://pypi.org/project/troi/`.

- A **Patch** is a data pipeline. It is built by instantiating **Element** objects and chaining
  them with `set_sources()`. A Patch itself processes no data: "A patch never processes data
  on its own — a Patch returns a constructed pipeline of Elements that are chained together.
  The Elements are the classes that process data, but those are invoked only after the pipeline
  is constructed and when Troi begins to generate a playlist."
- An **Element** is one pipeline node. It can be a data source (fetch from an API) and/or a
  filter/transform: it takes incoming data, acts on it, passes output on. Every Element subclass
  must override `inputs()`, `outputs()`, and `read()` (the docs list `inputs()` / `outputs()` /
  `set_sources()` as the important methods; `read()` is the execution entry point per the
  Element contract as documented).
- The common datum is a **Recording** object: `Recording(name, mbid, msid, duration,
  artist_credit, release, ranking, year, spotify_id, soundcloud_id, apple_music_id,
  musicbrainz, listenbrainz, acousticbrainz)`. Entities (User, Artist, Release, Recording)
  each carry two free-form dicts, `musicbrainz` and `listenbrainz`, for pipeline data that
  does not fit the fixed fields — e.g. a Recording loaded from ListenBrainz stats carries
  `listen_count` in its `listenbrainz` dict.
- Output format: generated playlists are output in **JSPF** (JSON version of XSPF).
- API-first: "users don't need to download massive amounts of data, but instead fetch the data
  via APIs as needed." Troi's datasets are hosted by MetaBrainz (Labs API at
  `labs.api.listenbrainz.org` is the stable tier; `datasets.listenbrainz.org` endpoints are
  experimental/unstable per MetaBrainz forum, 2023-10-06).
- Troi is explicitly a playground, not scale infrastructure: "Troi is great for experimentation,
  but not for scale, so our plan has always been to graduate patches that produce solid results
  up the chain" (MetaBrainz forum, Troi local-resolution thread). Patches that work are
  re-implemented server-side in `listenbrainz-server`.

### (a.2) The recommendation input: collaborative filtering over 180 days, then recency filtering

Source: MetaBrainz forum `https://community.metabrainz.org/t/does-the-generated-playlist-use-the-all-time-listens/727785`
(staff answer), `https://community.metabrainz.org/t/will-listening-to-weekly-exploration-influence-the-weekly-exploration/693507/2`,
`https://community.metabrainz.org/t/our-weekly-recommendations-are-now-live/646950`.

- ListenBrainz computes per-user recommended recordings with a **collaborative-filtering
  algorithm over the user's listens from the last 180 days (~6 months)**, using every user's
  last 6 months of listens as the CF corpus. The 6-month window exists because the CF job
  cannot fit all listens with current infrastructure.
- From the CF pool, Troi/server **filter out any tracks the user listened to in the past
  60 days**. This 60-day exclusion was added after early Daily Jams looped the same tracks
  ("we added a filter to not recommend any tracks that you listened to in the past 60 days…
  That drastically improved things").
- The surviving pool is split into two weekly playlists, generated every **Monday at midnight
  in the user's timezone**:
  - **Weekly Jams** = recommended tracks the user **has listened to before** (the "safe"
    playlist; "all feel-good tracks with nothing new being introduced", per the Daily Jams
    patch doc — Daily Jams is the daily form of the same idea).
  - **Weekly Exploration** = recommended tracks the user **has not listened to** (the discovery
    playlist; "we choose music that you haven't listened to… but that you might want to listen to").
- The CF dataset itself is a hosted ListenBrainz dataset ("Collaborative filtered recordings
  that suggest what recordings a user should listen to based on their previous listening
  habits", PyPI/README dataset list). Troi consumes it through the
  `UserRecordingRecommendationsElement`; Troi does **not** implement CF itself. The CF
  algorithm's internals (similarity metric, neighbourhood size, ranking) are server-side and
  were not visible in any source read for this spec — do not claim them.
- Feedback: disliking ("hating") a track on ListenBrainz removes it from future recommendations
  (Daily Jams doc: "If there is a track that you don't like, dislike the track on LB and the
  track will not be recommended to you again"). This is the `HatedRecordingsFilterElement`.

### (a.3) Built-in patches (candidate generation → filtering → assembly)

Source: `https://troi.readthedocs.io/en/latest/patches.html`,
`https://github.com/metabrainz/troi-recommendation-playground/blob/HEAD/docs/patches.rst`.

| Patch slug | What it does (per docs) | Pipeline role in our port |
|---|---|---|
| `daily-jams` | Daily playlist from LB recommended recordings; "designed for the user to put on in the background… all feel-good tracks with nothing new being introduced". Day 1 = Monday, Day 2 = Tuesday, … (day-of-week slice of the weekly recs). | Template for the "safe" (Jams) output; day-slicing is history-gated (§b.3). |
| `weekly-jams` / `weekly-exploration` | Weekly forms of the listened / unlistened split (§a.2). Served via `troi weekly-jams <username>` and the LB "Created for You" page. | Template for the two-output design. |
| `recs-to-playlist` | "Fetch ListenBrainz recommended tracks and upload them to ListenBrainz as a playlist." | The minimal patch: source = CF recs element, sink = playlist. Our analogue: source = audio-neighbour + similar-artist streams. |
| `weekly-flashback-jams` | "Generate playlists for past decades based on your listening history." | Year-range filtering pattern; maps to our `YearRangeFilterElement` analogue (only usable once release years are fetched; optional). |
| `area-random-recordings` | Random tracks from a MusicBrainz area + start/end year. | Random-stream pattern; not ported (no geographic requirement). |
| `playlist-from-mbids` | Playlist from a list of MBIDs. | Identity/resolution pattern; maps to our Spotify-ID → Deezer-ID resolution step. |
| `world-trip` | Playlist across a continent, sorted by longitude/latitude. | Not ported. |
| `lb-radio` (largest patch) | See §a.4. | The main port: prompt → weighted streams → interleaved assembly. |
| `ab-similar-recordings` | "Find acoustically similar recordings from AcousticBrainz" — `troi playlist --print ab-similar-recordings <recording-MBID> mfccs`. Uses AcousticBrainz data + Annoy indexes (per PyPI README: "The AcousticBrainz project hosts the Annoy indexes that allows you to find recordings that have similar characteristics given an initial recording"). | The direct ancestor of our audio-neighbour stream (§b.2, stream S3). Troi's version is MFCC + Annoy over full audio; ours is librosa/ReccoBeats features over 30 s previews + sklearn neighbours. |

### (a.4) LB Radio: the prompt → streams → weighted interleave → filters assembly

Source: `https://troi.readthedocs.io/en/latest/lb_radio.html` (full prompt reference),
`https://troi.readthedocs.io/en/latest/user-guide.html`.

- A radio prompt is one or more **terms** of the form `entity:values:weight:option`.
  **Each term generates a stream of recordings; recordings from each stream are then
  interleaved to make a single playlist.** The optional weight controls contribution:
  default weight 1; "A term with a weight of 3 will contribute 3 times more recordings than
  a term with weight 1."
- Entities: `artist` (tracks from this artist **and similar artists**), `tag` (tracks with the
  tag), `collection` (MusicBrainz collection; mode does not apply), `playlist` (LB playlist;
  mode does not apply), `stats` (a user's LB statistics), `recs` (a user's LB recommended
  recordings), `country` (artists from the given country).
- Options: per-term mode `easy` / `medium` / `hard`; `nosim` (artist/tag: no similar
  artists/tags, only the named one); tag `and` (default: all tags) / `or` (any tag);
  stats windows `week`, `month`, `quarter`, `half_yearly`, `year`, `all_time`,
  `this_week`, `this_month`, `this_year`; recs `listened` ("safe" playlists) /
  `unlistened` ("exploration" playlists).
- **Modes = tertiles of an ordered list.** "Almost all of the data sources… are ordered lists
  of data, with the most relevant data near the top and less relevant data near the bottom.
  Broadly speaking, the three modes divide each of these datasets into three chunks: easy
  mode will focus on the most relevant data, medium on the middle relevant data and hard on
  the tail end." Easy ≈ familiar, hard ≈ less familiar / more work to listen to. Tag entity
  is special: easy fetches recordings tagged with the given tag, medium picks tags from
  release/release-group tags, hard picks tagged recordings from artists.
- Example prompt (docs): `artist:(pretty lights):3:easy tag:(trip hop):2 artist:(morcheeba)::nosim`
  = 3 parts Pretty Lights (easy), 2 parts trip-hop tag, 1 part Morcheeba with no similar artists.
- The core design line, verbatim from the docs: LB Radio's job is "to intelligently, yet
  sloppily, pick from vast lists of data to form pleasing playlists."

### (a.5) Filters, operations, sorts, and the assembly tail

Source: `https://troi.readthedocs.io/en/latest/elements/index.html`,
`https://troi.readthedocs.io/en/latest/introduction.html`.

Filters (`troi.filters`):
- `ArtistCreditFilterElement(artist_credit_ids, include=False)` — remove (or keep only)
  recordings by a given artist list.
- `ArtistCreditLimiterElement(count=2, exclude_lower_ranked=True)` — if any one artist exceeds
  `count` recordings, drop the excess; with rankings present, drop the lowest-ranked.
  **Default cap is 2 tracks per artist.**
- `DuplicateRecordingMBIDFilterElement` — drop duplicate recordings by MBID, preserving input order.
- `DuplicateRecordingArtistCreditFilterElement` — drop duplicates by recording name + artist
  credit name, preserving order.
- `ConsecutiveRecordingFilterElement` — collapse consecutive duplicates by MBID
  (A,A,A,B,B,A,C → A,B,A,C).
- `EmptyRecordingFilterElement` — drop recordings with empty name or artist.
- `YearRangeFilterElement(start_year, end_year=None, inverse=False)` — keep only recordings
  in the year range (inclusive).
- `GenreFilterElement(genre_list)` — keep recordings sharing ≥1 genre with the list.
- `LatestListenedAtFilterElement(min_number_of_days=14, keep_unlistened_if_empty=False)` —
  drop recordings whose `latest_listened_at` is more recent than 14 days ago; unlistened
  (field None) are kept. **Default recency floor is 14 days.**
- `HatedRecordingsFilterElement` — drop recordings the user hated.

Operations (`troi.operations`): `UniqueElement(key='mbid')`, `UnionElement` (concatenate),
`IntersectionElement(key='mbid')`, `DifferenceElement(key='mbid')` (list A minus list B),
`ZipperElement` (pick recordings from each input alternatingly — the interleave primitive).

Sorts: `YearSortElement(reverse=False)`.

Assembly tail: "It is not uncommon to have a `PlaylistRedundancyReducerElement` at the end of
the pipeline which returns a Playlist object of the desired length with a limit to the number
of times a single artist's tracks can be included in the playlist."

Local content resolution (for completeness; **not** ported — Kevin owns no local files):
Troi can scan a local collection (MP3/Ogg/FLAC/M4A/WMA, or Subsonic API), store metadata in
SQLite, and resolve global JSPF playlists to local files via MBID → metadata → fuzzy matching
(`nmslib-metabrainz` optional). The local periodic-jams patch resolves the global recommended
list against the local collection first, which "gives better results than the global troi patch
by the same name." Our analogue of "resolve to what is playable" is Spotify-ID → Deezer
preview-ID resolution (already in `pipeline.py`); do not build a local-file resolver.

### (a.6) What Troi does NOT do

- No audio-feature scoring of its own: similarity comes from hosted datasets (CF recs,
  similar artists/recordings, tags, AcousticBrainz MFCC/Annoy). There is no BPM/key/energy
  model in Troi.
- No per-track user rating model: the only negative signal is hate/dislike + recency exclusion.
- No evaluation harness: no precision/NDCG/offline eval is documented in Troi; quality is
  judged by listening (the Daily Jams loop test) and by the 60-day filter fixing repetition.
  Our eval harness (§d) is therefore new work, not a port.
- No de-duplication against a "previously recommended, user rejected" list beyond hates:
  the 60-day listened filter is the repetition control. Our 60-entry history exclusion is
  stricter and must be kept (§b.2, filter F1).

---

## (b) Exact feature to port, mapped onto our stack

### (b.1) The ported feature (one sentence)

A Troi-style **patch pipeline** for ai-song-recs: declarative multi-stream candidate
generation (seed-artist / tag / audio-neighbour / same-artist streams with LB-Radio weights
and easy/medium/hard tertile modes), chained through Troi's exact filter set (dedupe,
per-artist cap 2, hated/negative exclusion, recency exclusion), assembled by weighted
round-robin interleave (Zipper) and a redundancy-reducing tail into a 10-track playlist,
emitting two outputs — a safe **Jams** list and an **Exploration** list — with every stage
independently testable and every constant named in one place.

New module (v3 build writes it; this spec does not): `~/workspace/audio-pipeline/troi_patch.py`,
plus tests in `~/workspace/audio-pipeline/tests/test_troi_patch.py`. No changes to the v2
files in flight; `troi_patch.py` imports scoring helpers from `scoring.py` / `pipeline.py`
(read-only use) and must not modify `spotify-liked-songs.txt` or append to
`spotify-recommended-tracks.txt`.

### (b.2) Liked-songs-only variant (works today — the v3 default)

Data available: 650 liked (title, artist, spotify:track:ID); 60-entry recommendation history;
13 confirmed negatives; librosa/ReccoBeats features for candidates (Deezer 30 s previews);
per-bucket audio profile (E/R/H) from `analysis.json` + `recommend.py` PROFILE.
Data NOT available: play counts, skip counts, timestamps, listen recency, release years
(unless fetched per-candidate), MusicBrainz IDs (unless resolved per-track).

**Entities (Troi → ours).** `Recording` dataclass with fields:
`title, artist, spotify_id, deezer_id, mbid=None, ranking=None, year=None`,
`listenbrainz={}` (empty in this variant; keys `listen_count`, `latest_listened_at` are
`None`), `musicbrainz={}` (keys `tags`, `similar_artists` when fetched).
`ranking` = audio-fit score rank (lower = closer to bucket profile), which is what
`ArtistCreditLimiterElement(exclude_lower_ranked=True)` uses to decide which duplicate to drop.

**Element base (Troi → ours).** Every stage is a class with `inputs()`, `outputs()`,
`set_sources(sources)`, `read()` — same contract as Troi's `Element`. A `Patch` class holds
the chain and executes it lazily on `generate()`. This is a re-implementation of the
architecture, not a copy of Troi code (GPL-2.0; do not vendor Troi source).

**Candidate streams (each is a Troi source Element; each returns an ordered list).**
Candidate pool cap: **300 candidates per run** (hard cap; see §b.4 for the VM math).
Per-stream caps sum to ≤ 300 before dedupe.

| Stream | Troi analogue | Our implementation (liked-only) | Ordered by | Mode semantics (tertiles of the ordered list) |
|---|---|---|---|---|
| S1 similar-artist | `artist:(X)` (similar artists included) | For each of the top 20 seed artists by liked-track count, fetch similar artists from the ListenBrainz Labs API similar-artists endpoint, then take each similar artist's top tracks (Deezer artist top-tracks). Cap 120 candidates. | Labs similarity rank, then liked-count of seed | easy = top tertile of similar artists; medium = middle tertile; hard = bottom tertile |
| S2 tag | `tag:(...)` | Pseudo-tag = E/R/H bucket. Candidates are tracks whose bucket (from `recommend.py` bucket assignment of the seed track that generated them) matches the target bucket. Implemented as a filter on S1/S3 output, not a separate fetch, in the liked-only variant. Full tag fetch via Labs tag-similarity is history-gated (§b.3) / unverified endpoint (§e). | bucket-profile distance | easy = same bucket only; medium = same bucket + adjacent bucket (E↔R, R↔H); hard = any bucket |
| S3 audio-neighbour | `ab-similar-recordings` (AcousticBrainz MFCC + Annoy) | sklearn `NearestNeighbors` (n_neighbors=10, metric=euclidean on standardized features) over the 5-dim librosa vector `[bpm, brightness_hz, energy_rms, harshness_zcr, key-match]` — key as 0/1 same-key-or-relative-major/minor vs seed, not a continuous dimension — fitted on the 93 analyzed liked tracks (+ any further analyzed liked tracks, max 650). Query each seed track's vector; union the neighbours. Cap 120 candidates. This replaces Annoy: at n ≤ 650, brute-force sklearn is exact and cheaper than an ANN index. | distance to seed vector (ascending) | easy = neighbours at distance rank 1–3 per seed; medium = ranks 4–7; hard = ranks 8–10 |
| S4 same-artist | `artist:(X)::nosim` | Unliked tracks by the top 20 seed artists (Deezer artist top-tracks minus liked set). Cap 60 candidates. | Deezer top-track rank | mode does not apply (Troi: collections/playlists have no ranking; same-artist top-tracks do have one — easy = top half, hard = bottom half; record the choice) |

Seed-artist definition (liked-only): artists ranked by count of liked tracks in the 650 file,
ties broken by alphabetical artist name (deterministic). Top 20. This is a **proxy** for
Troi's `stats` entity (top artists by listen count): liked-count ≠ play-count, and the spec
requires the code comment and the run log to say so. No play-count weighting is possible
until §b.3.

**Prompt (LB-Radio syntax, adapted).** The v3 default prompt, evaluated as one string the
patch parses (`entity:values:weight:option` per term, weights default 1):

```
audio-neighbour:liked-seeds:3:easy similar-artist:top20:2:easy same-artist:top20:1:easy
```

i.e. 3 parts audio-neighbour (easy), 2 parts similar-artist (easy), 1 part same-artist (easy)
→ Zipper weights 3:2:1. The Exploration output uses the same prompt with `:medium` on the
audio-neighbour and similar-artist terms and one `:hard` term added at weight 1
(`audio-neighbour:liked-seeds:1:hard`), giving a 10-track list of 7 easy/medium + 3 hard
(see assembly below). Prompt parsing must accept Troi's `entity:values:weight:option` grammar
including omitted weight (= 1) and `::nosim`.

**Filters (Troi name → ours, in pipeline order).** All run after stream union, before assembly:

- F1 `DifferenceElement` (×2): candidates minus liked set (650 Spotify IDs + normalized
  title+first-artist pairs) minus history set (all 60 entries). Must remove 100% of both sets;
  test T-03/T-04 assert 0 survivors.
- F2 `HatedRecordingsFilterElement`: drop the 13 confirmed negatives by Spotify ID and by
  normalized title+first-artist (both keys; the ID key alone would miss a re-release under a
  new ID). 0 of 13 may survive; test T-05.
- F3 `DuplicateRecordingMBIDFilterElement` + `DuplicateRecordingArtistCreditFilterElement`:
  dedupe by Spotify ID, then by normalized (title, first artist), preserving first-seen order
  (highest-ranked stream first). Test T-02.
- F4 `EmptyRecordingFilterElement`: drop candidates with empty title or artist, or with no
  Deezer preview match (unresolvable = unplayable-for-scoring; count them in the run log as
  `unresolved`, with the `analysis.json` precedent of 7/100 unmatched).
- F5 `ArtistCreditLimiterElement(count=2, exclude_lower_ranked=True)`: max **2** tracks per
  artist in the final 10, dropping the worse audio-fit rank. Troi default is 2; keep 2.
  Test T-06. Note: v6's 4 thumbs-down include 2 artists that also appear in v5 drops
  (Night Lovell, Freddie Dredd) — the cap does not substitute for the negative filter.
- F6 `GenreFilterElement` (bucket version): candidate's bucket must equal the target bucket of
  the run (E, R, or H per playlist brief). The v3 default run is per-bucket: generate one
  10-track list per bucket? No — the user-facing output is a single 10-track playlist; the
  bucket mix follows the liked distribution. Decision: **do not hard-filter by bucket**;
  instead record bucket as a feature and assert the final-10 bucket mix in the eval (§d).
  Rationale: Troi's GenreFilter is opt-in per patch, and v6's failure mode was H-bucket
  audio fit, not bucket membership. F6 is implemented but **disabled by default**
  (`genre_filter=None`); enabling it is a one-line prompt change.
- F7 `LatestListenedAtFilterElement(min_number_of_days=14)`: **NO-OP in the liked-only
  variant** — there is no `latest_listened_at` for any track. The element must exist, log
  `latest_listened_at_filter: skipped, reason=no timestamps (n=0 with data)`, and return its
  input unchanged. Silently dropping tracks here would be inventing data. Test T-08 asserts
  the no-op and the log line. History variant activates it (§b.3).
- F8 `ConsecutiveRecordingFilterElement`: after Zipper interleave, collapse consecutive
  same-artist tracks (same artist back-to-back is a playlist defect even under the cap of 2).

**Assembly (Zipper + redundancy reducer).**
- `ZipperElement`: weighted round-robin over the streams in prompt order, weights 3:2:1,
  cycling until 10 slots are filled or streams are exhausted; a stream that runs dry is
  skipped, not padded by repeating. Deterministic given a fixed seed (seed 42 for any
  shuffle/tie-break, same seed as `evaluate.py`).
- `PlaylistRedundancyReducerElement` (tail): truncate to exactly 10; enforce the per-artist
  cap of 2 a final time; if truncation would leave < 10 after cap enforcement, backfill from
  the next-ranked surviving candidates (same stream order), never by re-adding a filtered track.
- Outputs: **Jams (safe)** = prompt with all terms `:easy`, streams S4+S1 only, i.e. artists
  Kevin already likes, closest audio fit — the liked-only analogue of Weekly Jams (label it
  "Jams (liked-only analogue)" in output, never "Weekly Jams", because Troi Jams requires
  listened-before data we do not have). **Exploration** = full prompt above (7 easy/medium +
  3 hard by construction: positions 1–7 from easy/medium terms, positions 8–10 from the
  `:hard` term; test T-10 asserts exactly 3 hard-mode tracks in slots 8–10). Both lists are
  10 tracks, disjoint (a track in Jams must not appear in Exploration in the same run;
  test T-11), and both exclude liked/history/negatives.

**Scoring inside the pipeline (not Troi's — Troi has none).** Candidate ranking within each
stream is by the v2 audio-fit score (`scoring.py`: GMM cluster distance + learned LR log-odds;
higher = better) where available, falling back to the v6 per-bucket profile distance
(`recommend.py` `score()`, lower = closer) when v2 features are absent. The Troi port does
not change the scorer; it changes **what gets scored** (streams instead of a hand-written
`CANDIDATES` list) and **what survives to the playlist** (filters + interleave instead of
top-10 by score). This is the falsifiable claim of the port: same scorer, better candidate
generation + assembly → higher keep rate.

### (b.3) Listening-history variant (gated on the streaming export, expected ~Nov 26)

Gate condition (all must be true before any history-variant code path runs; otherwise the
run aborts with `history_variant: gated, reason=<first failing condition>`):
1. A Spotify Extended Streaming History export (or the friend's API, if signed in) is present
   as JSON with, per record, `ts` (timestamp), `master_metadata_track_name`,
   `master_metadata_album_artist_name`, `ms_played`, `spotify_track_uri`.
2. ≥ 1,000 plays in the file (below that, per-track listen counts are too sparse for the
   180-day/60-day windows; the threshold is a build-time constant `MIN_PLAYS = 1000`).
3. The export covers ≥ 60 days (else the 60-day exclusion window is meaningless).

When gated on, the following Troi semantics activate **exactly** (numbers are Troi's, not ours):
- H1 — Profile window: seed ranking uses listens from the **last 180 days** only (Troi CF
  window). `listen_count_180d` per track/artist replaces liked-count for seed-artist ranking.
- H2 — Exploration recency exclusion: drop any candidate the user listened to in the **past
  60 days** (Troi's 60-day filter), applied as `DifferenceElement` against the 60-day listened
  set. This replaces the liked-only F1's liked-set subtraction for the Exploration output
  (liked tracks not played in 60 days may re-enter as candidates; liked tracks played in the
  last 60 days may not).
- H3 — `LatestListenedAtFilterElement(min_number_of_days=14)`: active with Troi's default 14.
  `latest_listened_at` = max `ts` per track in the export.
- H4 — True Jams/Exploration split: Jams = candidates with `listen_count > 0` (listened
  before, any time); Exploration = candidates with `listen_count == 0` (never listened).
  The liked-only "analogue" labels are dropped.
- H5 — Skip proxy (our definition, to be validated, not Troi's — Troi has no skip concept):
  a play with `ms_played < 30,000` counts as a **skip**; a track with ≥ 3 skips and 0
  completions (`ms_played ≥ 120,000` or ≥ 80% of track duration where duration is known)
  joins the negative set for that run. Log skip counts; test T-13 asserts the definition on
  synthetic data. If validation shows the proxy mislabels intentional short plays, the
  threshold moves only with a recorded decision, not silently.
- H6 — Daily-Jams day-slicing: the weekly rec pool is sliced by weekday (Day 1 = Monday, …)
  for a daily output. Only meaningful with ≥ 7 days of history; otherwise disabled.
- H7 — `YearRangeFilterElement` activates if release years are fetched (via MusicBrainz or
  ReccoBeats metadata): Weekly Flashback analogue = Jams output restricted to one decade.
  Optional; default off.

Still NOT available even with history (see §e): collaborative filtering (needs other users'
histories / LB's hosted CF dataset), similar-user data, and any Troi dataset that is computed
server-side. Optional stretch (not required for v3): if Kevin creates a ListenBrainz account
and imports the export, call LB's hosted CF recommendation endpoint
(`api.listenbrainz.org/1/cf/recommendation/user/<user>/recording`, endpoint path as seen in
forum reports — **unverified**, must be verified live before use) as an *additional* candidate
stream S5, with its output scored by the same audio-fit scorer and subject to the same
filters. S5 is a fetch, not a port of CF.

**Deezer-preview-only fallback (mandatory in both variants).** Every feature used for ranking
must be computable from a Deezer 30 s preview (librosa) or the ReccoBeats API. If the friend's
full-song API becomes available, full-audio features are a **conditional** upgrade: same
feature names, recomputed from full audio, gated behind `FULL_AUDIO_AVAILABLE` (default
False) and an A/B comparison (same candidates, preview features vs full-audio features,
P@10 delta reported). No pipeline stage may require full audio. No stage may require a
local audio file (Kevin owns none).

### (b.4) VM / resource budget (1.5 shared CPUs, ~6 GB RAM)

- Candidate cap 300/run is set by: 300 Deezer preview fetches + librosa analyses dominate
  runtime; the existing pipeline already analyzed 93–100 tracks per run in `analysis.json`,
  so 300 is a 3× scale-up, not an order of magnitude. If a run exceeds the budget below,
  reduce the per-stream caps (S1 120 → 80, S3 120 → 80) before touching anything else.
- sklearn `NearestNeighbors` on ≤ 650 × 5-dim vectors: trivial (< 1 s, < 50 MB). No Annoy,
  no FAISS, no torch in the default path. (Troi used Annoy because AcousticBrainz indexes
  millions of recordings; we have 650 seeds.)
- No network fan-out beyond: Labs similar-artists (≤ 20 seed artists × 1 call), Deezer
  artist top-tracks (≤ 40 artists × 1 call), Deezer preview downloads (≤ 300 × ~500 KB).
  Cache all API responses under `~/workspace/audio-pipeline/data/troi_cache/` with the fetch
  date in the filename; never re-fetch within 7 days.
- Budget tests (T-14): full `troi_patch.py` run over the 300-candidate cap completes in
  **< 600 s wall** on this VM and peak RSS **< 4.5 GB** (measured with `/usr/bin/time -v`).
  These are budgets, not measured results — the first v3 run records the actuals.

---

## (c) TDD acceptance tests — written FIRST

Rules: write `tests/test_troi_patch.py` before `troi_patch.py`. Each test must **fail** on the
empty implementation (import error / `NotImplementedError`), then pass after implementation.
Run with the audio venv: `~/workspace/audio-env/bin/python -m pytest tests/test_troi_patch.py -v`.
Use the real data files (650 liked, 60 history, 13 negatives) — no mocks for the exclusion
tests. Synthetic data only where the real files cannot supply the case (T-07, T-09, T-13).
Seed 42 everywhere randomness is involved.

| ID | Test (name) | Setup | Assertion (falsifiable) |
|---|---|---|---|
| T-01 | `test_patch_chain_assembles_in_order` | Build patch with streams S1→F1→F3→F5→Zipper→Reducer; call `generate()` | `patch.stages` == the declared order; `generate()` returns a `Playlist` object with exactly 10 recordings. Fails before implementation: `ImportError`. |
| T-02 | `test_duplicate_filters_remove_both_kinds` | Candidate list with the same Spotify ID twice and the same (title, artist) under two different IDs | Output contains each track once by ID and once by (title, artist); input order of first occurrences preserved. |
| T-03 | `test_difference_removes_all_liked` | Candidate pool = 50 liked tracks (from the 650 file) + 50 non-liked | 0 of the 50 liked Spotify IDs survive F1. |
| T-04 | `test_difference_removes_all_history` | Candidate pool = all 60 history entries + 40 fresh | 0 of the 60 history Spotify IDs survive F1. |
| T-05 | `test_hated_filter_removes_all_13_negatives` | Candidate pool includes all 13 negatives by ID, plus `prove it` re-listed under a fake new Spotify ID with identical title/artist | 0 of 13 survive (12 by ID, 1 by title+artist fallback). |
| T-06 | `test_artist_cap_is_2_drop_lowest_ranked` | One artist with 4 candidates ranked 1–4 by audio fit | Exactly 2 survive, and they are ranks 1 and 2. |
| T-07 | `test_zipper_weights_3_2_1` | Three synthetic streams A(≥6 items), B(≥6), C(≥6), weights 3:2:1 | First 12 output positions are exactly A,A,A,B,B,C,A,A,A,B,B,C. |
| T-08 | `test_latest_listened_filter_noop_without_timestamps` | Liked-only run (no history file loaded) | Output == input (same IDs, same order); run log contains `latest_listened_at_filter: skipped, reason=no timestamps`. |
| T-09 | `test_mode_tertiles` | Ordered synthetic list of 90 items, modes easy/medium/hard | easy draws only from ranks 1–30, medium only 31–60, hard only 61–90. |
| T-10 | `test_exploration_has_3_hard_in_slots_8_to_10` | Full Exploration prompt run | Positions 8, 9, 10 are hard-mode tracks; positions 1–7 contain 0 hard-mode tracks. |
| T-11 | `test_jams_and_exploration_disjoint` | One run producing both outputs | `set(jams_ids) ∩ set(exploration_ids)` == ∅; both have length 10. |
| T-12 | `test_determinism_seed_42` | Run `generate()` twice, same inputs, seed 42 | Both 10-track outputs are byte-identical (same IDs in same order). |
| T-13 | `test_skip_proxy_definition` (history-gated; xfail without history) | Synthetic history: track X played 3× at 12 s, 0 completions; track Y played 1× at 200 s | X is in the run negative set; Y is not. Marked `pytest.mark.xfail` until the history gate opens. |
| T-14 | `test_resource_budget` | Full run, 300-candidate cap, `/usr/bin/time -v` | Wall < 600 s; peak RSS < 4.5 GB. Records actuals to the run log. |
| T-15 | `test_prompt_parser` | Parse `artist:(pretty lights):3:easy tag:(trip hop):2 artist:(morcheeba)::nosim` (Troi docs example) | Parsed terms == [(artist, [pretty lights], 3, easy), (tag, [trip hop], 2, None), (artist, [morcheeba], 1, nosim)]; omitted weight defaults to 1. |
| T-16 | `test_no_leakage_into_eval_pool` | The 130 held-out liked tracks (§d) are excluded from seed-artist ranking and from the S3 neighbour fit | Held-out IDs appear in neither the seed-artist input nor the neighbour index; assert by ID set intersection == ∅. |

Regression guard: after implementation, re-run the full existing suite (`evaluate.py`) and
record P@10 / NDCG@10 before and after the Troi patch is wired in; the patch must not
degrade the v2 scorer's offline numbers (§d).

---

## (d) Eval-harness metric that proves the feature beats v6 6/10

### (d.1) Offline harness (reuse + extend `evaluate.py`)

`evaluate.py` already defines the protocol; extend it, do not replace it:
- Pool: **130 held-out liked tracks** (test = most recent 20% by list position) as relevant
  (rel = 1); **13 confirmed negatives** (extend the current 9 v5-only set with the 4 v6
  H-bucket thumbs-down); **200 distractor tracks** sampled from the candidate pool, seed 42.
  Total pool = 343. Only items with librosa features are ranked; excluded counts are reported.
- **Correction to `evaluate.py`'s framing:** "most recent 20% by list position" is a
  positional proxy — `spotify-liked-songs.txt` has NO added dates, so list position is not
  known to be chronological. Report it as "held-out by list position (date order unverified)".
  When the streaming export lands, re-split by actual last-played date and report both splits.
- Metrics (exact definitions, same as `evaluate.py`): **P@10** = fraction of the top 10 that
  are held-out likes; **NDCG@10** with binary relevance and log2 discount; **mean rank of
  held-out likes**; **mean rank of the 13 negatives**; **count of negatives in the top 10**.
- Comparison: same pool, same features, three rankers — (i) v2 scorer alone (current
  `evaluate.py` v2 method), (ii) v2 scorer + Troi patch pipeline (streams + filters + Zipper;
  the v3 candidate), (iii) v6 per-bucket profile baseline (v6b). The patch changes candidate
  generation and assembly, not the scorer, so (i) vs (ii) isolates the port's contribution.

### (d.2) Pass thresholds (all must hold; a single number never passes the feature)

| # | Metric | Threshold | Why this number |
|---|---|---|---|
| M1 | P@10, v3 (ii) minus P@10, v2 (i) | **≥ +0.10 absolute** | 0.10 is larger than run-to-run jitter from the 200-distractor seed; smaller deltas are noise at n=343. Random P@10 on this pool is 130/343 ≈ 0.38, so the bar is above chance and above v2. |
| M2 | NDCG@10, v3 (ii) minus NDCG@10, v2 (i) | **≥ +0.10 absolute** | Same noise rationale; NDCG also penalizes relevant items buried below rank 10. |
| M3 | Negatives in top 10 (v3) | **0 of 13** | The 13 are confirmed rejects; any of them ranking top-10 is a pipeline failure, not a near-miss. (v6 put 4 of 4 H-bucket picks in the playlist and all 4 were rejected.) |
| M4 | Mean rank of the 13 negatives (v3) | **≥ 200 of 343** (bottom half of the pool) | Proves the negative signal generalizes beyond the exact 13 IDs (audio-fit demotion of the H-bucket failure region), not just list exclusion. Note F2 already excludes the 13 IDs from the playlist; M4 is measured on the *ranker*, before F2, so it tests the scorer, not the filter. |
| M5 | Mean rank of held-out likes (v3) | **lower (better) than v2 (i) by ≥ 10 ranks** | Guards against P@10 improving by luck at the cutoff while the bulk of likes sink. |
| M6 | Human keep rate (Kevin, second listen) | **≥ 7/10** on the next 10-track playlist | Beats the v6 6/10 baseline by 1 keep. This is the only metric that can pass the feature for production use; M1–M5 are offline gates. A playlist that passes M1–M5 but scores 6/10 or below with Kevin does not beat v6. |
| M7 | Leakage / integrity | **0** held-out liked IDs in the seed set or neighbour fit (T-16); **0** history IDs in the output; **0** duplicate (title, artist) pairs in the output | Any violation invalidates M1–M6 for that run. |

### (d.3) Reporting

Every eval run writes `evaluation_troi.json` next to `evaluation_v2.json` with: P@10, NDCG@10,
mean ranks, negatives-in-top-10, per-item ranks for the 13 negatives (audit), the prompt
string used, stream counts before/after each filter (F1–F8), and wall time / peak RSS.
If M1 or M2 fails but M6 passes (or vice versa), report both and do not claim the feature
"beats v6" — state which gate failed and by how much. **No adjectives in place of numbers:**
the v3 report sentence is "P@10 = x.xx (v2 y.yy, Δ +0.zz); NDCG@10 = x.xx (Δ +0.zz);
negatives in top 10 = 0/13; mean negative rank = NNN; Kevin keep rate = k/10", or it is not filed.

---

## (e) What it cannot do on our data and what would unblock it

| # | Cannot do | Why (data gap) | What would unblock it | Status |
|---|---|---|---|---|
| U1 | Collaborative filtering (Troi's actual recommender) | CF needs every user's 180-day listens; we have one user, zero listens, zero other users. Troi does not implement CF; it consumes LB's hosted CF dataset. | (a) Kevin creates a ListenBrainz account and imports Spotify Extended Streaming History (or the friend's API supplies plays), then LB computes CF for him; Troi/v3 fetches it as stream S5. (b) No local substitute exists at n=1 user. | Blocked; S5 is a fetch, not a port. |
| U2 | True Weekly Jams (listened-before) / Weekly Exploration (never-listened) split | Split key is `listen_count > 0` vs `== 0`; liked ≠ listened (a liked track may be unplayed for years; an unliked track may be played daily). | Streaming export with per-track play counts (§b.3 gate). Expected ~Nov 26. | Gated; liked-only analogues ship in v3. |
| U3 | 60-day recency exclusion / 14-day LatestListenedAt filter | Both need `latest_listened_at` per track; we have no timestamps at all. | Same export as U2. The filter code ships in v3 as a logged no-op (F7) and activates when the gate opens. | Gated. |
| U4 | Time-windowed profiling (180-day taste window) | Needs dated plays; liked file is undated. Seed-artist ranking by liked-count is a static proxy and will overweight old taste. | Same export as U2; re-split eval by actual date when it lands (§d.1). | Gated. |
| U5 | Skip / completion negative signal | No play durations or skip events. The 13 confirmed negatives are explicit rejects, not behavioural skips. | Export with `ms_played` (H5 skip proxy, threshold 30 s / 3 skips, to validate). | Gated. |
| U6 | Daily-Jams day-slicing | Needs a weekly rec pool sliced by weekday. | History (H6) + ≥ 7 days of coverage. | Gated. |
| U7 | Year/decade filtering (Weekly Flashback) | Release years are not in the liked file. | Per-candidate year fetch (MusicBrainz/ReccoBeats) — feasible in v3 as an optional filter, but unvalidated until a run measures coverage (what % of candidates return a year). | Optional; measure first. |
| U8 | AcousticBrainz-style MFCC similarity at Troi scale | Troi's Annoy indexes cover millions of recordings from full audio; we have 650 seeds and 30 s previews. | Nothing to unblock — the scale does not exist locally and is not needed: sklearn neighbours on ≤ 650 seeds is exact. Do not build an ANN index. | Won't do; by design. |
| U9 | Content resolution to a local library | Kevin owns no audio files; Troi's local resolver (SQLite + fuzzy matching + nmslib) has nothing to resolve against. | N/A — out of scope. Spotify-ID → Deezer-preview resolution is the analogue and already exists. | Won't do. |
| U10 | "Similar users" / taste neighbours | Needs a user population. | LB account + LB's similar-users data (fetch, not port). | Blocked. |

**Honest summary of the port's value.** With liked IDs only, the portable part of Troi is
the *assembly discipline* — multiple candidate streams, weighted interleave, dedupe,
per-artist cap, negative exclusion, and the Jams/Exploration two-output shape — not the
*recommendation intelligence*, which in Troi is the hosted CF dataset plus listening history.
The v3 hypothesis is therefore narrow and falsifiable: same audio scorer, Troi-style
candidate generation + assembly, measured by §d, beats v6's hand-picked candidates + top-10
by score. If M1/M2 fail, the port added ceremony, not quality, and the spec says so.

## Sources

- Troi docs (read 2026-11-22): `https://troi.readthedocs.io/en/latest/` (index),
  `https://troi.readthedocs.io/en/latest/introduction.html` (Patch/Element/Recording contract),
  `https://troi.readthedocs.io/en/latest/patches.html` (built-in patches),
  `https://troi.readthedocs.io/en/latest/lb_radio.html` (prompt reference: entities, weights,
  modes as tertiles, worked examples),
  `https://troi.readthedocs.io/en/latest/elements/index.html` (filters/operations/sorts with
  defaults: ArtistCreditLimiter count=2, LatestListenedAt 14 days),
  `https://troi.readthedocs.io/en/latest/user-guide.html` (`troi weekly-jams <user>`,
  `troi lb-radio easy 'artist:(...)'`).
- Repo: `https://github.com/metabrainz/troi-recommendation-playground` (HEAD 044398b,
  GPL-2.0); patches doc at
  `https://github.com/metabrainz/troi-recommendation-playground/blob/HEAD/docs/patches.rst`;
  PyPI `https://pypi.org/project/troi/` (2026.3.3.0).
- Server integration: `https://github.com/metabrainz/listenbrainz-server/blob/master/listenbrainz/webserver/views/explore_api.py`
  (`from troi.patches.lb_radio import LBRadioPatch`).
- CF windows + 60-day filter + weekly split: MetaBrainz forum
  `https://community.metabrainz.org/t/does-the-generated-playlist-use-the-all-time-listens/727785`
  (180-day CF window; filter out tracks listened to in past 60 days),
  `https://community.metabrainz.org/t/will-listening-to-weekly-exploration-influence-the-weekly-exploration/693507/2`
  (60-day filter added after Daily Jams looping),
  `https://community.metabrainz.org/t/our-weekly-recommendations-are-now-live/646950`
  (Monday-midnight generation; Exploration = never listened),
  `https://community.metabrainz.org/t/would-you-like-to-test-drive-our-first-recommendations-feature/626352?page=3`
  (dislike removes a track from future recommendations).
- Graduation / scale + local resolution: MetaBrainz forum
  `https://community.metabrainz.org/t/troi-local-resolution-of-recordings/703571`
  ("Troi is great for experimentation, but not for scale… graduate patches up the chain";
  local periodic jams resolves global recs against the local collection first),
  `https://community.metabrainz.org/t/sort-troi-patches-by-stability-and-use/658650`
  (Labs API stable tier vs datasets host; similar-artists / similar-recordings / tag-similarity viewers).
- Acoustic similarity: MetaBrainz forum
  `https://community.metabrainz.org/t/acoustic-similarity-in-acousticbrainz/549112`
  (`ab-similar-recordings <MBID> mfccs`; Annoy indexes hosted by AcousticBrainz).
- Local files read for grounding (not modified): `~/workspace/spotify-liked-songs.txt` (650),
  `~/workspace/spotify-recommended-tracks.txt` (60 entries; 13 negatives derived as in the
  header), `~/workspace/audio-pipeline/analysis.json` (93/100 analyzed; BPM median 123.0,
  brightness mean 2765.6 Hz, energy mean RMS 0.31854),
  `~/workspace/audio-pipeline/recommend.py` (PROFILE E/R/H; CANDIDATES hand list),
  `~/workspace/audio-pipeline/evaluate.py` (P@10/NDCG@10 protocol; 130 held-out + 9 negatives
  + 200 distractors, seed 42).

## Could not verify

1. **Patch source code line-by-line.** `troi/patches/*.py` (daily-jams, lb-radio, periodic-jams,
   ab-similar-recordings) contents were not read: the GitHub search index surfaced only docs,
   and a sparse clone of `listenbrainz-server` contains no `troi/` directory (Troi is a separate
   package). All mechanism claims above rest on the official docs + forum statements by
   MetaBrainz staff, cited per claim. If the v3 build needs exact function names beyond the
   documented Element/filter classes, it must read the repo files at
   `https://github.com/metabrainz/troi-recommendation-playground/tree/main/troi/patches`
   (path not verified via a returned link — confirm it exists before citing it) or install
   the `troi` PyPI package in a venv and inspect it.
2. **CF algorithm internals.** Similarity metric, neighbourhood size, and ranking formula of
   ListenBrainz's collaborative filter are server-side and not documented in any source read.
   This spec claims only the verified inputs/outputs (180-day window, 60-day exclusion,
   listened/unlistened split).
3. **Labs API endpoint paths/parameters** for similar-artists / similar-recordings /
   tag-similarity. Existence of the viewers is verified (forum, §Sources); exact request paths,
   auth, and rate limits are not. S1 must verify them live before the v3 build depends on them;
   if they are unusable, S1 falls back to Deezer's related-artists endpoint with the same
   tertile-mode contract, and the substitution is recorded in the run log.
4. **Daily-Jams day-slicing rule** beyond "Day 1 = Monday, Day 2 = Tuesday…" (forum). The exact
   slice sizes are not documented; H6 treats it as weekly-pool ÷ 7 and validates by inspection.
5. **Whether PyPI `troi` 2026.3.3.0 matches repo HEAD 044398b.** Version-to-commit mapping was
   not verified; cite the docs as of 2026-11-22, not a release number, for behaviour claims.
6. **Timing/resource numbers in §b.4** are budgets, not measurements. No v3 run has happened;
   the first run must record actual wall time and peak RSS (T-14).
7. A system execution-risk flag fired on the `/tmp/listenbrainz-server` clone result during
   this research; no further git/package commands were run on that clone, and no Troi code was
   executed, installed, or copied. Nothing in this spec depends on executing upstream code.
