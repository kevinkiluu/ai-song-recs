# Pipeline v5 evaluation report

**Status: insufficient evidence.** The question — does removing Embeat
channel 1 improve recommendations on real data — cannot be answered with
the labels available. The challenger is untested, not refuted, on fresh
data. No blind deck was generated and no ranking change is proposed.

## Correction (Nov 22, 2026) — stage 4 labels were wrong; verdict stands

The stage 4 numbers in the original v5 report treated playlist-remains as
"like". Kevin's rule, defined Nov 22, 2026, is three-tier: left in the
playlist = "ok"; added to Liked Songs = "kept" (like); removed from the
playlist = do not like (reject). Only Liked Songs membership counts as a
like. Verified live against his Liked Songs (657 tracks):

- The report's 2 "eligible likes" (Igloo, All Alone With You) are "ok",
  not likes. Eligible fresh likes from v5/v7/v8 = **0**, not 2.
- The 2 true likes among those judgments (Fallout, Sacrifice) are not in
  the frozen catalog and cannot count toward the gate.
- v9 adds 1 eligible fresh like (Pretty Plzzz, which joins the catalog by
  ISRC agreement). **Corrected shortfall: 9 eligible fresh likes** (not 8),
  0 rejects.
- Eligible fresh rejects are **11, not 5**: the 6 joinable v9 rejects
  (We Got That Cool, Like WOAH!, drift, life force, THE FALL BEFORE THE
  GIVE UP, Rocket) join the catalog. The brief for the rerun had expected
  drift and life force to fail the ISRC-agreement join; that expectation
  came from a title-normalization lookup that collided "drift (feat.
  Lames)" with DPR CREAM's different song "DRIFT" and "life force
  (slowed & reverb)" with the different version "life force". The by-ISRC
  join with title + first-artist agreement verifies both join to their own
  recordings.
- "Ok" judgments (7: Virus, Igloo, STEREOTYPE, Not So Bad, All Alone With
  You, Low Life, Heavy Serenade) are omitted from the labeled fresh set —
  neither like nor reject; their pool rows stay unlabeled.
- Diagnostic only: the 5 v6 true likes (Debut, minnesota is a place that
  exists, Jaded, new way out, Stuck In My Head) do not join the frozen
  catalog at all — no Deezer match, no ReccoBeats entry, no pool row
  bearing their titles — so they could not have counted toward the gate
  even if included. They were left out of the rerun.

Stage 4 below is rewritten from the corrected rerun (33 labeled judgments:
3 likes, 30 rejects; 7 ok omitted). The full per-item audit is
`audit-v4-fresh-corrected-labels.md` (local only, not committed). The
insufficient-evidence verdict is unchanged: the challenger remains
untested, not refuted, on fresh data.

Base: `pipeline-v4` at `b72e8b6725c715daaeda74ef695d9d9637220f54`.
One code change in v5 (see below). The v2 ranker remains in service.

## What v4 left open

v4 status was "insufficient evidence for a ranking change": the evaluation
code was ready but the checkout lacked `data/pool_features.json` and
`data/negatives_features.json`, so the runner exited code 2. v4 replayed
the published v3 audit on 306 stored rows with standard NDCG@10: v2
0.471036, v6 per-bucket 0.368658, Embeat full 0.603262, Embeat with
channel 1 off 0.689829 (historical/diagnostic only).

## Stage 1 — frozen inputs recovered

Both frozen caches were recovered intact from the original workspace and
verified by checksum (copies, never moves):

| File | sha256 | Records |
| --- | --- | --- |
| `data/pool_features.json` | `85a502d0…c8a7db` | 590 candidates |
| `data/negatives_features.json` | `bd0625dc…1dec78` | 13 rejects |

The liked-feature snapshot `features_v2.json` was recovered the same way:
its git blob SHA `cb5221065c…` equals the branch blob SHA, so the local
copy is the branch file byte-for-byte. Provenance findings: the liked
export has 649 valid tracks plus one junk `None` row (line 86, also inside
`features_v2.json`); the pool is disjoint from likes by Deezer ID but has
2 ISRC twins with liked recordings ("Him & I", "Trance"); 2 of 13 rejects
lack one feature family; 173/650 likes have no Deezer match. All 590 pool
rows have cached 30s previews. No pool was rebuilt. Full provenance:
local `stage1-provenance.md` (not committed).

## Stage 2 — v4 run before editing (full pool)

23/23 v4 tests pass; all three `check_v4.py` audits pass. All three runs
return `INSUFFICIENT_EVIDENCE` by design (no fresh judgments yet).
Known-like P@10 / standard NDCG@10 on the full pool:

| Run | v2 refit | v6 frozen | Embeat full | Embeat −ch1 | ΔNDCG (−ch1 − full) |
| --- | --- | --- | --- | --- | --- |
| Default (692 rows) | 0.30 / 0.2248 | 0.10 / 0.0784 | 0.30 / 0.3996 | 0.30 / 0.3531 | −0.0464 |
| Fold 1 (696 rows) | 0.20 / 0.1585 | 0.10 / 0.0694 | 0.10 / 0.2201 | 0.50 / 0.5264 | +0.3063 |
| Fold 2 (681 rows) | 0.10 / 0.0851 | 0.20 / 0.2935 | 0.20 / 0.1428 | 0.20 / 0.1642 | +0.0214 |

The minus-ch1 gain does **not** survive: the sign flips with position
window, and on the five fixed distractor draws the challenger beats intact
Embeat on 7/15 draws. Against historical v6 on NDCG, fold 2 is 0/5 — a
clean reversal. No run passes all screening legs even before the
fresh-judgment requirement. Historical results are diagnostic evidence
only.

Two findings, not fixed (ranker behavior, out of scope for the one change):
Embeat arms rank a confirmed historical reject at #2 in folds 1–2 ("Soft
Spine", "prove it") because non-similar channels also score them highly;
and the fold-1 win is driven by a degenerate channel 1 (seed bucket O
triggers 100 unfiltered neighbours vs 21 bucket-filtered in the default
run), so the ablation effect is seed/fold-dependent by construction.

## Stage 3 — the one evidence-backed change

Stage 2 exposed a gap between the documented screening rule ("a reversal
versus v2 or v6 on fixed distractor draws blocks eligibility") and the
implemented gate: `offline_decision` checked P@10 sign flips only, never
NDCG. In fold 2 the challenger loses NDCG to v6 on all five draws while
the P@10 check does not fire.

Fix (one line, `integrate_v4.py:109`): the distractor-draw reversal check
now also fires on NDCG sign flips vs both baselines. New test
`test_distractor_draw_ndcg_reversal_blocks_eligibility` failed before the
fix (`'ELIGIBLE_FOR_BLIND' != 'NO_GO_OFFLINE'`) and passes after. Suite:
23/23 before, 24/24 after. The three diagnostic runs re-run with the
fixed code are byte-identical to the originals (metrics, deltas,
trial_ids, logs) — the fix touches only the decision gate. `check_v4.py`
passes on all three fixed runs.

## Stage 4 — untouched judgments (corrected labels)

40 judgments (v5/v7/v8/v9), relabeled under the three-tier rule. 33 are
labeled (3 likes, 30 rejects); the 7 "ok" judgments are omitted from the
labeled set (see Correction above). The catalog is keyed by Deezer ID
with null Spotify IDs, so Spotify-only judgments cannot join directly;
the join was recovered from frozen records (`reccobeats_pool.json`, keyed
by ISRC with Spotify hrefs, built by the v2 pipeline via ReccoBeats
lookup by ISRC): judgment Spotify ID → ReccoBeats href → ISRC → pool row
with that ISRC, ISRC agreement verified for every match, plus a
title + first-artist agreement guard added in this rerun (a bare ISRC
lookup cannot distinguish a same-title different-artist recording).
v9 drift uses the corrected Spotify ID `69zSQXthMS87H2AvfnMWvd`; the ID
recorded in `recommendations_v2.json` was af1's stale ID
`0pFz5fQBkspNYZGKBgVfdl`, a product of the old first-hit search bug.
Corrected judgments file sha256 `c421a70c…`; enriched judgments file
sha256 `92fabb5e…` (labels and sources unchanged; only Deezer IDs/ISRCs
added where verified).

Matched: **1 eligible fresh like (Pretty Plzzz), 11 eligible fresh
rejects** (reject side meets the ≥2 requirement). Unmatched: 2 true likes
absent from the catalog with no frozen identity recoverable (Fallout,
Sacrifice); 9 v5 drops are already-labeled historical rejects (correctly
excluded); 10 more rejects absent from the catalog. Before enrichment the
runs return `INSUFFICIENT_EVIDENCE` with 0 eligible likes and 0 eligible
rejects (Spotify-only identities cannot join); the enriched runs score.

Enriched runs (all three splits; fold 1/2 pools byte-identical to default
in fresh mode) return `INSUFFICIENT_EVIDENCE` — 1 eligible like < 10.
Descriptive legs on 1 like / 11 rejects: full pool (589 rows) — v2 refit
P@10 0.00 / NDCG 0.00; v6 frozen 0.00 / 0.00; Embeat full 0.00 / 0.00;
Embeat −ch1 0.00 / 0.00. Challenger deltas: vs intact Embeat, vs v2, vs
v6, all 0.0000 on P@10, NDCG@10, and MAP@10 (every arm scores 0.0000 on
the like side: the single like ranks 16th under v2 refit, 305th under v6,
517th under both Embeat arms, score 0.0). The reversal leg passes, but
vacuously — the full-pool deltas are all 0, so no sign flip is detectable.
On the reject side, v2 refit puts 2 rejects in the top 10 (ranks 5, 8)
with mean reject rank 31.0; the challenger puts 1 in the top 10 (We Got
That Cool, rank 6) with mean reject rank 250.6; intact Embeat puts 0 in
the top 10 with mean reject rank 258.3. If the evidence gate were waived
the verdict on this evidence would be `NO_GO_OFFLINE` — but with 1 like,
P@10 ≥ 0.50 is structurally unreachable (max 0.10), so the challenger is
untested, not refuted, on fresh data.

**Still needed: 9 more eligible fresh likes** (10 − 1); 0 more rejects.
They must be fresh keep judgments on recordings in the frozen 590-row
catalog, carrying a Deezer ID or ISRC matching a catalog row. The
v5/v7/v8/v9 set cannot supply them: the 2 unmatched true likes are absent
from the catalog, and the 5 v6 true likes are not in the catalog at all
(see Correction above). A rating round must recommend from the frozen pool
itself — which is exactly what the blind deck does, and the deck is gated
on the offline decision. That sequencing problem is noted, not resolved
here: the parent should decide whether to run a pool-drawn rating round
outside the gate, or treat the offline gate as unpassable with the
current label set.

## Stage 5 — no deck

Not run. The offline gate did not clear, so no listener deck was
generated, no Spotify playlist was created, and nothing was appended to
the recommendation history.

## Rerun commands

From the checkout, with `/home/hatch/workspace/audio-env/bin/python`
(or Python with NumPy, SciPy, scikit-learn). Frozen caches must sit under
`data/` (pool_features.json, negatives_features.json); match/preview
inputs live in the original workspace (read-only):

```bash
python -m unittest -v test_evaluate_v4 test_integrate_v4 test_check_v4
python integrate_v4.py --output /tmp/evaluation-v4.json --liked-export /home/hatch/workspace/spotify-liked-songs.txt --matches /home/hatch/workspace/audio-pipeline/data/deezer_matches.json --previews /home/hatch/workspace/audio-pipeline/previews
python integrate_v4.py --output /tmp/evaluation-v4-fold1.json --holdout-start 130 --cutoff 260 --reject-fold 1 --liked-export /home/hatch/workspace/spotify-liked-songs.txt --matches /home/hatch/workspace/audio-pipeline/data/deezer_matches.json --previews /home/hatch/workspace/audio-pipeline/previews
python integrate_v4.py --output /tmp/evaluation-v4-fold2.json --holdout-start 260 --cutoff 390 --reject-fold 2 --liked-export /home/hatch/workspace/spotify-liked-songs.txt --matches /home/hatch/workspace/audio-pipeline/data/deezer_matches.json --previews /home/hatch/workspace/audio-pipeline/previews
python check_v4.py /tmp/evaluation-v4.json
python integrate_v4.py --output /tmp/evaluation-v4-fresh-enriched.json --fresh /tmp/judgments-v5-v7-v8-enriched.json --liked-export /home/hatch/workspace/spotify-liked-songs.txt --matches /home/hatch/workspace/audio-pipeline/data/deezer_matches.json --previews /home/hatch/workspace/audio-pipeline/previews
```

Per-item audit (local only, not committed — it contains listener-level
judgments): `audit-v4-fresh.md` in the v5 working directory, with the
original 30-judgment table, join evidence, per-item arm ranks/scores for
the 7 matched judgments, and the rerun commands. The corrected-labels
rerun (Nov 22, 2026) has its own audit, `audit-v4-fresh-corrected-labels.md`
(local only, not committed), with the 40-judgment table (33 labeled, 7 ok
omitted), the corrected join evidence (12 matched: 1 like, 11 rejects),
per-item arm ranks/scores for the 12 matched judgments, the v6 diagnostic
(0/5 join), and the corrected rerun commands:

```bash
python build_judgments_corrected.py
python integrate_v4.py --output evaluation-v4-fresh-corrected.json --fresh judgments-corrected.json --liked-export /home/hatch/workspace/spotify-liked-songs.txt --matches /home/hatch/workspace/audio-pipeline/data/deezer_matches.json --previews /home/hatch/workspace/audio-pipeline/previews
python enrich_judgments_corrected.py
python integrate_v4.py --output evaluation-v4-fresh-corrected-enriched.json --fresh judgments-corrected-enriched.json --liked-export /home/hatch/workspace/spotify-liked-songs.txt --matches /home/hatch/workspace/audio-pipeline/data/deezer_matches.json --previews /home/hatch/workspace/audio-pipeline/previews
python check_v4.py evaluation-v4-fresh-corrected-enriched.json
```

(Fold 1/2 use `--holdout-start 130 --cutoff 260 --reject-fold 1` and
`--holdout-start 260 --cutoff 390 --reject-fold 2` respectively, on both
the corrected and enriched judgments files. All six corrected runs return
`INSUFFICIENT_EVIDENCE` and pass `check_v4.py`.)

## Files changed in v5

- `integrate_v4.py` (+1 line): NDCG reversal check in `offline_decision`.
- `test_integrate_v4.py` (+1 test): distractor-draw NDCG reversal blocks
  eligibility.
- `evaluation_report_v5.md` (this file).

Correction PR (Nov 22, 2026, branch `fix/v5-report-corrected-labels`):
report-only change — the Stage 4 section rewritten from the corrected
rerun, plus the Correction banner above. No code changed; the
insufficient-evidence verdict stands.

Not committed (kept local): frozen caches (`data/pool_features.json`,
`data/negatives_features.json`, `features_v2.json`), the per-item audit,
the coordinator ledger and provenance notes, and the enriched judgments
file (listener-level results stay local).
