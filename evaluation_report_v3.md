# Evaluation report v3 — integration (2026-11-22)

**Verdict: NO-GO — offline. No candidate passes Legs 1–3 on the
decontaminated 306-item pool, so no new recommendation list is built or
spent. The v7 playlist (v2's list, pending Kevin's blind rating) stands
as the current best.**

Machine-readable numbers: `evaluation_v3.json`. Per-item scores for
independent re-derivation: `v3_scores_306.json`. Integration script:
`integrate_v3.py`. Acceptance checks: `checks_v3_integration.py`
(all pass — see §7).

## 1. Decontamination: the shared pool is 306, not 311

Five held-out likes share a Deezer preview ID with a train like — same
audio, different Spotify ID (Kevin liked the same recording twice,
single vs album version). They are removed from the eval pool for **all**
v3 verdicts.

| Removed held-out like | Spotify ID | Deezer ID |
|---|---|---|
| Warriors — Asketa & Natan Chaim, Requenze, M.I.M.E | `3IpjZ7WnAwMp5HLAAGW94d` | 3794801972 |
| CRAZY — LE SSERAFIM | `4gxEY3Mh5FZZDAJAKPNrCS` | 2967699621 |
| Red Rum — Cardeairè Jack, Witchouse 40k | `0fgXxAsTZj5nv4xsGS6DCX` | 2980031271 |
| [Gen]inside — Catch Your Breath, Kami Kehoe | `1BK0Biq1BJ3HH57zq3Rlpf` | 4015474791 |
| Partyy — Softwilly, Jøjo, LEX PAIN, Yumi | `0IaVAViDtNRVQfbkLnrs5h` | 3952394101 |

Pool: **306 = 94 held-out likes with librosa + 12 negatives with
librosa + 200 distractors** (seed 42, same draw as v2). Random baselines
on this pool, recomputed: **random P@10 = 94/306 = 0.307**, random
NDCG@10 ≈ 0.625 (Monte-Carlo under the v2 convention, see §5), random
mean rank **153.5**. Negative control: n = 12 ranked negatives,
95% half-width ≈ 50 ranks, labelled **UNDERPOWERED** — a 0/12 top-10
count is expected under chance (P ≈ 0.68) and proves little.

Decontamination moved the baselines, not just the candidates. v6-per-bucket
fell P@10 0.50 → **0.40** (one removed twin was in its top 10), so
deltas *vs v6b* on the 306 pool are flattered by a weaker baseline;
read them with that in mind. Recomputed 306 baselines: v2 **0.50 /
0.726**, mean neg rank 287.9; v6-per-bucket **0.40 / 0.654**, mean neg
rank 169.5; v6-global **0.60 / 0.660**, mean neg rank 174.3. These
match Troi's independently computed decontaminated baselines exactly.

## 2. Per-candidate results on the 306 pool

Comparator for every leg is v6-per-bucket. All 4-leg verdicts were
computed by `evaluate_v3.verdict_4leg`, not hand-computed; all point
metrics by `evaluate.rank_metrics` (the Wave-1 convention) on lists
ordered exactly as each Wave-1 team ordered them; MAP@10/MRR@10 from
`evaluate_v3.metrics_from_rel` on the top-10 relevance vector.

| Candidate | P@10 | NDCG@10 | MAP@10 | MRR@10 | Mean rank held-out | Neg in top 10 (H) | Mean neg rank | ΔP@10 vs v6b [95% CI] | ΔNDCG@10 vs v6b [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Troi | 0.80 | 0.820 | 0.608 | 0.500 | 171.1 | 0 (0) | 146.7 | +0.40 [−0.10, 0.80] | +0.166 [−0.214, 0.582] |
| Embeat full | 0.60 | 0.829 | 0.383 | 1.000 | 130.8 | 0 (0) | 158.2 | +0.20 [−0.30, 0.60] | +0.176 [−0.323, 0.534] |
| Embeat minus-ch1 | 0.60 | 0.948 | 0.509 | 1.000 | 131.4 | 0 (0) | 155.1 | +0.20 [−0.20, 0.70] | +0.295 [−0.095, 0.611] |
| bliss fixed | 0.20 | 0.544 | 0.073 | 0.333 | 154.0 | 1 (0) | 185.5 | −0.20 [−0.50, 0.30] | −0.110 [−0.721, 0.428] |
| bliss adaptive | 0.20 | 0.624 | 0.090 | 0.500 | 150.2 | 0 (0) | 178.0 | −0.20 [−0.60, 0.30] | −0.030 [−0.715, 0.495] |
| bliss+v2 blend (w=0.5) | 0.40 | 0.956 | 0.355 | 1.000 | 140.2 | 0 (0) | 267.2 | 0.00 [−0.40, 0.50] | +0.302 [−0.201, 0.618] |
| v2 unchanged | 0.50 | 0.726 | 0.284 | 0.500 | 134.6 | 0 (0) | 287.9 | +0.10 [−0.30, 0.60] | +0.072 [−0.333, 0.525] |

Notes on individual rows:

- **Troi** has the best P@10 (0.80) but the worst mean held-out rank
  (171.1 — worse than v6b's 134.8): it concentrates hits at the top
  and buries the rest. Its bootstrap CI on ΔNDCG ([−0.214, 0.582])
  differs trivially from Troi's own bootstrap ([−0.219, 0.572]);
  the difference is the RNG stream (`numpy default_rng` in
  `evaluate_v3` vs `RandomState` in Troi's own bootstrap), not the
  data. Seed sweep over distractor seeds 42–46, recomputed here:
  ΔP@10 min 0.2 / median 0.3 / max 0.6; ΔNDCG min 0.04 / median 0.166
  / max 0.32 — sign agrees in 5/5 seeds, magnitude does not.
- **Embeat full** on 306 is essentially its 311 result (0.60 / 0.829;
  mean neg rank 162.7 → 158.2 from removing 5 items above some
  negatives). The 20-seed mean (0.39 P@10) was not re-run on the 306
  pool; the medoid seed is the reported number, as in Wave 1.
- **bliss fixed** puts one negative in its top 10: "Martini Blue"
  (non-H) at rank 9. Alone it is below random (P@10 0.20 < 0.307).
- **bliss+v2 blend** has the best NDCG (0.956) and the best negative
  control among the blends (mean neg rank 267.2), but P@10 0.40 —
  below the Leg 2 floor of 0.50 — and ΔP@10 vs v6b is exactly 0.00.
- **v2 unchanged** is the control row: even the incumbent fails Leg 1
  (ΔNDCG CI [−0.333, 0.525] includes 0) and Leg 2 (P@10 CI low 0.2 <
  0.307). That is the honest statement of how weak the offline
  evidence is at this sample size: no ranking on this pool, including
  the shipped one, clears the pre-registered bar.

## 3. Four-leg verdicts per candidate

| Candidate | Leg 1 (beats v6) | Leg 2 (beats chance) | Leg 3 (negatives) | P-10 agreement | Verdict |
|---|---|---|---|---|---|
| Troi | FAIL — ΔNDCG CI includes 0 | PASS — 0.80, CI low 0.4 > 0.307 | FAIL — mean neg rank 146.7 < 153.5 | agree | **NO-GO — offline** |
| Embeat full | FAIL — ΔNDCG CI includes 0 | FAIL — P@10 CI low 0.2 < 0.307 | PASS | agree | **NO-GO — offline** |
| Embeat minus-ch1 | FAIL — ΔNDCG CI [−0.095, 0.611] includes 0 | FAIL — P@10 CI low 0.300 < 0.307 (by 0.007) | PASS | agree | **NO-GO — offline** |
| bliss fixed | FAIL — ΔP@10 −0.20, ΔNDCG CI includes 0 | FAIL — 0.20 < 0.50 | FAIL — 1 negative in top 10 | **DISAGREE** | **NO-GO — unresolved measurement disagreement** |
| bliss adaptive | FAIL — ΔP@10 −0.20 | FAIL — 0.20 < 0.50 | PASS | **DISAGREE** | **NO-GO — unresolved measurement disagreement** |
| bliss+v2 blend | FAIL — ΔNDCG CI includes 0 (ΔP@10 0.00, ΔMAP +0.171 does not rescue the CI) | FAIL — 0.40 < 0.50 | PASS | agree | **NO-GO — offline** |
| v2 unchanged | FAIL — ΔNDCG CI includes 0 | FAIL — P@10 CI low 0.2 < 0.307 | PASS | agree | **NO-GO — offline** |

Leg 4 (live keep rate, n = 42, k ≥ 32) was not run for any candidate:
the spec sequences it after Legs 1–3, and no candidate passes them.
No live list is spent.

The P-10 full-pool check (701 items = 99 held-out + 12 negatives +
all 590 pool tracks, recomputed here for every candidate): bliss fixed
sampled ΔP@10 −0.20 vs full-pool ΔP@10 +0.20 (full-pool P@10 0.30 vs
v6b 0.10); bliss adaptive sampled −0.20 vs full +0.10 (full 0.20 vs
0.10). Opposite signs = `SAMPLED_METRIC_DISAGREEMENT`, which forces
NO-GO on its own, before the legs. Every other candidate agrees in
sign (Troi full ΔP@10 +0.50; embeat full +0.10; embeat minus-ch1
+0.20; blend +0.30 with sampled ΔP@10 0.00, which is a tie, not a
disagreement; v2 +0.20). The disagreement is unresolved: the sampled
pool says bliss hurts, the full pool says it helps, and the spec's
Krichene & Rendle warning is exactly that sampled-metric orderings
need not match full-pool orderings. Do not resolve it by picking the
favourable pool.

## 4. Channel rulings (Embeat)

- **Channel 1 (similar): NOT cut, but the evidence against it
  strengthened.** Minus-channel-1 on the 306 pool: NDCG 0.948 vs
  0.829 (**+0.119**), P@10 unchanged at 0.60, mean neg rank 155.1 vs
  158.2. Identical pattern to the 311 pool. The §(d.2) ablation rule
  is stated on P@10 only, so no channel is cut on that rule — but
  channel 1 hurts within-top-10 ordering by the NDCG measure, and
  minus-ch1 is the closest any candidate comes to Leg 1 (CI low
  −0.095). If Embeat is revised, channel 1 is the first thing to
  re-test. It does not change this verdict: minus-ch1 still fails
  Legs 1–2.
- **Channel 4 (related artist): KEEP (confirmed on the 306 pool).**
  Minus-channel-4 on the 306 pool: P@10 0.5 vs full 0.6 — removal
  costs **0.10 P@10**, NDCG 0.883 vs 0.829. The coordinator's ruling
  stands: measurement beats the 50% contribution heuristic
  (contribution rate 8/20 = 40%). Channel 4 stays.

## 5. Grounding deltas, consolidated

Every number below is recomputed from the landed files in this run;
nothing is carried from the spec's stale 343-item pool.

- Ranked pool is **311 raw / 306 decontaminated**, not 343. Random
  P@10 is **0.318 (311) / 0.307 (306)**, not 0.379.
- **Random NDCG@10 ≠ random P@10.** Under the v2 NDCG convention
  (ideal DCG built from the hits *inside* the top-10 window, per
  `evaluate.py` `ndcg_at_k` and spec T-07), a random ranking scores
  NDCG@10 ≈ **0.628 (311) / 0.625 (306)**. Consequence: v2's NDCG
  0.726 sits only ~0.10 above the chance floor, and the blend's 0.956
  and minus-ch1's 0.948 are the numbers that look strong only because
  the floor is 0.625, not 0.307. Absolute NDCG values are **not**
  standard NDCG and are not comparable to any outside NDCG; relative
  comparisons and the bootstrap CI on the *difference* are unaffected.
- Random mean rank is **156.0 (311) / 153.5 (306)**, not 172.
- The split is **position-based, never time-based**: `added_at`
  coverage is 0/650. `evaluation_v2.json` `counts.test_likes` is 129,
  not 130, because `evaluate.py` drops the `'None'`-id track.
- The 13 negatives rank as 12 (KATSEYE "Touch" has no librosa).
- Troi's bootstrap CI differs trivially from its own JSON (RNG stream,
  §2). Bliss's full-pool numbers recomputed here match its JSON
  (bliss fixed full P@10 0.30 / NDCG 0.484; v6b full 0.10 / 0.387).

## 6. Honest limits and what is unverified

- **Twin leakage in training (disclosed by the bliss team, confirmed
  here).** Decontamination removes the twins from the *test* side.
  The *train* side still contains their audio twins: the v2 model was
  fit on all 520 train likes including the 5 twin rows, the v6
  per-bucket profiles are train statistics including them, Troi's S3
  neighbour index is fit on train including them, and Embeat's index
  is built on train embeddings including them. The bliss seed matrix
  is the one exception — it excludes twin train rows (363 rows, not
  368), which is how its fixed/adaptive numbers reproduce exactly.
  Net effect: every 306-pool number except the bliss rows is still
  slightly optimistic for a model that memorised twin audio.
  Retraining twin-free is out of scope here (v2 files are frozen);
  it is the single cheapest unblocking step for a future round.
- **One user, one ranking, wide CIs.** Every ΔNDCG 95% CI in §2 spans
  0.6–0.9 points. At this sample size the harness can convict a bad
  model (bliss alone) but cannot crown a good one. That is what the
  NO-GO says: not "these models are bad", but "the offline evidence
  cannot separate them from v6".
- **Negative control underpowered.** n = 12, half-width ≈ 50 ranks.
  Leg 3 passes on point conditions only; it cannot prove a model
  rejects bad tracks.
- **Bliss measurement disagreement unresolved** (§3). Also: Z2 (ZCR
  preview stability) fails honestly — ZCR is section-dependent on
  ~25% of tracks; B1 (golden-vector test) is **UNVERIFIED** (fixture
  absent); the bliss weight transfer (VERSION2_WEIGHTS tuned on
  someone else's library) is unproven for Kevin's taste.
- **Blend weight.** The Wave-1 blend's weight is undocumented. I
  recovered it post-hoc: w_v2 = 0.585 with pool-z reproduces the
  Wave-1 311-pool blend numbers exactly (0.60 / 0.907 / 0.458 / 1.0 /
  278.7 — verified by recomputation in `integrate_v3.py`). That
  weight was recovered *from* the eval pool, so adopting it would be
  tuning on the test set; it is **not adopted**. The candidate uses
  the spec's default equal weight (w = 0.5), and no train-only fit
  identifies an interior weight: a pseudo-heldout fit on train alone
  (positions 130–229 as pseudo-test, 289-row twin-excluded seed
  matrix) is flat in P@10 over w ∈ [0.46, 1.0] and rising in
  NDCG/MAP to w = 1.0 (pure v2). The honest statement is that the
  blend weight is unidentified from train data, and the Wave-1 blend
  number (0.60 P@10) is not reproducible from any documented
  procedure.
- **Embeat channel 2 popularity proxy is degraded** (Deezer rank not
  persisted in `pool_features.json`; falls back to v2-score order
  within bucket). Troi S1 is empty (ListenBrainz Labs returns `[]`
  for tested artists; S3+S4 only). Neither was re-verified live in
  this run beyond the Wave-1 findings.
- **Artist-disjoint split (P-11) not run** in this integration; the
  artist-memorisation share is unmeasured. Position-order anchor
  (P-02) still unrun: whether the file is newest-first is unverified.
- **Leg 4 unjudged.** No v3 list exists to judge; Kevin's v7 rating
  is the next live data point, and it judges v2, not v3.

## 7. Test suites, re-run fresh (2026-11-22)

| Suite | Command | Result |
|---|---|---|
| `test_pipeline_v2.py` | `audio-env/bin/python test_pipeline_v2.py` | **37 passed, 0 failed** |
| `test_evaluate_v3.py` | `audio-env/bin/python test_evaluate_v3.py` | **97 passed, 0 failed** |
| `test_troi_patch.py` | `pytest test_troi_patch.py -q` | **20 passed, 1 xpassed** |
| `test_embeat_port.py` | `audio-env/bin/python test_embeat_port.py` | **20 passed, 1 failed (T-16, expected)** |
| `test_bliss_features.py` | `pytest test_bliss_features.py -q` | **20 passed, 1 failed (Z2, expected), 1 skipped** |

No new failures. T-16 (Embeat: eval beats baseline) fails honestly —
the port does not beat v6 on the neg-rank leg (162.7 vs 178.2
required +20) — and was not "fixed" by weakening the test. Z2 fails
honestly as in Wave 1. Note: running `test_embeat_port.py`
re-writes `evaluation_v3_embeat.json` as a side effect; the re-write
is deterministic and its metrics are identical to the Wave-1 file
(medoid 0.6 / 0.829 / 162.7). No Wave-1 file was hand-edited.

Acceptance checks (`checks_v3_integration.py`, written before the
integration ran and watched failing on the missing outputs): **all
pass** — pool counts and removed titles, baselines vs Troi's verified
numbers, every candidate metric re-derived from the per-item audit
file, every verdict re-derived via `verdict_4leg`, and the sequencing
check (no `recommendations_v3.json` exists because nothing passes
Legs 1–3).

Two integration bugs were caught by the checks / cross-checks and
fixed in `integrate_v3.py`, not papered over:

1. **Embeat distractor-key collapse.** Keying Embeat scores by
   Spotify ID collapsed all 200 distractors (Spotify ID `None`) into
   one entry with port 0.0, and the first run reported a phantom
   P@10 = 1.0. Fixed by keying on `embeat_port.cand_key` (ISRC, else
   title/artist), the identity the port itself uses. Corrected
   numbers: full 0.60 / 0.829, minus-ch1 0.60 / 0.948.
2. **v6b tie-order bootstrap drift.** v6b scores are heavily tied
   (rounded to 2 dp). Feeding each candidate's own ranked order into
   the bootstrap changed how v6b ties break and made its point delta
   disagree with the reported point metrics (Troi ΔNDCG read 0.175
   vs 0.820 − 0.654 = 0.166). Fixed by feeding the bootstrap the
   canonical build order for every candidate, matching
   `evaluate.rank_metrics`' tie order.

## 8. What v3 ships

The chosen v3 ranker is **v2 unchanged**. v3 ships the harness
(`evaluate_v3.py`, 97 checks) and the three feature modules as
experimental code — Troi patch, Embeat port, bliss23 features — with
their honest NO-GO verdicts attached. No new ranker, no new
recommendation list, no live list spent. `recommendations_v3.json`
was **not** created, per the eval-spec sequencing.

Recommended next steps (for the coordinator, not taken here): (1)
retrain v2 twin-free and re-run this integration — the cheapest
unblocking step; (2) resolve the bliss sampled/full-pool disagreement
before any bliss-based ranker is reconsidered; (3) if Embeat is
revised, re-test channel 1 removal first (NDCG +0.119, P@10 flat);
(4) collect Kevin's v7 rating — it is the next live evidence, and
Leg 4 needs n = 42 judged tracks before any future GO.
