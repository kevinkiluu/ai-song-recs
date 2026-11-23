# Recommendation pipeline audit

**v2 remains the production ranker. The fresh Embeat channel-1 ablation verdict is INSUFFICIENT EVIDENCE.** The 306 historical score rows support a narrower result. Full Embeat and Embeat without channel 1 each returned six held-out likes in ten slots. Removing the channel moved the likes earlier, increasing standard NDCG@10 from 0.603262 to 0.689829. Those observations do not measure what a listener will keep on a fresh list.

The review starts at `463331ccaed9ab5f3b8d1c9703bff956b155f566`. It uses the eight PR heads, executable code, checked-in item rows and read-only Spotify observations. The numerical comparison consults the old prose after computing the results. The fresh catalog/judgment join and the original v10 generation inputs remain unavailable. [Independent raw calculations and comparison](recomputation.md) · [Actual playlist reads and three-tier evidence](ratings.md)

## Findings, fixes and negative results

Every finding table cites the **original** source locations at the audited SHA. Each reproduction uses a named test or a script. The historical artifacts retain their original metric values. The patched inferential evaluation uses standard NDCG with held-out positives in the ideal ranking.

| Review order | Evidence |
|---|---|
| 1. Evaluation definitions, bootstrap, leakage and live stopping | [Severity, reproduction and expected/actual outcomes](metrics.md) |
| 1. Independent verifier and the four requested fault injections | [Mutation matrix and original/fixed detections](mutations.md) |
| 2. v2 ranking, Spotify verification, historical bug replay and the v10 path | [Ranker findings and replay outcomes](ranker.md) |
| 3. Frozen input handling, provider failures and the genre mapping | [Data plumbing findings](plumbing.md) |
| Reproducibility and acceptance | [Fresh checkout, per-suite counts and checksums](verification.md) |
| Next work in priority order | [Six evidence gaps, falsifiable claims and measurements](plan.md) |

## What held under attack

- A known reject at rank 1 triggered the offline veto. A broken scored-row join also failed the old verifier. The separate forged-score and source-label cases exposed verification defects.
- The fold-local evaluator excluded transitive recording aliases on a controlled split. Changing a discarded training twin’s features left the eligible scores unchanged.
- The enrichment helper returned the ISRC and preview digest on its copied row, preserved the input row, and withheld an ISRC belonging to a different provider recording.
- The widened verify window reached beyond 30 exhausted candidates. On the 94 archived held-out feature rows, pinned and patched MMR returned the same ordered top ten. This experiment holds the saved scores fixed.
- The independent paths reproduced all nine historical P@10/legacy-NDCG pairs on the 306 rows, and all four archived standard-NDCG comparisons. The historical per-bucket/global v6 comparison could also be reconstructed for the original 311-row panel.

The full-pool eligibility join, original offer denominators and exact v10 replay require the private source bundle. [Evidence required for acceptance](verification.md)
