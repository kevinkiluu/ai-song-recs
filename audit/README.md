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

The tried attacks include cases that held on main. Planting a known reject at rank 1 triggered the offline veto. Corrupting a scored row's join to the manifest failed verification. The fold-local evaluator withstood transitive recording aliases on a controlled experiment. The enrichment helper returned the added ISRC and preview digest without modifying the original row. The widened verify window reached beyond the first 30 exhausted candidates in an offline replay. None of these controls supplies the missing eligibility join, original offer denominators or the ten generation-time v10 outputs.
