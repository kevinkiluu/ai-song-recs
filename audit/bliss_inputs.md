# Bliss pool sensitivity

The historical 306-row archive reproduces P@10 of .40 and standard NDCG@10 of .538886 for the equal-weight Bliss/v2 blend. The v3 report's opposite-sign precision deltas concerned fixed and adaptive Bliss. The archived blend tied v6 per-bucket on precision. Testing the historical reversal on the original full pool still requires the fitted v2 weights, Bliss23 vectors, reject features and frozen catalog.

## Replay the saved scores

From the checkout, choose a new output filename.

```bash
python -m audit.bliss_sensitivity --archive v3_scores_306.json --output bliss-archive-replay.json
```

The archive's SHA-256 is `65b4ee2c90fe9b451c7d714f17e35d9b4a091a7c8d00fd318d76413cbff16092`. Its historical row-order replay includes both copies of “Control.” The sensitivity panel consolidates those two identical score/label records under Deezer ID 4015474741. It keeps the published score columns, including the blend normalized on the original 306 rows. Every sensitivity ranking breaks ties by ascending recording ID. This changes v6's NDCG from .368658 under the original tie order to .363624 on the 305 recordings. The blend's top ten stays the same.

Five seeded draws each keep 93 likes and 12 rejects, and select 100 of the 200 archived distractors without replacement in their frozen source order. The smaller draw size lets catalog membership vary within the available sample. The rows below give **candidate minus v6 per-bucket**, using precision and standard NDCG at ten.

| Panel | Fixed ΔP | Fixed ΔNDCG | Adaptive ΔP | Adaptive ΔNDCG | Blend ΔP | Blend ΔNDCG |
|---|---:|---:|---:|---:|---:|---:|
| All 305 archived recordings | −.20 | −.168435 | −.20 | −.139618 | .00 | +.175262 |
| 42 | −.20 | −.045630 | −.10 | −.066120 | +.10 | +.258208 |
| 43 | −.20 | −.146427 | .00 | −.013375 | .00 | +.140917 |
| 44 | −.20 | −.273606 | −.10 | −.131933 | +.10 | +.120152 |
| 45 | −.20 | −.194255 | .00 | −.058569 | +.20 | +.271569 |
| 46 | −.10 | −.219612 | .00 | −.008666 | +.20 | +.230019 |

No draw gives an opposite nonzero sign relative to the archive reference on either metric. Zero deltas are ties. The original historical full-pool comparison remains unverified.

The JSON includes the frozen score panel, input hashes, every draw's member IDs, top-ten IDs in rank order, each arm's entrants and displaced IDs against v6, and each draw's top-ten changes against the reference for the same method. The score-panel SHA-256 is `91b91306daf568333a0791b68e1c9b31fc03fefc0e5263108bae5b5d6dfe165e`. It covers the UTF-8 encoding of the panel array serialized as compact JSON with sorted object keys and unescaped Unicode. Array order preserves the sampling frame.

## Score the full supplied pool

```bash
python -m audit.bliss_sensitivity --raw-root /path/to/frozen-snapshot --output bliss-full-replay.json
```

Supply the original `features_v2.json`, `model_v2.json`, `model_v2.npz`, `data/bliss23.json`, `data/negatives_features.json` and `data/pool_features.json`. The CLI records the six input hashes and checks available pinned digests for the liked features and model configuration. Attributing results to the original full pool also requires independent verification of the other four files. The archived catalog was reported to contain 590 rows. A run on another snapshot describes that supplied snapshot.

The full comparison keeps one eligible representative per recording component identified by available provider IDs, ISRC or preview digest. The identity join also uses an embedded ReccoBeats ISRC when its Spotify URL matches the recording. Shared coverage requires finite BPM, RMS and brightness and a 23-value preview Bliss vector. Held-out likes occupy positions 0–129. Components also present among older training likes are excluded from the panel. Rejects are retained for this historical sensitivity. The original fitted v2 model used the historical judged rejects. The JSON lists ranked recording IDs that match the 13 configured fit rejects. All 12 ranked archive rejects match. These metrics describe the fitted historical model.

Bliss averages the ten nearest distances to distinct train-only liked recordings using the original fixed or inverse-variance weights. The blend uses the original fitted v2 predictions rounded to three decimals. It combines equal weights on z-scored v2 and negative fixed Bliss distance, normalized once across the full eligible panel. The comparator uses the original frozen v6 scoring function and v3 artist-bucket lookup. Seeds 42–46 each retain all judged recordings and draw up to 200 eligible catalog distractors. Every draw slices the full panel's score columns. The primary result is the full eligible pool. The JSON counts distinct memberships and identifies draws that differ from the reference. When every draw retains the entire pool, rankings are descriptive and the reversal result is null. The eligibility counts distinguish an empty catalog from a supplied catalog with zero eligible distractors. A missing catalog file blocks raw scoring.

[Historical Bliss implementation](../integrate_v3.py) · [Historical report](../evaluation_report_v3.md) · [Saved scores](../v3_scores_306.json)
