# PR #9 acceptance record

November 26, 2026. Every test file ran from a fresh clone of `b61140a`. The private inputs and v2 model were restored outside Git. The original eligibility join reproduced at recording level. The historical suites still have six failed assertions or missing-input cases across Bliss and Embeat. An independent, end-to-end replay of v10 remains unavailable.

## Frozen sources

The original Muse handoff contains 50 files. Its checksum list covers the other 49. All 49 hashes matched on restoration and after the runs. The 590-row candidate pool, 13 rejects and Nov 21 liked export also match the full digests recorded in the Nov 21 provenance file. The handoff included the Nov 22 and Nov 24 libraries, 90-entry recommendation history, fitted v2 weights and ten original v10 rows. The separate judged panel checksum list also verified. All nine baseline JSON files tracked in Git still match `463331c` byte for byte.

The 49-file list was created for the handoff. The older manifest contained filenames only. Historical hash evidence is therefore stronger for the three files with Nov 21 digests. The 1,101-entry preview index records audio hashes and lengths. The audio bytes were omitted from the handoff.

## Eligibility behind the nine-like shortfall

The original rule was rerun from the new judged panel, raw ReccoBeats pool crosswalk, frozen catalog, original liked list, historical rejects and prior recommendation ledger. For each offered Spotify ID, require the matching Spotify href, agreement between the crosswalk key and returned ISRC, a catalog row with that ISRC, agreement on title and first artist, and an eligible previously unknown recording with audio features. The panel's corrected ID for the disputed v9 offer agrees with the dated ledger.

| Original v5 evaluation cohort (rounds v5, v7, v8 and v9) | Likes | Rejects |
|---|---:|---:|
| Binary judgments among 40 offers, after setting aside seven neutral ratings | 3 | 30 |
| Already known in the frozen inputs | 0 | 9 |
| No eligible match by the historical catalog join | 2 | 10 |
| Eligible recordings | **1** | **11** |

The resulting twelve catalog row indices and labels match the full-pool rows in **each of the three** archived 589-row evaluations. The ten-like minimum therefore leaves **nine** more likes under the original rule. With the generation-time Deezer IDs in the archived offers and the corrected v9 identity, a separate retrospective join finds **two likes and 21 rejects**, leaving eight more likes under that rule. The folds reuse the same judgments.

The accepted streaming-history comparison retained v2. Across the two later ten-track rounds mean standard NDCG@10 moved from 0.1781 to 0.2153 on one held-out keep. That experiment evaluates order within already-issued lists.

## Historical suites

Each legacy entry point ran in its own local copy with networking disabled. For the v2 recommendation acceptance and Troi suites, the pre-v7 history was reconstructed from the 60 ledger entries in rounds v3 through v6. The 90-row frozen master includes the issued v7 tracks themselves. Both the original master and the dated input files kept their hashes.

| Entry point | Verified outcome |
|---|---|
| v4 and audit unit suites | 83 passed |
| Enhancement suites | 77 passed |
| v2 pipeline, verify dedup, search order, wide window | 37, 15, 8 and 5 passed respectively |
| v3 evaluation and archive integration | 95 assertions and 49 archive checks passed |
| Troi | 20 passed, one existing expected-failure case unexpectedly passed |
| Embeat | 20 passed, T-16 failed its historical performance target. P@10 was 0.60 against a 0.60 best baseline. Mean reject rank was 162.7 against 178.2. |
| Bliss, with the pinned upstream reference audio | 17 passed, five failed. Four tests require the missing original MP3s. B1 ran and failed its numerical comparison. |

The four recording-specific Bliss tests (T3, Z2, S3 and C2) together need **44 unique preview files**, totaling 21,113,226 bytes. Match them against the private preview manifest before rerunning. For B1, the 221,444-byte FLAC from upstream commit `0de922d` matches Git blob `9007c9cce914f86eee6f434c2a4540aeb9281cbb`. With that fixture placed at `tests/fixtures/s16_mono_22_5kHz.flac`, the port met the 0.05 tolerance on 20 of 23 dimensions. Signed errors (port minus reference) on the remaining three were −0.706606 for tempo, +0.412997 for mean rolloff and +0.093132 for rolloff standard deviation. The first two exceed the maximum allowed error of 0.10. [Pinned upstream fixture](https://github.com/Polochon-street/bliss-rs/blob/0de922d/data/s16_mono_22_5kHz.flac) · [Historical test](../test_bliss_features.py)

The raw v4 producer also stopped at its provider-identity guard. Thirteen liked rows carry conflicting ISRCs for Spotify and Deezer. A separate diagnostic withheld the disputed Deezer ISRC assertions, kept the frozen files intact and produced a scored report that passed the source-locked checker. The untouched-source run still requires identity adjudication for an unqualified result. [Verifier](../check_v4.py)

## Original v10 row comparison

The actual generation path loaded the fitted v2 model, scored all 590 catalog rows, selected its 100-candidate MMR window and found 30 previously known candidates. The replay compared the generated ten JSON objects with the original ten, including order, both providers' IDs, the scores, cluster and audio-source fields.

| Verification evidence supplied | Original Spotify IDs recovered | Original positions | Full generated row list equal |
|---|---:|---:|---|
| Only generation-time material actually archived | — | — | Strict replay stopped before selection. The historical Spotify search responses were never saved. |
| Resolve only the ten archived offers, assume misses for competitors and the two uncached ReccoBeats fills | 10/10 | 10/10 | Yes, all fields |
| Also resolve competing candidates through the frozen identity crosswalk | 7/10 | 2/10 | No |
| Use the frozen identity crosswalk alone | 4/10 | 2/10 | No |

Using the derived Spotify answers with the untouched ReccoBeats cache also stops. Two offers needed lookups that are absent from the positive-only cache. Exact generation-time selection would require the responses for searched candidates, including misses, and those two fill outcomes. The archive can establish compatibility with every original output row. It cannot reconstruct the independent selection of those ten tracks. [Strict replay entry point](replay_v10.py)

## Items for merge review

- Run the four audio-dependent checks against the 44 hash-matched previews. Resolve the Bliss B1 discrepancy or carry it as a documented limitation of the historic extractor.
- Record a disposition for Embeat T-16 as a measured negative result and for the 13 disputed provider identities. The historical success thresholds and raw files were preserved.
- If exact v10 selection is required for merge, recover a contemporaneous verification log. If none exists, a reviewer must explicitly accept the documented limit of the retrospective replay.

[Earlier clean-checkout record before the private handoff](https://github.com/kevinkiluu/ai-song-recs/blob/b61140a5cd204a39db217974a388d39f941eaba1/audit/verification.md) · [Original audit findings](README.md)
