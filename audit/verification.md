# Acceptance status

**The full-suite and historical replay acceptance conditions remain open.** A fresh clone of implementation commit `d2330a2ec65d52bc1e3c164a7f946d467ab771ea` ran every `test_*.py`, the standalone v3 integration checker and the v4 producer/checker. It began and ended with empty Git status. The legacy scripts have module-level test execution and some call `sys.exit`, so they ran in separate processes. A private Linux mount mapped the fresh clone onto the scripts’ historical `~/workspace/audio-pipeline` location. Network was disabled for the historical suites. Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0 were used.

| Entry point | Fresh-checkout result | What it establishes or what failed |
|---|---|---|
| `python -m unittest test_evaluate_v4 test_check_v4 test_integrate_v4 test_audit_metrics test_audit_mutations test_ranker_audit test_data_plumbing_audit` | **83 passed** | 24 existing v4 tests plus 59 audit tests and controls. See the finding tables for the before/fix evidence. |
| `python checks_v3_integration.py` | **49 passed** | Historical archive compatibility, including recomputation from the 306 per-item rows. |
| `python test_search_order.py` | **8 passed** | Search order, title and artist matching, and refusal of an unrelated hit. |
| `python test_pipeline_v2.py` | 26 passed, 4 failed | Missing fitted NPZ, negative features, GTZAN result, and liked export required for recommendation checks. The committed ten-row recommendations file itself loaded. |
| `python test_verify_dedup.py` | 9 passed, 1 failed | Fixture checks ran. The real-data block stops at the absent liked export; it also requires the pool and history. |
| `python test_wide_window.py` | 0 passed, 3 failed | Each case calls the original generation path and stops at the absent liked export. These grouped failures occur before the five historical assertions. Controlled window tests exercised the same production function separately. |
| `python test_evaluate_v3.py` | 82 passed, 1 failed | T-01, the real-data block, stopped at the missing negative feature file. The synthetic checks ran with their corrected metric definitions. |
| `python test_embeat_port.py` | 3 passed, 15 failed | Missing fitted model, pool and private inputs. Downstream checks also reported unset variables after setup failed. |
| `python -m pytest -q test_troi_patch.py` | 10 passed, 10 failed, 1 unexpected pass | Missing private liked/history exports, negatives, model or pool. The test marked xfail is reported separately by pytest. |
| `python -m pytest -q test_bliss_features.py` | 13 passed, 8 failed, 1 skipped | Recording-specific checks need the original previews and `data/bliss23.json`. The suite skipped its optional upstream FLAC fixture. |

The counts are the results the scripts actually emitted. Some historical checks sit inside a shared block and never execute after the first missing input.

## CLI and v10

The full synthetic fresh experiment went through `integrate_v4.py`, strict `check_v4.py --source-lock`, and the independent `audit/recompute.py --trial` path successfully. It had ten synthetic likes and two rejects. With the production paths and missing private files, the producer returned exit 2 and wrote `INSUFFICIENT_EVIDENCE`. The checker identified that output as an unscored record.

The v10 generation entry point ran to completion on a controlled snapshot on both pinned and fixed code. It selected the same ten ordered fixture IDs beyond the exhausted window. Holding the available 94 historical scores fixed also reproduced the same MMR top ten. The actual private replay command `python audit/replay_v10.py .` returned exit 2 and named the absent fitted weights, exact candidate pool, liked and history exports, original Spotify search responses and original ten v10 rows. The replay also requests cached ReccoBeats responses when the original path needs a fill. Seven visible playlist survivors provide a current membership check. They cannot establish the original order or any extra offers. [Replay evidence and snapshot layout](ranker.md)

## Frozen inputs

Before edits, SHA-256 was calculated for **all nine JSON files tracked at `463331c`**. All nine matched after the fresh-checkout tests, including `features_v2.json`, `recommendations_v2.json` and `v3_scores_306.json`. No tracked JSON, fitted artifact or historical report changed in the PR. To repeat the byte comparison on a clone:

```bash
git ls-tree -r --name-only 463331c | rg '\.json$' | while IFS= read -r path; do
  git show "463331c:$path" | cmp - "$path" || exit 1
done
```

The original Nov 21 liked export (650 rows), Nov 22 dump (657), 90-entry history, 590-row pool, 13-row reject feature file and their independent checksum manifest were absent from the checkout, all available PR heads, local archives, releases and Actions artifacts. Only the available bytes can be certified. The current Spotify library and committed derived features are separate sources. No private history, raw listening export or audio has been committed. Spotify access was read-only.

To close the remaining checks, recover the original hashed files, generation-time model/caches and original v10 offers, plus the original offers, post-rating playlist snapshots and cutoff liked membership for v5, v7, v8, v9 and v10. The scored v5 per-record manifest enables a separate arithmetic replay. The raw catalog/judgment sources and identity crosswalk enable independent confirmation of the eligible count behind the nine-like shortfall. [Recomputation and missing inputs](recomputation.md) · [Playlist inventory](ratings.md)
