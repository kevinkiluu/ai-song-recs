# v2 ranker, recording identity and v10 replay

Pinned source `463331ccaed9ab5f3b8d1c9703bff956b155f566`. Line references below apply to that tree unless a different commit is named. The deployed score remains `0.5*z(−nearest-centroid distance) + 0.5*z(LR log-odds)`. Both score windows retain cosine MMR with `lambda=0.7`. Fixes affect identity, ties and empty results. The production model was neither refitted nor replaced.

## Reproduced defects

| Priority | Pinned location | Input and observed result | Expected and fixed result |
|---|---|---|---|
| P1 | `recommend_v2.py:246`, `:285` | Query `drift (feat. Lames)` with only an `af1` hit. Put the `af1` recording key in history but give the hit a different ID. The old verifier emitted `af1`’s ID under `drift`’s title because it checked the candidate’s key after substituting an unrelated result. Empty title/artist pairs could also agree, and `clean` discarded bracketed versions. | Require a nonempty title/version and first-artist match plus a valid track URI. Keep an exact `drift` hit even when listed second. Drop unrelated, unverified or wrong-version hits. |
| P1 | `recommend_v2.py:73–88`, `:280–290` | A two-column known track vanished from the exclusion index. Padded URIs missed exact ID matches. A candidate with a known source ID could pass after re-resolution. | Index the available ID and recording key independently, normalize URIs, and check both source and resolved IDs. The original recording key is `clean(title), clean(first_artist)`. Alternate Spotify IDs for the same normalized `af1 (sped up)` or `A Milli (Album Version Explicit)` are excluded. |
| P1 | `recommend_v2.py:275–291`, `:349–353` | Four rows resolved to `(1,sid1), (2,sid1), (3,sid2), (4,sid3)` where row 3 repeated the first recording key. All four passed. The final ID-to-score index would also associate both `sid1` entries with the last row’s score. | Only `(1,sid1), (4,sid3)` pass. Every accepted ID and exclusion key joins the seen set before processing the next row. |
| P1 | `recommend_v2.py:195–204`, `:301`, `:335–338` | Three identical feature vectors and equal scores followed input order `[3,1,2]`. Reversing 105 ties changed the selected ten from IDs 101–110 to 205–196. | Break exact ties with the same stable identity order at the top-100 and top-20 score cutoffs and at each MMR selection. Real pool IDs are unique. The primary tiebreak is lexicographic Deezer ID. Both orders now produce 101–110 at score `0.500`. |
| P1 | `recommend_v2.py:303`, `:316–321`, `:365` | A successful exhausted run returned `[]` but left the last published list in `recommendations_v2.json`. Filtering out every preview instead raised `ValueError` on `zip(*)`. The prior result could be read as the new offer. | A completed empty run returns and writes `[]` for both routes. Successful output uses the existing JSON writer. A nonzero Spotify CLI status, malformed JSON or explicit error payload aborts before replacing the last successful output. The provider-failure regression failed for each input against the first empty-output patch. A timed-out subprocess also propagates its exception. |

The old search-order test explicitly expected an unrelated first hit when no title matched (`test_search_order.py:85–91`). Its expectation now matches the identity contract. A separate negative control still picks the right primary artist among identical titles. The verifier checks the supplied liked/history snapshot and identity rule. Unrecorded prior offers, arbitrary retitles lacking a shared ID, unreliable upstream names, and distinct artists with identical display names remain outside what those inputs can establish. Different IDs for `af1` and `af1 (sped up)` remain distinct under the established key, as do `A Milli` and `A Milli (Album Version Explicit)`. Regression tests cover both directions and a shared version title with an added feature credit. The original `clean` strips parenthesized feature credits and square-bracket text while retaining other parenthesized words. Identifying equivalence across releases with genuinely different IDs and normalized titles would require additional evidence.

## Historical behavior exercised directly

I executed `recommend()` from the following git trees with a file-backed v2 scoring fixture and a captured-response-shaped Spotify subprocess stub. Each run used a temporary output directory. These are constructed inputs reproducing the failure mechanisms. The original search payloads and historical pool are unavailable here.

| Code | Prior key, different Spotify ID | `drift` hit preceded by `af1` | Only `af1` present | 40 ordered scores, first 30 in history by key |
|---|---|---|---|---|
| `8f8086f^` | Re-emitted both historic titles | Wrong ID | Wrong ID | Re-emitted known rows 0–9 |
| `8f8086f` | Blocked both, then crashed when the fixture had no fresh rows | Wrong ID | Wrong ID | Crashed at empty rescore |
| `77255c9` | Blocked both, then crashed on empty rescore | Correct ID | Wrong ID | Crashed at empty rescore |
| `463331c` | Returned `[]` | Correct ID | Wrong ID | Rows 30–39, in order |
| Audited code | Returned and persisted `[]` | Correct ID | Dropped candidate | Same rows 30–39, same order |

The pre-fix code is inspectable at `8f8086f^:recommend_v2.py:228` (first URI) and `:257` (ID-only gate). `8f8086f:recommend_v2.py:255` introduced the key check. `77255c9:recommend_v2.py:241` retained the unsafe fallback; `:304` used 30 slots and `:320` unpacked an empty result. `bec5626` widened the window and added the verified-empty return. A fixed-30 mutation of the forty-row test must drop all candidates; the pinned and audited top-100 paths both select the ten clean rows. The preexisting fixed-window policy may produce fewer than ten when clean rows exist only below rank 100.

Three fixture checks can be repeated on earlier revisions from this checkout (also run them on the audited tree). A failed assertion or empty-rescore exception on old code is expected.

```sh
case_dir=$(mktemp -d)
git archive 77255c9 scoring.py recommend_v2.py pipeline.py features_v2.py | tar -x -C "$case_dir"
cp test_ranker_audit.py "$case_dir/"
(cd "$case_dir" && python -m unittest \
  test_ranker_audit.GenerationTests.test_historical_release_keys_are_dropped_even_with_other_spotify_ids \
  test_ranker_audit.GenerationTests.test_historical_drift_search_exact_or_absent \
  test_ranker_audit.GenerationTests.test_thirty_blocked_top_scores_leave_ten_clean_picks)
```

## v10 generation and original-output comparison

**The original v10 ordered output is unverified.** Running `python audit/replay_v10.py .` returns exit 2 and names six absent inputs: `model_v2.npz`, `data/pool_features.json`, the two dated liked/history TSVs, archived search responses and `recommendations_v10.json`. The ReccoBeats ID-fill cache is also required if the original rescore needed lookups. A tree/history inventory found the earlier `recommendations_v2.json` was introduced at `b7f3ef0`; its bytes cannot establish a v10 output. Fresh searches, an after-the-fact history or a newly trained model could produce a different list for reasons independent of the fixes.

Independent comparisons using the committed rows:

- The 10 earlier recommendations have 10 distinct Spotify IDs, zero Spotify ID overlap with `features_v2.json`’s 650 listed tracks. Three share a Deezer ID with the 200 sampled catalog rows in `v3_scores_306.json`. Their v2 scores agree across both files to all three published decimals (1.241, 1.283, 1.116). These are stored scores, not a regeneration from the missing weights.
- The private read-only v10 playlist observation contains seven distinct current recording keys. Two match catalog rows by cleaned title + first artist, neither by Spotify ID because all 200 catalog rows omit Spotify IDs. Their stored v2 scores are 0.596 and 0.603. No current survivor matches an older recommendation or the liked snapshot by key or Spotify ID. Current playlist membership cannot reconstruct the three absent offers, original offer order, or verification inputs. Individual current judgments are excluded from this report.
- Of the 200 sampled catalog scores, six rounded values are shared by more than one row. The raw-score boundaries 100/101 and 20/21 in that **sample of published rounded scores** have distinct values. This does not measure tie prevalence in the full original pool or final candidate set.
- **Actual recording-level MMR comparison.** Joined all 94 archived held-out score rows to `features_v2.json` on both Spotify and Deezer ID. Rebuilt median imputation and standardization from the 520 training-position likes (78 columns), then ran pinned and audited MMR on the published three-decimal v2 scores. The top-20-to-ten step returned the same ten ordered Spotify/Deezer pairs. For all 94 items, 92 stayed in the same position. The only swap was a pair of Spotify aliases with the same Deezer ID, recording key and score; the ordered Deezer sequence was identical. The SHA-256 of the fixed ten `spotify_id|deezer_id` lines (UTF-8, LF including final LF) is `f993541477a159cab35641aeff546aaef893b0064f4842b471a616dea0f0c486`. This exercises MMR on genuine features, outside the generation-time catalog and without fitting or replacing the deployed ranker.
- A controlled run of the real `recommend()` used a one-coefficient fixture NPZ, new-response fixtures and 40 synthetic candidates. Both pinned and fixed code returned IDs 30–39 in order after the first 30 became known. Another test supplies the replay command with ten hand-calculated complete output rows, then reverses them, then removes a required search. The command distinguishes identical output, same membership with different order, and missing evidence. None of these fixtures estimates v10’s actual history or taste scores.

**Private replay procedure.** Supply one directory containing the original snapshot. Keep personal exports and search data outside git. The two TSVs use `title<TAB>artist<TAB>spotify:track:ID`. `spotify_search_responses.json` is a list of `{ "title": "…", "artist": "…", "response": <archived CLI search JSON> }` keyed by exact query, once per query. Provide the production `features_v2.json`, `model_v2.json`, `model_v2.npz`, `data/pool_features.json`, and complete frozen output rows in `recommendations_v10.json`. If a finalist needs ReccoBeats, `data/reccobeats.json` must map every requested ID to its recorded result; use JSON `null` only for a confirmed lookup with no features. The command refuses missing lookup coverage. It runs the production model/MMR/verifier in a temporary output directory, never calls either service, and emits aggregate comparison counts.

```sh
python audit/replay_v10.py /path/to/private/v10-snapshot
python -m unittest -v test_ranker_audit
python test_search_order.py
```

Exit 0 means complete JSON output equality, including rank, ID, scores, cluster and metadata. Exit 1 means a measured difference; the result gives common IDs and matching positions. Exit 2 identifies missing or invalid replay evidence. The historical archive remains read-only.

The new ranker suite ran 18 tests. The original search-order script passed eight assertions. The original `test_verify_dedup.py` passed nine fixture assertions and failed its one private-data check on the absent liked export. `test_wide_window.py` requires the absent Hatch output/model/pool; it now writes each run to a temporary directory and restores test functions even when an input is missing. Pinned line 75 installed a fake verifier before an unguarded read of the absent output at line 76. Importing that test left the fake in place, causing six ranker assertions and one error in a discovery run. The same ranker suite passed after isolation. The controlled window test covers the production generation call without those private files.
