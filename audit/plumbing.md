# Data plumbing findings

Locations refer to main at `463331c`. Each reproduction below is a method of `test_data_plumbing_audit.DataPlumbingTests`. Run one by appending the method name to `python -m unittest test_data_plumbing_audit.DataPlumbingTests.`. The fixtures use temporary files and recorded response shapes. They make no network requests.

| Severity | Original file and line | Reproduction | Expected | Observed on main |
|---|---|---|---|---|
| P0 | `checks_v3_integration.py:13` | `test_integration_verifies_this_checkout` | Verify the artifacts beside the script under review | Opened the artifact in a different workspace. A passing check could certify a different checkout. |
| P0 | `recommend_v2.py:31`, `features_v2.py:35` | `test_replay_imports_code_from_reviewed_checkout` | Replay the scoring code in the reviewed checkout | Both modules prepended the legacy data directory to the import path. A separate scoring module there executed during replay. |
| P1 | `features_v2.py:91` | `test_stale_genre_mapping_cannot_replace_snapshot` | Reject a mapping with zero valid labels for these 650 likes before feature assembly | 635 unrelated IDs passed the size check and the last complete snapshot was overwritten. |
| P1 | `fetch_genre_buckets.py:61` | `test_incomplete_pagination_preserves_genre_mapping` | Leave the last completed mapping in place if page two fails | Swallowed the read failure and published the first page as complete. |
| P1 | `fetch_genre_buckets.py:59` | `test_duplicate_page_does_not_truncate_later_genre_members` | Follow the continuation cursor through a duplicate-only page | Dropped the new member on the final page. |
| P1 | `pipeline.py:27`, `features_v2.py:132` | `test_transient_deezer_failure_is_retried_before_caching` | Retry a timeout and cache the successful provider ID | The lower-level search converted the timeout into an empty result. Feature extraction permanently marked the recording unmatched. |
| P1 | `features_v2.py:160` and `:270` | `test_failed_preview_can_be_recovered_from_existing_cache` | Retry the same provider recording and analyze the repaired preview | A cached match plus a cached analysis error prevented both steps on later runs. |
| P1 | `features_v2.py:144` | `test_interrupted_preview_never_publishes_empty_audio` | Failed reads leave no published audio | Timeout created an empty mp3 which later code treated as downloaded. |

The first five checks failed individually in the initial red run. The next two failed against the existing download and analysis code before those fixes. The checkout-provenance test then failed with an exception raised by the unrelated scoring module. All eight pass on the patch. These reproduce mechanisms that affect coverage. Attribution of specific historical missing previews requires the private match and download logs.

## Historical checks

The PR 1 and 2 heads are the same Git object (`b7f3ef0`). The checked-in implementation already contains the persistent genre mapping and a minimum-count guard. No distinct pre-wipe feature snapshot is present in either diff. In a disposable fixture the pre-fix guard still rejected a missing mapping. The adversarial cases above exercise stale or incomplete mappings that survived that historical repair.

PR 3 introduces the fixed v3 score export and its acceptance check. PR 4 includes `integrate_v4._enrich_records`. At `integrate_v4.py:323–345` on main, a two-field input plus a matching Deezer record and temporary preview produced a *returned* row containing the ISRC and SHA-256, left the original dict unchanged, and refused an ISRC from a different Deezer ID. The older dict-copy failure could not be reproduced at this SHA. The committed helper already appends and returns each enriched copy. A historical failure of an unpublished helper cannot be dated from these commits.

PR 5 changes the sampled-pool gate in the runner. PR 6 changes ID/key verification and search ordering. PR 7 changes the v5 report only and adds no underlying rating event or playlist snapshot. PR 8 changes the verify window and empty result handling. The corresponding behavioral checks and remaining holes are in the other audit sections.
