# Can the five historical rounds be relabeled?

**The surviving Spotify data verify current save status for 23 distinct recordings. None of the five rounds has both an authenticated original offer list and an independently identified post-rating membership snapshot.** A per-round reject total, a five-round hit rate, and confidence bounds on performance across all offers therefore remain unverified. A missing source supplies no observed negatives.

## What Spotify returned

On November 23, 2026, from 02:22 to 02:29 UTC, signed-in, read-only playlist searches and collection reads recovered four same-owner collections bearing the round names. Each returned one complete `READY` page. Separate `get_entity` calls gave explicit library save states for every member. All 23 unique IDs agreed with the Liked Songs traversal below. The entries are **current playlist members**, with dates of addition and rating unavailable.

| Round | Members now | Saved now | Unsaved now | Missing historical evidence |
|---|---:|---:|---:|---|
| v5 | Unknown | — | — | Versioned searches found no matching collection. The training labels give nine IDs, plus a comment naming a survivor. Neither source is a playlist snapshot. |
| v7 | Unknown | — | — | Versioned searches found no matching collection. The archived ranker output contains ten IDs and lacks a playlist ID, round label or creation receipt. |
| v8 | 6 | 1 | 5 | The original offer IDs and post-rating snapshot are missing. |
| v9 A | 3 | 1 | 2 | A and B have identical names and owners. Their roles and historical snapshots are missing. |
| v9 B | 10 | 1 | 9 | All three IDs in A occur in B, including its one saved ID. |
| v10 | 7 | 0 | 7 | The original offer IDs, original order and post-rating snapshot are missing. |

Membership alone has a concrete consequence here. Treating all six v8 survivors as likes would add five positives that Spotify explicitly marks unsaved. If B were the unedited v9 offer and A the rated copy, the observed sets would yield **1 like, 2 OK and 7 rejects**. The identifiers and names establish the subset relationship. They supply no creation event or version history establishing those roles. For v10, all seven visible tracks are unsaved. The original count and what happened to the other offers cannot be inferred from those seven.

## Liked Songs and historical pointers

The signed-in Liked Songs collection advertised 659 items. Following cursors to exhaustion twice returned the **same 658 valid, unique Spotify track IDs**, once in 11 pages of up to 60 and once in 14 pages of up to 50. One advertised position was unreturned in both traversals. Neither traversal reported a filtering reason. Direct per-track library status establishes the status of recovered playlist members despite that discrepancy.

The frozen `features_v2.json` snapshot has 650 rows. Of those, 649 have distinct valid Spotify IDs and all 649 occur in the present library. One row has the literal string `None` as its ID, title and artist. Nine returned current IDs fall outside the 649. Both saved tracks on the recovered v8 and v9 collections are among those nine. These observations establish current membership and absence from the valid frozen IDs. They leave the timing and cause of the saves open.

The nine v5 hard-coded training negatives in [negatives_features.py](../negatives_features.py) resolved by exact Spotify ID in a separate catalog read. All nine are unsaved now. The survivor named in the source comment resolves to a saved track by title and artist. The ten scored recommendations in [recommendations_v2.json](../recommendations_v2.json) likewise resolved by exact ID and all ten are unsaved now. A listener can leave an unsaved track in a playlist as OK. The training inputs and ranker output therefore cannot establish which tracks were removed from v5 or v7.

## Identity checks and recovery

The checkout, its branch histories and all eight remote PR heads contain no original playlist-creation payload or versioned offer/member export for these rounds. The v4 Git bundle, scratch files and shared downloads also yielded none. The apparent scratch inputs are a test fixture with short synthetic identifiers, including `l0`, `r0` and `c0`. The repository provided no release assets or Actions artifacts. The source inventory used the raw files and Git objects. Prose totals did not participate in the joins.

The following SHA-256 digests identify the observed sets without publishing listener rows. To reproduce a digest, extract canonical 22-character Spotify track IDs, discard duplicates, sort ASCII bytewise, and hash each ID followed by one LF byte. These digests verify membership, without making a claim about historical order.

| Collection | SHA-256 |
|---|---|
| Liked Songs, returned set | `7ef540a7ae5028473b63b53f3804b94fc809161193a86e24798b0e7d2394a1ae` |
| v8 | `69965693480dd1b344c7896b4aa7989958bb1cb7be3df6b35e944ea03ba6db45` |
| v9 A | `8a4b1506547b57a601df9f68df90c32e8961ca02bbc614416e5c515bb5d46c9a` |
| v9 B | `f4f82991583f3ffa48e78c431339c9664f808dd0d0250d8b8218ec0308871cd6` |
| v10 | `9df4f6587aeb1730dbd54ccc700de1034a118694fc0843fa57b80f44d8b2c82b` |

Historical reconstruction requires the exact offered IDs tied to each playlist and round, the retained IDs after rating, and Liked Songs membership at the intended cutoff. With those sources, count offered and liked as like, offered and retained but unliked as OK, and offered, removed and unliked as reject. Resolve any recording variants explicitly and preserve unknowns for unavailable IDs. Frozen-catalog eligibility additionally requires the real recording crosswalk and the prespecified historical/training exclusions.
