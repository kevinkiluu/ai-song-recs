# Independent numerical audit

The historical 306-row file reproduces the published precision and NDCG. Removing Embeat channel 1 moved likes earlier in this ranking. Each arm returned six known likes in its top ten. The corrected v5 claim of nine more eligible likes depends on a frozen catalog join whose inputs are absent here. The three v5 fold tables and the fresh 589-row table therefore remain unverified.

Run `python audit/recompute.py` from the repository root. The script uses Python’s standard library, reads per-item scores and features, and prints the input hashes, hit ranks, and full-precision results. It imports no ranking or evaluation module and reads no reports. Source commit `463331ccaed9ab5f3b8d1c9703bff956b155f566`. [Independent calculation](recompute.py)

## Ranked retrieval

There are 94 held-out likes, 12 configured historical rejects and 200 unrated distractors. I sorted all 306 rows for each method, retained the archived order for equal scores, and treated only held-out likes as positive. A v6 per-bucket tie at ranks 6–7 places the like first under that convention. The two Embeat columns have unique scores throughout their top tens and at the rank-10 boundary. The omitted Embeat cosine tie-break prevents exact replay of some ranks farther down the list. [Input rows](../v3_scores_306.json) · [Historical sorting procedure](../integrate_v3.py)

| Method | Like positions in top ten | P@10 | Standard NDCG@10 | Legacy NDCG@10 |
|---|---|---:|---:|---:|
| v2 | 2, 3, 5, 7, 10 | .50 | .471036 | .725864 |
| v6 per-bucket | 2, 5, 6, 9 | .40 | .368658 | .653894 |
| v6 global | 3, 5, 6, 7, 8, 10 | .60 | .480003 | .659953 |
| Troi | 2, 3, 4, 5, 6, 7, 8, 10 | .80 | .713654 | .820174 |
| Embeat full | 1, 4, 5, 7, 9, 10 | .60 | .603262 | .829420 |
| Embeat without channel 1 | 1, 2, 3, 5, 8, 9 | .60 | .689829 | .948440 |
| Bliss fixed | 3, 5 | .20 | .195189 | .543771 |
| Bliss adaptive | 2, 5 | .20 | .224006 | .624051 |
| Bliss + v2 | 1, 2, 4, 5 | .40 | .538886 | .955830 |

Both NDCG columns discount a hit at position *i* by 1/log₂(*i*+1). Standard NDCG divides by the ideal gain at min(10, positives in the pool). The legacy score divides by the ideal gain at the number of hits returned in that specific top ten. All nine P@10 and legacy NDCG pairs match v3 to its displayed precision. All four standard NDCGs in v4’s replay and the v5 recap match to six decimals. The standard gain from the Embeat ablation is .086566. Ranking order can reverse when the number of hits differs. Bliss + v2 beats Troi under the legacy metric and trails it under standard NDCG. That contradicts the v3 report’s claim that relative comparisons are unaffected. Bliss fixed returns “Martini Blue,” a historical reject, at rank 9. No other listed method places a historical reject in this sample’s top ten. [v3 comparison](../evaluation_report_v3.md) · [v4 comparison](../evaluation_report_v4.md) · [v5 recap](../evaluation_report_v5.md)

The chance known-like precision is 94/306 = .307190. For a random permutation, the exact expected legacy NDCG is .624087, using the hypergeometric distribution of the number of hits and averaging the discounted hit positions conditional on that count. The corresponding 311-row values are 99/311 = .318328 and .633730. The v3 report quotes Monte Carlo approximations .625 and .628 for legacy NDCG. On these pools the exact expected standard NDCG equals the known-like prevalence.

## Original 311-row comparison

I rebuilt the v6 per-bucket and global scores of all 99 preview-covered holdout likes from their source features and the published v6 distance constants. The 94 likes common to the smaller pool reproduced both stored scores exactly. Combining those 99 likes with the original-order scores of the 12 historical rejects and 200 distractors gives the following results. This recovers both v6 headline pairs in the v2 report. The five discarded likes lack archived v2 scores and the fitted v2 model file is absent, so the original 311-row v2 pair has only a reported value. [Feature rows](../features_v2.json) · [Frozen v6 profile](../scoring.py) · [v2 comparison](../evaluation_report_v2.md)

| Method | Like positions in top ten | P@10 | Standard NDCG@10 | Legacy NDCG@10 |
|---|---|---:|---:|---:|
| v6 per-bucket | 2, 5, 6, 9, 10 | .50 | .432279 | .666139 |
| v6 global | 3, 5, 6, 7, 8, 9 | .60 | .482637 | .663573 |

## Coverage and recording duplicates

There are 650 ordered liked rows, 649 valid Spotify IDs, and zero verified add timestamps. Of 650, 477 carry a Deezer ID, 475 carry librosa features, 447 carry ReccoBeats features, and 327 have both. The newest 130 positions contain 99 preview-feature rows. Removing five cross-split Deezer matches gives the exact set of 94 stored held-out Spotify IDs. Twelve of the 13 configured historical reject IDs occur in the ranking. The missing ID belongs to “Touch.” The v2 headline’s mean reject rank used 12 items. Grouping this liked snapshot alone by provider identity reproduces the v4 report’s 506 training representatives and 98 eligible test representatives. [Feature rows](../features_v2.json) · [Configured historical negative IDs](../model_v2.py) · [Ranked rows](../v3_scores_306.json)

| Holdout title | Deezer ID | Holdout position | Training positions |
|---|---:|---:|---|
| Warriors | 3794801972 | 21 | 263 |
| CRAZY | 2967699621 | 28 | 418 |
| Red Rum | 2980031271 | 30 | 306, 622 |
| [Gen]inside | 4015474791 | 33 | 148 |
| Partyy | 3952394101 | 129 | 163 |

Fourteen Deezer IDs repeat within the liked file. The five cross-split recordings account for six training rows. The 306 scored rows still include two different Spotify IDs for “Control,” Deezer 4015474741, leaving 305 distinct Deezer IDs and 93 distinct held-out Deezer IDs. Neither copy enters a top ten, so deleting one with scores held fixed leaves the table unchanged. Provider IDs establish recording aliases here. Verifying identical preview bytes needs the original audio. The claimed 363-row Bliss seed matrix also requires `data/bliss23.json`. The archived per-item distances suffice to rank this sample. No full 701-row Bliss score export was found, so the reported reversal against the per-bucket baseline on the full pool remains unverified. [Stored rows](../v3_scores_306.json) · [Bliss seed selection](../integrate_v3.py)

## Live keep rates and their Wilson intervals

The two-sided 95% Wilson calculation uses a standard-normal 97.5th percentile of 1.959964. A keep requires membership in Liked Songs. Each rate over originally offered tracks needs an original offer list and a rating for each unique track. The observed shares below use the current visible collections as denominators. Removing an unsaved track changes that denominator, and the v9 objects overlap. The row replay found 26 current memberships across 23 distinct tracks. All three v9 A tracks also belong to v9 B. On every row, Spotify’s entity save status agreed with presence in the returned Liked Songs pages. The interval endpoints describe the stated counts only. They give no estimate of the missing offers or a comparison across rankers.

| Live collection or scenario | Kept / denominator | Wilson 95% |
|---|---:|---:|
| Visible v8 playlist | 1/6 | .0301–.5635 |
| Visible three-track v9 playlist | 1/3 | .0615–.7923 |
| Visible ten-track v9 playlist | 1/10 | .0179–.4042 |
| Visible v10 playlist | 0/7 | .0000–.3543 |
| Historical six-of-ten claim, given those counts | 6/10 | .3127–.8318 |
| First live checkpoint boundary | 31/42 | .5893–.8470 |
| First live checkpoint pass count | 32/42 | .6147–.8652 |
| Second live checkpoint boundary | 36/50 | .5833–.8253 |
| Second live checkpoint pass count | 37/50 | .6045–.8413 |

The historical Wilson values agree with the research specification at its displayed precision. The checkpoint rows are prospective arithmetic. I independently counted the private tokenized collection rows with `python audit/recompute.py --spotify-snapshot PATH`. The local input was deleted after validation. The snapshot option accepts a JSON map from collection name to rows containing a track token, `library_status` (`SAVED` or `NOT_SAVED`), and `in_liked_collection` (boolean). Use `python audit/recompute.py --keeps 1 6 --keeps 1 3 --keeps 1 10 --keeps 0 7` to reproduce all interval endpoints above. [Historical threshold specification](../research/eval-spec.md)

## Fresh shortfall and missing evidence

The pinned gate requires ten eligible fresh likes and two eligible fresh rejects. Under the report’s claimed one eligible like and 11 eligible rejects, the arithmetic yields max(0, 10−1) = 9 more likes and max(0, 2−11) = 0 more rejects. If the single eligible like and its reported rank outside the top ten for every arm hold, each arm has P@10 = 0 and standard NDCG@10 = 0. Both statements are conditional. A Spotify save establishes a keep. Eligibility additionally requires a verified join to a feature-covered frozen catalog recording and exclusion of historical labels or conflicts. The 200 stored catalog distractors carry Deezer IDs but no Spotify IDs or ISRCs. The two saved tracks seen in the recovered v8 and v9 playlists each have zero Spotify-ID or normalized title-and-first-artist matches among those 200, the 650 frozen liked rows, and the ten archived recommendations. The other 390 catalog rows could still contain a match. The v9 catalog join therefore remains open. [Gate](../integrate_v4.py) · [Manifest rules](../evaluate_v4.py) · [Claim under review](../evaluation_report_v5.md)

A recovered scored export can be replayed with `python audit/recompute.py --trial /path/to/evaluation-v4-fresh-corrected-enriched.json`. The `supplied_trial` section counts unique declared test labels, sorts the four full-pool score columns with their stored tie keys, validates each rank, and calculates P@10 and standard NDCG@10. For a fresh trial it also calculates both shortfalls and checks that scored likes and rejects link to consistent fresh-judgment rows in their declared catalog recording groups. Position folds return ranking metrics only for this purpose. The replay uses the recording resolution and label provenance declared in the supplied manifest. Verifying how the run assigned identities, removed aliases or acquired its labels requires the original files.

A historical synthetic position fixture reproduced both metrics for all four arms. A second synthetic trial from the same source fixture with one fresh like and two rejects reproduced the independently expected nine-like and zero-reject shortfalls and all four arms’ metrics. The replay rejected files with missing or duplicate scored rows, wrong ranks, invalid labels, nonfinite scores, false judgment links, and extra score columns. Neither fixture contributes evidence about the listener.

The checkout, its fetched branch histories, local v4 bundle, and workspace scratch inventory lack the production `data/pool_features.json` and `data/negatives_features.json`. Recovering the eligibility count also requires original-offer and judgment membership, the historical exclusion identities, and a frozen Spotify-to-catalog crosswalk such as the ReccoBeats records carrying Spotify hrefs and ISRCs. A complete per-item corrected scored manifest would allow an independent recount of its declared labels and comparison of arm ranks. Verifying its joins against source data additionally requires the frozen inputs. Scratch `catalog.json`, `likes.json`, and `rejects.json` from the v4 verification run use synthetic IDs such as `c0`. The only checked-in model artifact for v2 contains summary parameters. The fitted `model_v2.npz` and original preview directory are absent.
