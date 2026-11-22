#!/usr/bin/env python3
"""v3 integration: decontaminate the shared pool to 306 items, re-score
every candidate on it, compute the pre-registered 4-leg GO verdict per
candidate via evaluate_v3.verdict_4leg, and write evaluation_v3.json +
v3_scores_306.json (per-item audit file the checks re-derive from).

Candidates:
  troi            Troi patch (S3 audio-neighbour + S4 same-artist, zipper)
  embeat_full     Embeat port, all four channels, medoid seed
  embeat_minus_ch1  Embeat port without channel 1 (similar)
  bliss_fixed     bliss23 fixed VERSION2_WEIGHTS, 10-NN mean distance
  bliss_adaptive  bliss23 inverse-variance (adaptive) weights
  bliss_v2_blend  z-scored mean of v2 and -bliss distance, w_v2 = 0.5
                  (equal weight: no train-only fit identifies an interior
                  weight — see BLEND_WEIGHT_NOTE in the output JSON)
  v2_unchanged    v2 as-is (v3 ships harness + ports, no new ranker)

Rules honoured here:
  * Nothing in the Wave-1 modules is edited; all scoring goes through
    their public functions.
  * The 5 duplicate-audio held-out likes are removed from the pool for
    EVERY candidate (306 = 94 held-out + 12 negatives + 200 distractors).
  * Baselines are recomputed on the 306 pool, not copied from any JSON.
  * Point metrics use evaluate.rank_metrics (the Wave-1 convention:
    stable sort, input order breaks ties) on lists ordered exactly as
    each Wave-1 team ordered them. MAP@10/MRR@10 come from
    evaluate_v3.metrics_from_rel on the top-10 relevance vector.
  * The 4-leg verdict is computed ONLY by evaluate_v3.verdict_4leg.
  * sampled_disagreement is the P-10 check: sign(delta_P@10) on the
    306 sampled pool vs sign(delta_P@10) on the full 701-item pool
    (99 held-out + 12 negatives + all 590 pool tracks).
"""
import json, os, sys, time

import numpy as np

BASE = os.path.expanduser('~/workspace/audio-pipeline')
sys.path.insert(0, BASE)
os.chdir(BASE)

import evaluate_v3 as ev3
from evaluate import rank_metrics

REMOVED_TITLES = {'Warriors', 'CRAZY', 'Red Rum', '[Gen]inside', 'Partyy'}


def bucket_map_and_fn():
    """Artist -> modal genre bucket, and bucket_for(rec), identical to
    evaluate.py / troi._build_eval_items."""
    from collections import Counter
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    tracks = [t for t in doc['tracks'] if t['spotify_id'] != 'None']
    artist_bucket = {}
    for t in tracks:
        a = (t.get('artist') or '').split(',')[0].strip().lower()
        b = t.get('genre_bucket')
        if a and b:
            artist_bucket.setdefault(a, Counter())[b] += 1
    artist_bucket = {a: c.most_common(1)[0][0]
                     for a, c in artist_bucket.items()}

    def bucket_for(rec):
        b = rec.get('genre_bucket')
        if b in ('E', 'R', 'H'):
            return b
        a = (rec.get('artist') or '').split(',')[0].strip().lower()
        return artist_bucket.get(a)

    return bucket_for


def point_metrics(ordered_items, score_key, higher_better, n_relevant=94):
    """Wave-1-convention point metrics from an already-ordered list."""
    m = rank_metrics([dict(x) for x in ordered_items], score_key,
                     higher_better=higher_better)
    rel = [x['relevant'] for x in ordered_items[:10]]
    extra = ev3.metrics_from_rel(rel, n_relevant)
    neg_top = [x for x in ordered_items[:10] if x['kind'] == 'negative']
    m['MAP@10'] = extra['MAP@10']
    m['MRR@10'] = extra['MRR@10']
    m['n_neg_top10'] = len(neg_top)
    m['n_neg_top10_H'] = sum(1 for x in neg_top
                             if x.get('bucket') == 'H')
    m['top10_ids'] = [x['rec']['spotify_id']
                      for x in ordered_items[:10]]
    return m


def main():
    t0 = time.time()
    import troi_patch as tp
    import embeat_port as ep
    import bliss_features as bf
    from scoring import V2Model, v6_score

    bucket_for = bucket_map_and_fn()

    # ---- 1. rebuild the 311 pool, verify baselines vs evaluation_v2.json
    items311, counts311 = tp._build_eval_items()
    assert counts311['n_ranked'] == 311, counts311
    ref = json.load(open(os.path.join(BASE, 'evaluation_v2.json')))
    for key, mkey in (('v2', 'v2'), ('v6b', 'v6_per_bucket'),
                      ('v6g', 'v6_global')):
        got = rank_metrics([dict(x) for x in items311], key,
                           higher_better=(key == 'v2'))
        for k in ('P@10', 'NDCG@10', 'mean_rank_heldout_like',
                  'mean_rank_negative'):
            assert got[k] == ref[mkey][k], (mkey, k, got[k], ref[mkey][k])
    print('311-pool baselines reproduce evaluation_v2.json exactly')

    # ---- 2. decontaminate: drop the 5 duplicate-audio held-out likes
    dups = tp.find_duplicate_audio_items(items311)
    dup_titles = {x['rec']['title'] for x in dups}
    assert dup_titles == REMOVED_TITLES, dup_titles
    assert len(dups) == 5
    dup_ids = {id(x) for x in dups}
    items = [x for x in items311 if id(x) not in dup_ids]
    assert len(items) == 306
    assert sum(1 for x in items if x['kind'] == 'heldout') == 94
    assert sum(1 for x in items if x['kind'] == 'negative') == 12
    assert sum(1 for x in items if x['kind'] == 'distractor') == 200
    removed = [{'title': x['rec']['title'], 'artist': x['rec']['artist'],
                'spotify_id': x['rec']['spotify_id'],
                'deezer_id': x['rec'].get('deezer_id')} for x in dups]
    for x in items:
        x['bucket'] = bucket_for(x['rec'])
    print('decontaminated pool: 306 = 94 held-out + 12 negatives + 200 distractors')

    # ---- 3a. Troi on the 306 pool
    troi_order = tp._patch_rank_order(items)
    for pos, x in enumerate(troi_order, 1):
        x['troi'] = -pos
    troi_by_id = {id(x['rec']): x['troi'] for x in troi_order}

    # ---- 3b. Embeat full + minus channel 1 on the 306 pool.
    # rank_eval_pool ranks the 311 pool; per-item (port, cosine) scores
    # are pool-independent, so filter to the 306 ids and re-sort with the
    # identical key (-port, -cosine, title, artist).
    seed_rec = ep.eval_seed()
    assert seed_rec['spotify_id'] == '6HseMEzSf4mxnUem0O1hJp', seed_rec
    embeat_scores = {}
    for name, channels in (
            ('embeat_full', ('similar', 'popular', 'same_artist',
                             'related_artist')),
            ('embeat_minus_ch1', ('popular', 'same_artist',
                                  'related_artist'))):
        ranked311 = ep.rank_eval_pool(seed_rec['spotify_id'],
                                      channels=channels)
        # Key by ep.cand_key (ISRC else title/artist): pool distractors
        # have spotify_id None, so a spotify_id key collapses all 200
        # distractors into one entry (caught 2026-11-22: P@10 read 1.0).
        per_id = {}
        for x in ranked311:
            per_id[ep.cand_key(x['rec'])] = (
                float(x['port']), float(x['port_cosine']),
                x['rec'].get('title', ''), x['rec'].get('artist', ''))
        sub = []
        for x in items:
            k = ep.cand_key(x['rec'])
            assert k in per_id, k
            p, c, ti, ar = per_id[k]
            sub.append((x, p, c, ti, ar))
        sub.sort(key=lambda t: (-t[1], -t[2], t[3], t[4]))
        embeat_scores[name] = [t[0] for t in sub]
        for x, p, c, ti, ar in sub:
            x[name] = p
    print('embeat full + minus-ch1 scored on 306')

    # ---- 3c. Bliss fixed/adaptive + blend on the 306 pool.
    # Seed matrix D: distinct-Deezer train-like bliss vectors, excluding
    # any train row whose audio duplicates a held-out like (the twin
    # guard). 363 rows. This matrix reproduces the Wave-1 bliss numbers
    # on the 311 pool exactly (verified in the probe step); the 377-row
    # and 368-row variants do not.
    bliss = json.load(open(os.path.join(BASE, 'data/bliss23.json')))
    by_dz = {v['deezer_id']: v['bliss23'] for v in bliss.values()
             if v.get('deezer_id')}
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    train = [t for t in doc['tracks']
             if t.get('position') is not None and t['position'] >= 130
             and t['spotify_id'] != 'None']
    heldout_dz = {t['deezer_id'] for t in doc['tracks']
                  if t.get('position') is not None and t['position'] < 130
                  and t.get('deezer_id')}
    seen, rows = set(), []
    for t in train:
        dz = t.get('deezer_id')
        if dz in by_dz and dz not in seen and dz not in heldout_dz:
            seen.add(dz)
            rows.append(by_dz[dz])
    M = np.array(rows)
    assert M.shape == (363, 23), M.shape
    adapt_w = bf.compute_adaptive_weights(M)
    for x in items:
        v = np.array(by_dz[x['rec']['deezer_id']])
        x['bliss_fixed'] = bf.bliss_score_to_liked(
            v, M, bf.VERSION2_WEIGHTS, k=10)
        x['bliss_adaptive'] = bf.bliss_score_to_liked(v, M, adapt_w, k=10)
    # blend: equal-weight z-scored mean of v2 and -bliss distance,
    # z-scored over the 306 pool. See BLEND_WEIGHT_NOTE below.
    def z(a):
        a = np.asarray(a, dtype=float)
        return (a - a.mean()) / a.std()
    v2a = np.array([x['v2'] for x in items])
    bfa = np.array([x['bliss_fixed'] for x in items])
    blend = 0.5 * z(v2a) + 0.5 * z(-bfa)
    for x, s in zip(items, blend):
        x['bliss_v2_blend'] = float(s)
    print('bliss fixed/adaptive/blend scored on 306 (seed matrix 363)')

    # attach troi + embeat scores to the item dicts
    for x in items:
        x['troi'] = troi_by_id[id(x['rec'])]
    # NOTE: embeat scores were attached above in the loop over sub.

    # ---- 4. point metrics on the 306 pool (Wave-1 ordering per method)
    def ordered_by(items, key, higher):
        return sorted(items, key=lambda x: x[key], reverse=higher)

    cands = {}
    specs = [
        ('troi', 'troi', True, troi_order),
        ('embeat_full', 'embeat_full', True, embeat_scores['embeat_full']),
        ('embeat_minus_ch1', 'embeat_minus_ch1', True,
         embeat_scores['embeat_minus_ch1']),
        ('bliss_fixed', 'bliss_fixed', False,
         ordered_by(items, 'bliss_fixed', False)),
        ('bliss_adaptive', 'bliss_adaptive', False,
         ordered_by(items, 'bliss_adaptive', False)),
        ('bliss_v2_blend', 'bliss_v2_blend', True,
         ordered_by(items, 'bliss_v2_blend', True)),
        ('v2_unchanged', 'v2', True, ordered_by(items, 'v2', True)),
    ]
    # baselines on the 306 pool (recomputed, not copied)
    baselines = {}
    for name, key, higher in (('v2', 'v2', True),
                              ('v6_per_bucket', 'v6b', False),
                              ('v6_global', 'v6g', False)):
        baselines[name] = point_metrics(ordered_by(items, key, higher),
                                        key, higher)
    for name, key, higher, order in specs:
        cands[name] = {'score_key': key, 'higher_better': higher,
                       'metrics': point_metrics(order, key, higher)}

    # ---- 5. paired bootstrap vs v6-per-bucket + 4-leg verdicts.
    # ev3 items carry every score column, fed in the CANONICAL build
    # order (held-out, negative, distractor as _build_eval_items built
    # them) for every candidate. This matters because v6b scores are
    # heavily tied (rounded to 2 dp): feeding each candidate's own
    # ranked order changes how v6b ties break inside the bootstrap and
    # makes its point delta disagree with the reported point metrics
    # (caught 2026-11-22: troi delta_NDCG read 0.175 vs 0.820-0.654 =
    # 0.166). Build order matches evaluate.rank_metrics' tie order.
    def ev3_items():
        out = []
        for x in items:
            d = {'spotify_id': x['rec'].get('spotify_id'),
                 'kind': x['kind'], 'relevant': x['relevant'],
                 'artist': x['rec'].get('artist', ''),
                 'bucket': x.get('bucket')}
            for k in ('v2', 'v6b', 'v6g', 'troi', 'embeat_full',
                      'embeat_minus_ch1', 'bliss_fixed', 'bliss_adaptive',
                      'bliss_v2_blend'):
                d[k] = x.get(k)
            out.append(d)
        return out

    rb = ev3.random_baseline(94, 306)
    assert abs(rb['random_P@10'] - 94 / 306) < 1e-9
    random_p10 = round(rb['random_P@10'], 3)          # 0.307
    random_mean_rank = (306 + 1) / 2.0                # 153.5
    v6b_map = baselines['v6_per_bucket']['MAP@10']

    # ---- 6. full-pool (701) P-10 sensitivity for every candidate
    full = build_full_pool(tp, ep, bf, bucket_for, M, adapt_w, by_dz,
                           seed_rec)
    for name, key, higher, order in specs:
        full_m = full[name]
        d_samp = cands[name]['metrics']['P@10'] - baselines['v6_per_bucket']['P@10']
        d_full = full_m['P@10'] - full['v6_per_bucket']['P@10']
        cands[name]['full_pool'] = full_m
        cands[name]['sampled_disagreement'] = ev3.sampled_metric_disagreement(
            d_samp, d_full)
        cands[name]['delta_P@10_fullpool'] = round(d_full, 3)
        cands[name]['delta_P@10_sampled'] = round(d_samp, 3)

    for name, key, higher, order in specs:
        boot = ev3.paired_bootstrap(ev3_items(), key, 'v6b',
                                    higher_a=higher, higher_b=False)
        m = cands[name]['metrics']
        vin = {
            'delta_ndcg_ci_low': boot['delta_ndcg_ci'][0],
            'delta_ndcg': boot['delta_ndcg'],
            'delta_p10': boot['delta_p10'],
            'delta_map10': round(m['MAP@10'] - v6b_map, 3),
            'p10_v3': m['P@10'],
            'p10_v3_ci_low': boot['p10_a_ci'][0],
            'random_p10': random_p10,
            'n_neg_top10': m['n_neg_top10'],
            'n_neg_top10_H': m['n_neg_top10_H'],
            'mean_rank_negative_v3': m['mean_rank_negative'],
            'random_mean_rank': random_mean_rank,
            'sampled_disagreement': bool(cands[name]['sampled_disagreement']),
            'live_keeps': None, 'live_n': None,
        }
        verdict = ev3.verdict_4leg(vin)
        cands[name]['bootstrap_vs_v6b'] = boot
        cands[name]['verdict_inputs'] = vin
        cands[name]['verdict'] = verdict['verdict']
        cands[name]['offline_pass'] = verdict['offline_pass']
        cands[name]['legs'] = verdict['legs']
        print(f"{name}: P@10={m['P@10']} NDCG={m['NDCG@10']} "
              f"neg_top10={m['n_neg_top10']} mean_neg_rank={m['mean_rank_negative']} "
              f"-> {verdict['verdict']}")

    # ---- 7. blend-weight reconstruction note (post-hoc, NOT adopted):
    # w_v2 = 0.585 with pool-z reproduces the Wave-1 311-pool blend
    # numbers exactly. It was recovered FROM the eval pool, so adopting
    # it would be tuning on the test set; the candidate uses w = 0.5.
    items311_blend = None
    blend_recon = None
    try:
        v2_311 = np.array([x['v2'] for x in items311])
        bf_311 = np.array([bf.bliss_score_to_liked(
            np.array(by_dz[x['rec']['deezer_id']]), M,
            bf.VERSION2_WEIGHTS, k=10) for x in items311])
        s = 0.585 * z(v2_311) + 0.415 * z(-bf_311)
        for x, sc in zip(items311, s):
            x['blend_recon'] = float(sc)
        order311 = sorted(items311, key=lambda x: x['blend_recon'],
                          reverse=True)
        blend_recon = point_metrics(order311, 'blend_recon', True,
                                    n_relevant=99)
        blend_recon = {k: blend_recon[k] for k in
                       ('P@10', 'NDCG@10', 'MAP@10', 'MRR@10',
                        'mean_rank_negative', 'n_neg_top10')}
    except Exception as e:  # pragma: no cover - diagnostic only
        blend_recon = {'error': str(e)}

    out = {
        'pool': {
            'n_ranked': 306, 'heldout_total': 129,
            'heldout_with_librosa': 94, 'negatives': 13,
            'negatives_with_librosa': 12, 'distractors': 200,
            'pool_total': 590,
            'n_dup_audio_removed': 5,
            'removed_duplicate_audio': removed,
            'random_P@10': random_p10,
            'random_NDCG@10_v2_convention': round(
                rb['random_NDCG@10'], 3),
            'random_mean_rank': random_mean_rank,
            'split_label': 'position-based holdout (NOT time-based)',
            'note': ('5 held-out likes share a Deezer preview ID with a '
                     'train like (same audio, different Spotify ID); '
                     'removed from the eval pool for ALL v3 verdicts. '
                     'The v2/v6 models were TRAINED on the twin rows, so '
                     'the train side still contains their audio twins; '
                     'only the test side is decontaminated here.'),
        },
        'baselines': baselines,
        'candidates': cands,
        'full_pool': {
            'n_ranked': 701,
            'note': ('99 held-out + 12 negatives + all 590 pool tracks. '
                     'P-10 sensitivity only; the 306 sampled pool governs.'),
            'v6_per_bucket': full['v6_per_bucket'],
        },
        'blend_weight_note': {
            'candidate_weight_w_v2': 0.5,
            'candidate_definition': ('equal-weight z-scored mean of v2 '
                'score and -bliss distance, z-scored over the 306 pool. '
                'No train-only fit identifies an interior weight: a '
                'pseudo-heldout fit on train alone (positions 130-229 as '
                'pseudo-test, 289-row twin-excluded seed matrix) is flat '
                'in P@10 over w in [0.46, 1.0] and rising in NDCG/MAP to '
                'w = 1.0 (pure v2), so no interior optimum exists to fit.'),
            'wave1_reconstruction': blend_recon,
            'wave1_reconstruction_note': ('w_v2 = 0.585 with pool-z '
                'reproduces the Wave-1 311-pool blend numbers exactly '
                '(P@10 0.60, NDCG 0.907, MAP 0.458, MRR 1.0, mean neg '
                'rank 278.7). That weight was recovered FROM the eval '
                'pool after the fact, so it is reported here only as a '
                'reconstruction of what Wave 1 must have used; it is NOT '
                'adopted, because adopting it would be tuning on the '
                'test set. The Wave-1 blend weight itself is undocumented.'),
        },
        'verdict': ('NO-GO — offline' if not any(
            c['offline_pass'] for c in cands.values()) else
            'PENDING — live leg not yet judged (n=0/42)'),
        'recommendations_built': False,
        'recommendations_note': ('No candidate passes Legs 1-3, so per '
            'the eval-spec sequencing no new recommendation list is built '
            'or spent. The v7 playlist (v2 list, pending Kevin rating) '
            'stands as the current best.')
            if not any(c['offline_pass'] for c in cands.values()) else
            'A candidate passes Legs 1-3; see recommendations_v3.json.',
        'wall_s': round(time.time() - t0, 1),
    }
    json.dump(out, open(os.path.join(BASE, 'evaluation_v3.json'), 'w'),
              indent=1)
    audit = {'items': [
        {'spotify_id': x['rec'].get('spotify_id'),
         'deezer_id': x['rec'].get('deezer_id'),
         'title': x['rec'].get('title'),
         'artist': x['rec'].get('artist'),
         'kind': x['kind'], 'relevant': x['relevant'],
         'bucket': x.get('bucket'),
         'scores': {k: x.get(k) for k in
                    ('v2', 'v6b', 'v6g', 'troi', 'embeat_full',
                     'embeat_minus_ch1', 'bliss_fixed', 'bliss_adaptive',
                     'bliss_v2_blend')}}
        for x in items]}
    json.dump(audit, open(os.path.join(BASE, 'v3_scores_306.json'), 'w'),
              indent=1)
    print(f"wrote evaluation_v3.json + v3_scores_306.json in {out['wall_s']}s")
    print('OVERALL VERDICT:', out['verdict'])
    return 0


def build_full_pool(tp, ep, bf, bucket_for, M, adapt_w, by_dz, seed_rec):
    """Score the full 701-item pool (99 held-out + 12 negatives + all 590
    pool tracks) with every candidate + v6b, for the P-10 check.
    Returns {candidate: point-metrics dict} (Wave-1 ordering per method)."""
    from scoring import V2Model, v6_score
    doc = json.load(open(os.path.join(BASE, 'features_v2.json')))
    tracks = [t for t in doc['tracks'] if t['spotify_id'] != 'None']
    neg_recs = json.load(open(os.path.join(BASE, 'data',
                                           'negatives_features.json')))
    pool = json.load(open(os.path.join(BASE, 'data', 'pool_features.json')))
    test_lb = [t for t in tracks if t['position'] < 130 and t.get('librosa')]
    neg_lb = [r for r in neg_recs if r.get('librosa')]
    items = []
    for t in test_lb:
        items.append({'kind': 'heldout', 'relevant': 1, 'rec': t})
    for r in neg_lb:
        items.append({'kind': 'negative', 'relevant': 0, 'rec': r})
    for p in pool:
        items.append({'kind': 'distractor', 'relevant': 0, 'rec': p})
    assert len(items) == 701, len(items)
    for x in items:
        x['bucket'] = bucket_for(x['rec'])
    model = V2Model()
    s = model.score_records([x['rec'] for x in items])
    for x, sc in zip(items, s['final']):
        x['v2'] = round(float(sc), 3)
    for x in items:
        sc_b, _ = v6_score(x['rec'].get('librosa'), x['bucket'])
        x['v6b'] = sc_b
    # bliss scores on the full pool
    for x in items:
        v = np.array(by_dz[x['rec']['deezer_id']])
        x['bliss_fixed'] = bf.bliss_score_to_liked(
            v, M, bf.VERSION2_WEIGHTS, k=10)
        x['bliss_adaptive'] = bf.bliss_score_to_liked(v, M, adapt_w, k=10)

    def z(a):
        a = np.asarray(a, dtype=float)
        return (a - a.mean()) / a.std()
    v2a = np.array([x['v2'] for x in items])
    bfa = np.array([x['bliss_fixed'] for x in items])
    for x, sc in zip(items, 0.5 * z(v2a) + 0.5 * z(-bfa)):
        x['bliss_v2_blend'] = float(sc)
    # troi order on the full pool
    order = tp._patch_rank_order(items)
    for pos, x in enumerate(order, 1):
        x['troi'] = -pos
    # embeat on the full pool: same channel machinery, index over 701
    index = ep.build_index([x['rec'] for x in items],
                           os.path.join(BASE, 'model_v2.npz'))
    embeat_full_orders = {}
    for name, channels in (
            ('embeat_full', ('similar', 'popular', 'same_artist',
                             'related_artist')),
            ('embeat_minus_ch1', ('popular', 'same_artist',
                                  'related_artist'))):
        lists = {}
        if 'similar' in channels:
            lists['similar'] = ep.recall_similar(index, seed_rec, top_k=10)
        if 'popular' in channels:
            lists['popular'] = ep.recall_popular(index, seed_rec, top_k=10)
        if 'same_artist' in channels:
            lists['same_artist'] = ep.recall_same_artist(
                index, seed_rec, top_k=10)
        if 'related_artist' in channels:
            lists['related_artist'] = ep.recall_related_artist(
                index, seed_rec, top_k=10)
        merged = ep._merge_scores(lists)
        by_key = {d['key']: d for d in merged}
        se = ep._seed_emb(index, seed_rec)
        cosines = index['embeddings'] @ se
        scored = []
        for pos_i, x in enumerate(items):
            k = ep.cand_key(x['rec'])
            m = by_key.get(k)
            scored.append((x, float(m['score_raw']) if m else 0.0,
                           float(cosines[pos_i]),
                           x['rec'].get('title', ''),
                           x['rec'].get('artist', '')))
        scored.sort(key=lambda t: (-t[1], -t[2], t[3], t[4]))
        embeat_full_orders[name] = [t[0] for t in scored]
        for x, p, c, ti, ar in scored:
            x[name] = p
    res = {}
    res['v6_per_bucket'] = point_metrics(
        sorted(items, key=lambda x: x['v6b']), 'v6b', False, n_relevant=99)
    res['troi'] = point_metrics(order, 'troi', True, n_relevant=99)
    res['embeat_full'] = point_metrics(
        embeat_full_orders['embeat_full'], 'embeat_full', True,
        n_relevant=99)
    res['embeat_minus_ch1'] = point_metrics(
        embeat_full_orders['embeat_minus_ch1'], 'embeat_minus_ch1', True,
        n_relevant=99)
    for name, key, higher in (
            ('bliss_fixed', 'bliss_fixed', False),
            ('bliss_adaptive', 'bliss_adaptive', False),
            ('bliss_v2_blend', 'bliss_v2_blend', True),
            ('v2_unchanged', 'v2', True)):
        res[name] = point_metrics(
            sorted(items, key=lambda x: x[key], reverse=higher),
            key, higher, n_relevant=99)
    return res


if __name__ == '__main__':
    sys.exit(main())
