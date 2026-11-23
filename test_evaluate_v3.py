#!/usr/bin/env python3
"""Acceptance tests for the v3 eval harness (evaluate_v3.py).

TDD contract (obra TDD skill): these tests were written FIRST, from
research/eval-spec.md §(c) tests T-01..T-11 (plus T-12/T-13 for P-11/P-05
units the spec's list leaves implicit), and run against the v2-only code
before any v3 implementation existed.

Run:  ~/workspace/audio-env/bin/python test_evaluate_v3.py
Exit 0 = every check green. Exit 1 = at least one check red (listed).

RED state (2026-11-21): ALL tests fail — evaluate_v3.py does not exist yet
(ModuleNotFoundError on import). That is the expected RED: every test below
imports the v3 module, so a pass before implementation is impossible.
GREEN requires evaluate_v3.py implementing P-01..P-11 + verdict_4leg.

GROUNDING DELTAS vs the spec (verified from the landed v2 files before
writing these tests; the spec's assumed 343-item pool is STALE):
  G1. Real ranked pool is 311, not 343: 99 held-out likes WITH librosa
     (130 by position, 31 excluded incl. the unresolvable 'None' id at
     position 85), 12 negatives WITH librosa (13 defined; 'Touch' by
     KATSEYE, 6aJn7Cst74cj4lNIiPRgav, has no librosa), 200 distractors.
  G2. Random baseline is therefore 99/311 = 0.318 (not 0.379);
     random mean rank 156.0 (not 172.0); uniform-rank SD 89.78 (not 99.0).
  G3. Negative-control power at the real pool: P(0 of 12 in top-10|random)
     = 0.676 (spec's 0.681 was for 13/343); n_required for alpha=0.05 is 92
     (spec: 102); mean-rank 95% half-width at n=12 is 50.8 ranks (spec: 53.8
     at n=13/343).
  G4. evaluation_v2.json counts.test_likes is 129, not 130: evaluate.py
     drops the 'None'-id track before counting. This file asserts the
     position-based total (130) AND the post-drop total (129) separately.
  G5. Tests live at repo root (test_evaluate_v3.py), NOT in
     research/test_eval_harness.py as the spec §(c) header says — v2
     convention is root-level test files (test_pipeline_v2.py).
  G6. Spec P-08's literal worst-case imputation (excluded held-out likes
     counted as relevance 0 and ranked last) leaves the top 10 unchanged,
     so P@10_all == P@10 by construction; the test asserts the reported
     numbers and the audit list, and the implementation documents this
     degeneracy rather than inventing a different imputation.
  G7. Spec T-02's Monte-Carlo band [0.359, 0.399] is for the stale 343
     pool; the band here is [0.298, 0.338] around the true 0.318.
  G9. The frozen files conditioned the ideal on retrieved hits. These
     tests check standard NDCG against the ideal over all relevant rows.

All expected values below are hand-derived from the grounded counts above
(or from hand-computed relevance vectors), never from the code under test.
No network, no Spotify/Deezer calls — fixtures are synthetic except T-01,
which reads the real landed data files.
"""
import json, math, os, sys

BASE = os.path.expanduser('~/workspace/audio-pipeline')
sys.path.insert(0, BASE)

try:
    import evaluate_v3 as ev3
    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - RED state: module does not exist yet
    ev3 = None
    _IMPORT_ERROR = e

failures = []
passes = []


def check(name, cond, detail=''):
    if cond:
        passes.append(name)
        print(f'PASS  {name}')
    else:
        failures.append(name)
        print(f'FAIL  {name}  {detail}')


def require_module():
    assert ev3 is not None, f'evaluate_v3 import failed: {_IMPORT_ERROR}'


# ---------------------------------------------------------------- fixture
def make_pool(seed=42):
    """Synthetic 311-item pool mirroring the grounded v2 shape:
    99 held-out likes (H000..H098), 12 negatives (N00..N11, N08..N11 are
    the 4 v6 H-bucket thumbs-down), 200 distractors (D000..D199).
    v3 scores (higher better): top 10 = 8 held-out + 2 distractors.
    v6 scores (LOWER better): top 10 = 5 held-out + 5 distractors."""
    items = []
    for i in range(99):
        items.append({'spotify_id': f'H{i:03d}', 'kind': 'heldout',
                      'relevant': 1, 'artist': f'ArtistH{i % 40}',
                      'popularity': 45, 'bucket': 'E'})
    for i in range(12):
        items.append({'spotify_id': f'N{i:02d}', 'kind': 'negative',
                      'relevant': 0, 'artist': f'ArtistN{i}',
                      'popularity': 45,
                      'bucket': 'H' if i >= 8 else 'E'})
    for i in range(200):
        items.append({'spotify_id': f'D{i:03d}', 'kind': 'distractor',
                      'relevant': 0, 'artist': f'ArtistD{i % 60}',
                      'popularity': 40, 'bucket': 'E'})
    by_id = {x['spotify_id']: x for x in items}
    # v3: H000..H007 score 100..93; D000,D001 score 92,91; rest low.
    for i in range(8):
        by_id[f'H{i:03d}']['score_v3'] = 100.0 - i
    by_id['D000']['score_v3'] = 92.0
    by_id['D001']['score_v3'] = 91.0
    for i in range(8, 99):
        by_id[f'H{i:03d}']['score_v3'] = 50.0 - (i - 8) * 0.1
    for i in range(2, 200):
        by_id[f'D{i:03d}']['score_v3'] = 40.0 - i * 0.05
    for i in range(12):
        by_id[f'N{i:02d}']['score_v3'] = -50.0 - i
    # On the v6 ascending scale hits occupy ranks 1, 4, 7, 10.
    # The eight v3 hits occupy ranks 1..8. Both use the same 10-hit ideal.
    v6_held = {0: 1.0, 1: 4.0, 2: 7.0, 3: 10.0, 4: 13.0,
               5: 16.0, 6: 19.0, 7: 22.0}
    for i, v in v6_held.items():
        by_id[f'H{i:03d}']['score_v6'] = v
    d_vals = [d for d in range(2, 46) if float(d) not in set(v6_held.values())]
    for i, v in enumerate(d_vals):
        by_id[f'D{i:03d}']['score_v6'] = float(v)
    for i in range(len(d_vals), 200):
        by_id[f'D{i:03d}']['score_v6'] = 50.0 + i * 0.05
    for i in range(8, 99):
        by_id[f'H{i:03d}']['score_v6'] = 50.0 + (i - 8) * 0.1
    for i in range(12):
        by_id[f'N{i:02d}']['score_v6'] = 200.0 + i
    return items


# ---------------------------------------------------------------- T-01
def test_t01():
    """T-01 (P-01/P-06): real-data pool audit. Fails pre-port: v2 has no
    such assertion; its docstring says 9 negatives while 13 are defined."""
    require_module()
    audit = ev3.audit_real_pool()
    check('T-01 test total by position == 130', audit['test_likes_total'] == 130,
          f"got {audit['test_likes_total']}")
    check('T-01 test post-None-drop == 129', audit['test_likes_after_none_drop'] == 129,
          f"got {audit['test_likes_after_none_drop']}")
    check('T-01 test with librosa == 99', audit['test_with_librosa'] == 99,
          f"got {audit['test_with_librosa']}")
    check('T-01 negatives defined == 13', audit['negatives_defined'] == 13,
          f"got {audit['negatives_defined']}")
    check('T-01 negatives with librosa == 12', audit['negatives_with_librosa'] == 12,
          f"got {audit['negatives_with_librosa']}")
    check('T-01 excluded negative is Touch/KATSEYE',
          audit['negatives_excluded_ids'] == ['6aJn7Cst74cj4lNIiPRgav'],
          f"got {audit['negatives_excluded_ids']}")
    check('T-01 distractors == 200', audit['distractors'] == 200,
          f"got {audit['distractors']}")
    check('T-01 n_ranked == 311', audit['n_ranked'] == 311,
          f"got {audit['n_ranked']}")
    check('T-01 negatives disjoint from heldout',
          audit['negatives_disjoint_heldout'] is True)
    check('T-01 negatives disjoint from train',
          audit['negatives_disjoint_train'] is True)
    check('T-01 distractors disjoint from heldout+negatives',
          audit['distractors_disjoint'] is True)
    check('T-01 split labelled position-based, not time-based',
          audit['split_label'] == 'position-based holdout (NOT time-based)',
          f"got {audit['split_label']!r}")
    check('T-01 no added_at anywhere (0/650)',
          audit['added_at_coverage'] == 0, f"got {audit['added_at_coverage']}")


# ---------------------------------------------------------------- T-02
def test_t02():
    """T-02 (P-01): random baseline calibration on the grounded 311 pool.
    Fails pre-port: v2 reports no random baseline at all."""
    require_module()
    import numpy as np
    items = make_pool()
    rb = ev3.random_baseline(n_heldout_with_features=99, n_ranked=311, k=10)
    check('T-02 random_P@10 == 99/311', abs(rb['random_P@10'] - 99 / 311) < 0.001,
          f"got {rb['random_P@10']}")
    check('T-02 exact random NDCG@10 == 99/311 when at least 10 are relevant',
          abs(rb['random_NDCG@10'] - 99/311) < 1e-12)
    check('T-02 random_Recall@10 == 10*99/311/99',
          abs(rb['random_Recall@10'] - (10 * 99 / 311) / 99) < 0.001,
          f"got {rb['random_Recall@10']}")
    check('T-02 random mean rank == 156.0', abs(rb['random_mean_rank'] - 156.0) < 0.001,
          f"got {rb['random_mean_rank']}")
    # Monte-Carlo: 200 uniform-random rankings, seeds 42..241.
    p10s, ndcgs = [], []
    for s in range(42, 242):
        rng = np.random.RandomState(s)
        scored = [dict(x, score=float(rng.rand())) for x in items]
        m = ev3.rank_metrics(scored, 'score', higher_better=True)
        p10s.append(m['P@10'])
        ndcgs.append(m['NDCG@10'])
    mean_p = float(np.mean(p10s))
    mean_n = float(np.mean(ndcgs))
    check('T-02 mean random P@10 in [0.298, 0.338]', 0.298 <= mean_p <= 0.338,
          f'got {mean_p:.3f}')
    check('T-02 mean random standard NDCG@10 in [0.298, 0.338]',
          0.298 <= mean_n <= 0.338, f'got {mean_n:.3f}')
    m50 = ev3.rank_metrics([dict(x, score=x['score_v3']) for x in items],
                           'score', higher_better=True)
    check('T-02 BELOW_RANDOM false at P@10=0.80',
          ev3.random_baseline(99, 311)['random_P@10'] < m50['P@10']
          and m50.get('below_random') is False,
          f"got below_random={m50.get('below_random')}")


# ---------------------------------------------------------------- T-03
def test_t03():
    """T-03 (P-04/P-05): degraded-model detection. Fails pre-port: v2 has
    no GO gate and no popularity flag, so A-C produce numbers, no verdict."""
    require_module()
    import numpy as np
    items = make_pool()
    m_v3 = ev3.rank_metrics([dict(x, score=x['score_v3']) for x in items],
                            'score', higher_better=True)
    check('T-03 fixture v3 P@10 == 0.80', abs(m_v3['P@10'] - 0.80) < 0.001,
          f"got {m_v3['P@10']}")
    # A: inverted scores.
    bad = [dict(x, score=-x['score_v3']) for x in items]
    m_bad = ev3.rank_metrics(bad, 'score', higher_better=True)
    check('T-03A inverted P@10 <= 0.20', m_bad['P@10'] <= 0.20,
          f"got {m_bad['P@10']}")
    check('T-03A inverted NDCG < v3 NDCG - 0.20',
          m_bad['NDCG@10'] < m_v3['NDCG@10'] - 0.20,
          f"bad={m_bad['NDCG@10']} v3={m_v3['NDCG@10']}")
    check('T-03A inverted mean_rank_negative < 156 (negatives float up)',
          m_bad['mean_rank_negative'] < 156.0,
          f"got {m_bad['mean_rank_negative']}")
    v_bad = ev3.verdict_4leg({'delta_ndcg_ci_low': -0.5, 'delta_ndcg': -0.4,
                              'delta_p10': -0.6, 'delta_map10': -0.3,
                              'p10_v3': m_bad['P@10'], 'p10_v3_ci_low': 0.0,
                              'random_p10': 99 / 311,
                              'n_neg_top10': m_bad['n_neg_top10'],
                              'n_neg_top10_H': m_bad['n_neg_top10_H'],
                              'mean_rank_negative_v3': m_bad['mean_rank_negative'],
                              'random_mean_rank': 156.0,
                              'sampled_disagreement': False,
                              'live_keeps': None, 'live_n': None})
    check('T-03A GO gate False for inverted model', v_bad['verdict'].startswith('NO-GO'),
          f"got {v_bad['verdict']}")
    # B: shuffled scores, one draw near chance; 50 shuffles mean near chance.
    rng = np.random.RandomState(7)
    scores = np.array([x['score_v3'] for x in items], dtype=float)
    rng.shuffle(scores)
    m_shuf = ev3.rank_metrics([dict(x, score=float(s)) for x, s in zip(items, scores)],
                              'score', higher_better=True)
    check('T-03B single shuffle |P@10 - 0.318| <= 0.25',
          abs(m_shuf['P@10'] - 99 / 311) <= 0.25, f"got {m_shuf['P@10']}")
    means = []
    for s in range(50):
        r = np.random.RandomState(1000 + s)
        sc = np.array([x['score_v3'] for x in items], dtype=float)
        r.shuffle(sc)
        means.append(ev3.rank_metrics(
            [dict(x, score=float(v)) for x, v in zip(items, sc)],
            'score', higher_better=True)['P@10'])
    mean_shuf = float(np.mean(means))
    check('T-03B mean of 50 shuffles in [0.268, 0.368]',
          0.268 <= mean_shuf <= 0.368, f'got {mean_shuf:.3f}')
    # C: popularity-only scores; top 10 = 10 most popular distractors.
    pop_items = []
    for x in items:
        y = dict(x)
        y['popularity'] = 95 if y['spotify_id'] in {f'D{i:03d}' for i in range(10)} else 40
        if y['kind'] == 'heldout':
            y['popularity'] = 45
        y['score'] = float(y['popularity'])
        pop_items.append(y)
    m_pop = ev3.rank_metrics(pop_items, 'score', higher_better=True)
    pa = ev3.popularity_audit(pop_items, 'score', higher_better=True)
    check('T-03C popularity-only top10 has 0 held-out likes',
          m_pop['P@10'] == 0.0, f"got {m_pop['P@10']}")
    check('T-03C POPULARITY_BIAS raised', pa['popularity_bias'] is True,
          f"got {pa}")
    check('T-03C popularity gap > 15 points', pa['gap'] > 15, f"gap={pa['gap']}")
    v_pop = ev3.verdict_4leg({'delta_ndcg_ci_low': -0.1, 'delta_ndcg': -0.2,
                              'delta_p10': -0.5, 'delta_map10': -0.1,
                              'p10_v3': 0.0, 'p10_v3_ci_low': 0.0,
                              'random_p10': 99 / 311, 'n_neg_top10': 0,
                              'n_neg_top10_H': 0, 'mean_rank_negative_v3': 200.0,
                              'random_mean_rank': 156.0,
                              'sampled_disagreement': False,
                              'live_keeps': None, 'live_n': None})
    check('T-03C GO gate False for popularity-only model',
          v_pop['verdict'].startswith('NO-GO'), f"got {v_pop['verdict']}")


# ---------------------------------------------------------------- T-04
def test_t04():
    """T-04 (P-06): positive control — memorisation must be crowned, and
    the leakage guard must fire when train contains a test item."""
    require_module()
    items = make_pool()
    cheat = [dict(x, score=100.0 if x['kind'] == 'heldout' else 0.0)
             for x in items]
    m = ev3.rank_metrics(cheat, 'score', higher_better=True)
    check('T-04 memorised model P@10 >= 0.90', m['P@10'] >= 0.90,
          f"got {m['P@10']}")
    check('T-04 memorised model NDCG@10 >= 0.90', m['NDCG@10'] >= 0.90,
          f"got {m['NDCG@10']}")
    train_ids = {'H000', 'D000'}  # H000 is a held-out like: leakage.
    heldout_ids = {f'H{i:03d}' for i in range(99)}
    try:
        ev3.check_leakage(top10_ids=[x['spotify_id'] for x in
                                     ev3.rank_items(cheat, 'score', True)[:10]],
                          train_ids=train_ids, heldout_ids=heldout_ids,
                          negative_ids={f'N{i:02d}' for i in range(12)},
                          distractor_ids={f'D{i:03d}' for i in range(200)})
        raised = False
    except (AssertionError, ValueError):
        raised = True
    check('T-04 leakage guard fires when train contains a test item', raised)
    # Guard must NOT fire on a clean fixture.
    try:
        ev3.check_leakage(top10_ids=[x['spotify_id'] for x in
                                     ev3.rank_items(cheat, 'score', True)[:10]],
                          train_ids={'TRAIN1', 'TRAIN2'},
                          heldout_ids=heldout_ids,
                          negative_ids={f'N{i:02d}' for i in range(12)},
                          distractor_ids={f'D{i:03d}' for i in range(200)})
        clean_ok = True
    except (AssertionError, ValueError):
        clean_ok = False
    check('T-04 leakage guard passes on clean fixture', clean_ok)


# ---------------------------------------------------------------- T-05
def test_t05():
    """T-05 (P-04): 5-seed distractor sweep + paired bootstrap CI.
    Fails pre-port: v2 runs one seed, so the range is unmeasured."""
    require_module()
    import numpy as np
    # Tail draws change how many distractors outrank the fixed held-out
    # scores. The five draws measure sensitivity to distractor composition.
    per_seed = []
    for seed in (42, 43, 44, 45, 46):
        rng = np.random.RandomState(seed)
        cand = []
        for i in range(5000):
            s3 = float(3.0 * (i / 4999.0) ** 3 + rng.randn() * 0.02)
            cand.append({'spotify_id': f'C{i:04d}', 'kind': 'distractor',
                         'relevant': 0, 'score_v3': s3,
                         'score_v6': -s3 + float(rng.randn() * 0.02)})
        draws = ev3.draw_distractors(cand, 200, seed)
        items = []
        v3_h = {0: 3.5, 1: 3.2, 2: 2.8, 3: 2.5}
        v6_h = {0: -3.5, 1: -2.0, 2: -1.9, 3: -1.8, 4: -1.7}
        for i in range(99):
            items.append({'spotify_id': f'H{i:03d}', 'kind': 'heldout',
                          'relevant': 1,
                          'score_v3': v3_h.get(i, -1.0),
                          'score_v6': v6_h.get(i, 1.0)})
        for i in range(12):
            items.append({'spotify_id': f'N{i:02d}', 'kind': 'negative',
                          'relevant': 0, 'score_v3': -5.0,
                          'score_v6': 5.0})
        items += draws
        m3 = ev3.rank_metrics([dict(x, score=x['score_v3']) for x in items],
                              'score', higher_better=True)
        m6 = ev3.rank_metrics([dict(x, score=x['score_v6']) for x in items],
                              'score', higher_better=False)
        per_seed.append({'p10_a': m3['P@10'], 'p10_b': m6['P@10'],
                         'ndcg_a': m3['NDCG@10'], 'ndcg_b': m6['NDCG@10']})
    sweep = ev3.seed_sweep_summary(per_seed)
    check('T-05 P@10 range across 5 seeds <= 0.20 (v3)',
          sweep['p10_a_range'] <= 0.20, f"range={sweep['p10_a_range']}")
    check('T-05 P@10 range across 5 seeds <= 0.20 (v6)',
          sweep['p10_b_range'] <= 0.20, f"range={sweep['p10_b_range']}")
    check('T-05 sign(delta_NDCG) same in >= 4 of 5 seeds',
          sweep['ndcg_sign_agreement'] >= 4,
          f"agreement={sweep['ndcg_sign_agreement']}/5 sweep={sweep}")
    check('T-05 sweep reports 5 seeds', sweep['n_seeds'] == 5)
    # Independent full-pool IDCG values (10 hits in the ideal) give
    # 0.870125 versus 0.451865 on this fixture.
    items = make_pool()
    boot = ev3.paired_bootstrap(items, 'score_v3', 'score_v6',
                                higher_a=True, higher_b=False,
                                B=2000, seed=42)
    check('T-05 bootstrap delta_P@10 == 0.40',
          abs(boot['delta_p10'] - 0.40) < 0.001, f"got {boot['delta_p10']}")
    check('T-05 bootstrap delta_NDCG == 0.418',
          abs(boot['delta_ndcg'] - 0.418) < 0.001, f"got {boot['delta_ndcg']}")
    check('T-05 bootstrap B recorded', boot['B'] == 2000)
    strong = make_pool()
    rng_s = np.random.RandomState(7)
    for x in strong:
        x['score_v3'] = 100.0 if x['kind'] == 'heldout' else 0.0
        x['score_v6'] = float(rng_s.rand())
    boot_s = ev3.paired_bootstrap(strong, 'score_v3', 'score_v6',
                                  higher_a=True, higher_b=True,
                                  B=2000, seed=42)
    check('T-05 strong-gap bootstrap delta_NDCG == 0.849',
          abs(boot_s['delta_ndcg'] - 0.849) < 0.001,
          f"got {boot_s['delta_ndcg']}")
    check('T-05 strong-gap bootstrap CI low on delta_NDCG > 0',
          boot_s['delta_ndcg_ci'][0] > 0.0,
          f"CI={boot_s['delta_ndcg_ci']}")
    check('T-05 strong-gap bootstrap CI low >= 0.25',
          boot_s['delta_ndcg_ci'][0] >= 0.25,
          f"CI={boot_s['delta_ndcg_ci']}")


# ---------------------------------------------------------------- T-06
def test_t06():
    """T-06 (P-01 + §d): negative-control power at the GROUNDED pool.
    Uniform ranks sampled without replacement give P(no top-ten reject)
    = comb(299, 10)/comb(311, 10), with a finite-population rank SE. Fails pre-port: v2 reports a
    mean negative rank with no uncertainty and no required-n."""
    require_module()
    nc = ev3.negative_control_stats(n_negatives_ranked=12, n_ranked=311,
                                    k=10, n_negatives_defined=13)
    check('T-06 P(0 of 12 negs in top10|random) == 0.671',
          abs(nc['p_zero_neg_in_top10_random'] - 0.671) < 0.001,
          f"got {nc['p_zero_neg_in_top10_random']}")
    check('T-06 n_required_for_alpha_0.05 == 80',
          nc['n_required_for_alpha_0.05'] == 80,
          f"got {nc['n_required_for_alpha_0.05']}")
    check('T-06 mean-rank half-width == 49.9',
          abs(nc['mean_rank_half_width'] - 49.9) < 0.1,
          f"got {nc['mean_rank_half_width']}")
    check('T-06 negative control labelled UNDERPOWERED',
          nc['power_label'] == 'UNDERPOWERED', f"got {nc['power_label']}")
    check('T-06 ranked negatives (12) distinguished from defined (13)',
          nc['n_negatives_ranked'] == 12 and nc['n_negatives_defined'] == 13,
          f"got {nc}")


# ---------------------------------------------------------------- T-07
def test_t07():
    """T-07 (P-03): metric definitions on hand-computed vectors.
    Fails pre-port: v2 has no MAP/MRR and no `defined` flag."""
    require_module()
    cases = [
        ([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 0.20, 0.359, 1.000),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 0.10, 0.139, 0.500),
        ([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 0.20, 0.330, 1.000),
    ]
    for rel, p, n, m in cases:
        got = ev3.metrics_from_rel(rel, n_relevant=130)
        check(f'T-07 rel={rel[:3]} P@10 == {p}', abs(got['P@10'] - p) < 0.001,
              f"got {got['P@10']}")
        check(f'T-07 rel={rel[:3]} NDCG@10 == {n}', abs(got['NDCG@10'] - n) < 0.001,
              f"got {got['NDCG@10']}")
        check(f'T-07 rel={rel[:3]} MRR@10 == {m}', abs(got['MRR@10'] - m) < 0.001,
              f"got {got['MRR@10']}")
    zero = ev3.metrics_from_rel([0] * 10, n_relevant=130)
    check('T-07 all-zero P@10 == 0.0', zero['P@10'] == 0.0)
    check('T-07 all-zero NDCG@10 == 0.0', zero['NDCG@10'] == 0.0)
    check('T-07 all-zero defined is False', zero['defined'] is False)
    check('T-07 all-zero n_hits == 0', zero['n_hits'] == 0)
    check('T-07 all-zero carries a warning', bool(zero.get('warning')),
          f"got {zero}")
    got = ev3.metrics_from_rel([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], n_relevant=130)
    check('T-07 MAP@10 == 0.167 (min(k,n_relevant) norm)',
          abs(got['MAP@10'] - 0.167) < 0.001, f"got {got['MAP@10']}")
    check('T-07 metric_definitions names log2 discount',
          ev3.METRIC_DEFINITIONS['NDCG@10']['discount'] == 'log2')
    check('T-07 metric_definitions names map_at_k normalisation',
          'min(k, n_relevant)' in ev3.METRIC_DEFINITIONS['MAP@10']['normalisation'])


# ---------------------------------------------------------------- T-08
def test_t08():
    """T-08 (P-07): determinism + average-rank tie handling. Fails
    pre-port: v2 sorts by score only; tie order follows input order."""
    require_module()
    items = make_pool()
    for x in items[:5]:  # 5 items share the exact top score
        x['score_v3'] = 999.0
    r1 = ev3.rank_items([dict(x, score=x['score_v3']) for x in items],
                        'score', higher_better=True)
    r2 = ev3.rank_items([dict(x, score=x['score_v3']) for x in items],
                        'score', higher_better=True)
    check('T-08 two runs bit-identical JSON',
          json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True))
    tied = [x for x in r1 if x['score'] == 999.0]
    check('T-08 5 tied items at positions 1-5',
          sorted(x['position'] for x in tied) == [1, 2, 3, 4, 5])
    check('T-08 tied items each get average rank 3.0',
          all(abs(x['rank'] - 3.0) < 1e-9 for x in tied),
          f"got {[x['rank'] for x in tied]}")
    check('T-08 n_tied_scores recorded',
          ev3.rank_metrics([dict(x, score=x['score_v3']) for x in items],
                           'score', higher_better=True)['n_tied_scores'] >= 5)


# ---------------------------------------------------------------- T-09
def test_t09():
    """T-09 (P-02): date guard. Fails pre-port: v2's docstring calls the
    split 'most recent 20%' with no date check anywhere."""
    require_module()
    try:
        ev3.assert_split_allowed(split='time-based', added_at_coverage=0,
                                 total_liked=650)
        refused = False
        msg = ''
    except SystemExit as e:
        refused = True
        msg = str(e)
    check('T-09 time-based split refused with no added_at', refused)
    check('T-09 refusal names TIME_SPLIT_REFUSED', 'TIME_SPLIT_REFUSED' in msg,
          f'msg={msg!r}')
    check('T-09 refusal states coverage 0/650', '0/650' in msg, f'msg={msg!r}')
    label = ev3.split_label(split='position-based')
    check('T-09 position split labelled NOT time-based',
          label == 'position-based holdout (NOT time-based)', f'got {label!r}')
    # Position-based must NOT raise.
    try:
        ev3.assert_split_allowed(split='position-based', added_at_coverage=0,
                                 total_liked=650)
        pos_ok = True
    except SystemExit:
        pos_ok = False
    check('T-09 position-based split allowed without dates', pos_ok)
    # 94% coverage (< 95% / < 618 of 650) must still refuse time-based.
    try:
        ev3.assert_split_allowed(split='time-based', added_at_coverage=611,
                                 total_liked=650)
        refused94 = False
    except SystemExit:
        refused94 = True
    check('T-09 time-based refused at 611/650 (<95%)', refused94)


# ---------------------------------------------------------------- T-10
def test_t10():
    """T-10 (P-08): missing-feature sensitivity. Fails pre-port: v2 prints
    exclusion counts to stdout only; none are in the JSON and no worst-case
    metric exists."""
    require_module()
    items = make_pool()
    excluded_ids = [f'H{i:03d}' for i in range(90, 100)]  # 10 held-out, no librosa
    ranked = [x for x in items if x['spotify_id'] not in set(excluded_ids)]
    # Rebuild fixture scores so the ranked set still has a top 10 to score.
    m = ev3.rank_metrics([dict(x, score=x['score_v3']) for x in ranked],
                         'score', higher_better=True)
    audit = ev3.missing_feature_audit(n_heldout_total=130,
                                      excluded_ids=excluded_ids,
                                      ranked_metrics=m,
                                      n_heldout_ranked=89)
    check('T-10 n_heldout_excluded == 10', audit['n_heldout_excluded'] == 10,
          f"got {audit['n_heldout_excluded']}")
    check('T-10 excluded ids listed', audit['excluded_spotify_ids'] == excluded_ids)
    check('T-10 P@10_all reported beside P@10',
          'P@10_all' in audit and 'P@10' in audit,
          f"got {audit}")
    check('T-10 P@10_all == P@10 under literal bottom-rank imputation (G6)',
          abs(audit['P@10_all'] - audit['P@10']) < 1e-9,
          f"P@10={audit['P@10']} P@10_all={audit['P@10_all']}")
    check('T-10 audit states the imputation degeneracy',
          'degenerate' in audit.get('imputation_note', '').lower()
          or 'unchanged' in audit.get('imputation_note', '').lower(),
          f"note={audit.get('imputation_note')!r}")


# ---------------------------------------------------------------- T-11
def test_t11():
    """T-11 (P-10): full-pool disagreement flag. Fails pre-port: v2 has no
    full-pool run."""
    require_module()
    check('T-11 opposite signs -> disagreement True',
          ev3.sampled_metric_disagreement(delta_sampled=+0.10,
                                          delta_full=-0.05) is True)
    check('T-11 same signs -> disagreement False',
          ev3.sampled_metric_disagreement(delta_sampled=+0.10,
                                          delta_full=+0.05) is False)
    v = ev3.verdict_4leg({'delta_ndcg_ci_low': 0.05, 'delta_ndcg': 0.10,
                          'delta_p10': 0.10, 'delta_map10': 0.03,
                          'p10_v3': 0.60, 'p10_v3_ci_low': 0.40,
                          'random_p10': 99 / 311, 'n_neg_top10': 0,
                          'n_neg_top10_H': 0, 'mean_rank_negative_v3': 200.0,
                          'random_mean_rank': 156.0,
                          'sampled_disagreement': True,
                          'live_keeps': 35, 'live_n': 42})
    check('T-11 disagreement forces NO-GO despite passing legs',
          v['verdict'].startswith('NO-GO') and 'disagreement' in v['verdict'].lower(),
          f"got {v['verdict']}")
    check('T-11 disagreement leg recorded as fail',
          v['legs']['sampled_metric_agreement']['passed'] is False)


# ---------------------------------------------------------------- T-12
def test_t12():
    """T-12 (P-11): artist-disjoint split — no artist in both train/test,
    achieved test size reported."""
    require_module()
    recs = []
    for a in range(10):  # 10 artists x 15 tracks = 150 tracks
        for i in range(15):
            recs.append({'spotify_id': f'A{a}T{i:02d}', 'artist': f'Artist{a}'})
    train_ids, test_ids = ev3.artist_disjoint_split(recs, target_test_n=130)
    artists_train = {r['artist'] for r in recs if r['spotify_id'] in train_ids}
    artists_test = {r['artist'] for r in recs if r['spotify_id'] in test_ids}
    check('T-12 no artist in both train and test',
          artists_train.isdisjoint(artists_test))
    check('T-12 test size >= 130 (greedy whole-artist assignment)',
          len(test_ids) >= 130, f'got {len(test_ids)}')
    check('T-12 train+test covers all tracks',
          len(train_ids) + len(test_ids) == 150)


# ---------------------------------------------------------------- T-13
def test_t13():
    """T-13 (P-05): popularity audit unit — pool SD printed beside the
    15-point flag rule; no flag when the gap is small."""
    require_module()
    items = make_pool()
    pa = ev3.popularity_audit([dict(x, score=x['score_v3']) for x in items],
                              'score', higher_better=True)
    check('T-13 no bias flag on fixture (gap <= 15)',
          pa['popularity_bias'] is False, f"got {pa}")
    check('T-13 pool SD reported', pa['pool_popularity_sd'] > 0,
          f"got {pa}")
    check('T-13 threshold recorded as 15', pa['threshold'] == 15)
    # Verdict: full GO path — legs 1-3 pass, live 32/42 -> GO.
    v = ev3.verdict_4leg({'delta_ndcg_ci_low': 0.05, 'delta_ndcg': 0.10,
                          'delta_p10': 0.10, 'delta_map10': 0.03,
                          'p10_v3': 0.60, 'p10_v3_ci_low': 0.40,
                          'random_p10': 99 / 311, 'n_neg_top10': 0,
                          'n_neg_top10_H': 0, 'mean_rank_negative_v3': 200.0,
                          'random_mean_rank': 156.0,
                          'sampled_disagreement': False,
                          'live_keeps': 32, 'live_n': 42})
    check('T-13 legs 1-3 pass + 32/42 live -> GO',
          v['verdict'].startswith('GO'), f"got {v['verdict']}")
    # 31/42: binomial passes but Wilson fails -> NO-GO (marginal).
    v31 = ev3.verdict_4leg({'delta_ndcg_ci_low': 0.05, 'delta_ndcg': 0.10,
                            'delta_p10': 0.10, 'delta_map10': 0.03,
                            'p10_v3': 0.60, 'p10_v3_ci_low': 0.40,
                            'random_p10': 99 / 311, 'n_neg_top10': 0,
                            'n_neg_top10_H': 0, 'mean_rank_negative_v3': 200.0,
                            'random_mean_rank': 156.0,
                            'sampled_disagreement': False,
                            'live_keeps': 31, 'live_n': 42})
    check('T-13 31/42 -> NO-GO (marginal)',
          v31['verdict'].startswith('NO-GO (marginal)'),
          f"got {v31['verdict']}")
    # Legs 1-3 pass but live not yet judged -> pending, not GO.
    vpend = ev3.verdict_4leg({'delta_ndcg_ci_low': 0.05, 'delta_ndcg': 0.10,
                              'delta_p10': 0.10, 'delta_map10': 0.03,
                              'p10_v3': 0.60, 'p10_v3_ci_low': 0.40,
                              'random_p10': 99 / 311, 'n_neg_top10': 0,
                              'n_neg_top10_H': 0, 'mean_rank_negative_v3': 200.0,
                              'random_mean_rank': 156.0,
                              'sampled_disagreement': False,
                              'live_keeps': None, 'live_n': None})
    check('T-13 live leg unjudged -> pending, not GO',
          'pending' in vpend['verdict'].lower(), f"got {vpend['verdict']}")


TESTS = [test_t01, test_t02, test_t03, test_t04, test_t05, test_t06,
         test_t07, test_t08, test_t09, test_t10, test_t11, test_t12,
         test_t13]


def main():
    if ev3 is None:
        print(f'RED: evaluate_v3 import failed: {_IMPORT_ERROR}')
    for t in TESTS:
        try:
            t()
        except Exception as e:  # noqa: BLE001 - record the failure, keep going
            failures.append(t.__name__)
            print(f'FAIL  {t.__name__}  {type(e).__name__}: {e}')
    print(f'\n{len(passes)} passed, {len(failures)} failed: {failures}')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
