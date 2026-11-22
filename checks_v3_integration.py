# Acceptance checks for the v3 integration (TDD: written BEFORE integrate_v3.py
# exists, must FAIL now and pass only after the integration runs).
#
# Each check names the wrong result it catches. Expected values are derived
# independently: pool counts from the decontamination definition (311 - 5 dup
# audio = 306; 99 - 5 = 94 held-out), baselines from the coordinator-verified
# Troi JSON (recomputed there, not from my integration), random baselines
# from arithmetic (94/306, (306+1)/2), and verdicts re-derived from the
# per-item score audit file via evaluate_v3 functions (a second derivation
# path, not a copy of evaluation_v3.json).
import json, math, os, subprocess, sys

BASE = os.path.expanduser('~/workspace/audio-pipeline')
sys.path.insert(0, BASE)
import evaluate_v3 as ev3

FAILURES = []

def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), '-', name, detail)
    if not cond:
        FAILURES.append(name)

# --- C1: decontaminated pool counts (catches: wrong pool, dup items left in)
ev = json.load(open(os.path.join(BASE, 'evaluation_v3.json')))
pool = ev['pool']
check('C1 pool n_ranked == 306', pool.get('n_ranked') == 306, str(pool))
check('C1 heldout == 94', pool.get('heldout_with_librosa') == 94)
check('C1 negatives ranked == 12', pool.get('negatives_with_librosa') == 12)
check('C1 distractors == 200', pool.get('distractors') == 200)
removed = {r['title'] for r in pool.get('removed_duplicate_audio', [])}
check('C1 removed 5 dup-audio titles',
      removed == {'Warriors', 'CRAZY', 'Red Rum', '[Gen]inside', 'Partyy'},
      str(removed))
check('C1 random baselines arithmetic',
      abs(pool.get('random_P@10', -1) - 94/306) < 0.001
      and abs(pool.get('random_mean_rank', -1) - 153.5) < 0.01,
      f"random_P@10={pool.get('random_P@10')} random_mean_rank={pool.get('random_mean_rank')}")

# --- C2: baselines recomputed on the 306 pool match Troi's verified numbers
# (catches: pool drift, wrong distractor draw, score-column mixup)
base = ev['baselines']
check('C2 v2 P@10 0.50 / NDCG 0.726', base['v2']['P@10'] == 0.5
      and base['v2']['NDCG@10'] == 0.726, str(base['v2']))
check('C2 v6b P@10 0.40 / NDCG 0.654', base['v6_per_bucket']['P@10'] == 0.4
      and base['v6_per_bucket']['NDCG@10'] == 0.654, str(base['v6_per_bucket']))
check('C2 v6g P@10 0.60 / NDCG 0.660', base['v6_global']['P@10'] == 0.6
      and base['v6_global']['NDCG@10'] == 0.66, str(base['v6_global']))

# --- C3: every candidate metric re-derived from the per-item audit file
# (catches: copied Wave-1 numbers, 311-pool numbers passed off as 306)
audit = json.load(open(os.path.join(BASE, 'v3_scores_306.json')))
items = audit['items']
def ev3_items(score_key, higher):
    return [{'spotify_id': x['spotify_id'], 'kind': x['kind'],
             'relevant': x['relevant'], 'artist': x.get('artist', ''),
             'bucket': x.get('bucket'), score_key: x['scores'].get(score_key)}
            for x in items]
for cand in ('troi', 'embeat_full', 'embeat_minus_ch1', 'bliss_fixed',
             'bliss_adaptive', 'bliss_v2_blend', 'v2_unchanged'):
    rec = ev['candidates'][cand]
    higher = rec['higher_better']
    m = ev3.rank_metrics(ev3_items(rec['score_key'], higher),
                         rec['score_key'], higher_better=higher)
    check(f'C3 {cand} P@10 re-derived', m['P@10'] == rec['metrics']['P@10'],
          f"audit={m['P@10']} json={rec['metrics']['P@10']}")
    check(f'C3 {cand} NDCG re-derived', m['NDCG@10'] == rec['metrics']['NDCG@10'],
          f"audit={m['NDCG@10']} json={rec['metrics']['NDCG@10']}")
    check(f'C3 {cand} neg_top10 re-derived',
          m['n_neg_top10'] == rec['metrics']['n_neg_top10'],
          f"audit={m['n_neg_top10']} json={rec['metrics']['n_neg_top10']}")

# --- C4: every 4-leg verdict re-derived via verdict_4leg, not hand-computed
# (catches: hand-computed verdict, wrong comparator, stale random baseline)
for cand, rec in ev['candidates'].items():
    v = ev3.verdict_4leg(rec['verdict_inputs'])
    check(f'C4 {cand} verdict matches verdict_4leg',
          v['verdict'] == rec['verdict'],
          f"fn={v['verdict']} json={rec['verdict']}")
    check(f'C4 {cand} offline_pass consistent',
          v['offline_pass'] == rec['offline_pass'])

# --- C5: sequencing — no recommendations file when no candidate passes
# Legs 1-3 (catches: spending a live list after an offline NO-GO)
any_offline_pass = any(c['offline_pass'] for c in ev['candidates'].values())
recs_exists = os.path.exists(os.path.join(BASE, 'recommendations_v3.json'))
check('C5 no recs file if no offline pass',
      (not any_offline_pass and not recs_exists)
      or (any_offline_pass and recs_exists),
      f'any_offline_pass={any_offline_pass} recs_exists={recs_exists}')
check('C5 verdict is NO-GO offline when nothing passes',
      (not any_offline_pass and 'NO-GO' in ev['verdict'])
      or any_offline_pass,
      ev['verdict'])

# --- C6: report exists and names the verdict + NDCG caveat + twin-leak caveat
rep = open(os.path.join(BASE, 'evaluation_report_v3.md')).read()
check('C6 report states NO-GO/GO verdict', ev['verdict'].split(' — ')[0] in rep)
check('C6 report states NDCG-convention caveat',
      'v2 convention' in rep or 'NDCG' in rep and '0.63' in rep)
check('C6 report states twin-leakage-in-training caveat',
      'twin' in rep.lower() or 'duplicate-audio' in rep.lower())

print()
if FAILURES:
    print(f'{len(FAILURES)} CHECK(S) FAILED: {FAILURES}')
    sys.exit(1)
print('ALL CHECKS PASSED')
