#!/usr/bin/env python3
"""Acceptance tests for the Embeat port v3 (obra TDD: checks first, watch
fail, then the work that passes them).

Spec: ~/workspace/audio-pipeline/research/embeat-spec.md sections (b)-(d).
Tests T-01..T-16 are the spec's tests, written BEFORE embeat_port.py exists.
Root-level file (v2 convention), NOT tests/ — deviation from the spec,
recorded in the build report.

Run: ~/workspace/audio-env/bin/python test_embeat_port.py
Exit 0 = every check green. Exit 1 = at least one check red (listed).

RED state (2026-11-21): every test FAILs — embeat_port.py does not exist
yet (ModuleNotFoundError). A test that passes before implementation is
void and must be rewritten.

Fixed seeds (train likes, position >= 130, with librosa, spanning E/R/H;
IDs copied from features_v2.json, verified 2026-11-21):
  E: 1tO6hb9Ym7aTV8hJ31tLjr oops! - ILLIT (pos 332)
  E: 24Y63uAvj6Oc9jGuTlhlgf HOT - LE SSERAFIM (pos 357)
  R: 44r1T1DN4jHojCmYvYUs9H GREEN LIGHT - SXMPRA (pos 132)
  H: 5LrXxb87HY9uamlNiNugIb I Think We Should Talk - Savage Ga$p (pos 160)
  H: 0j2T0R9dR9qdJYsB7ciXhf Stronger - Kanye West (pos 226)
"""
import io, json, os, resource, sys, time
from contextlib import redirect_stdout

BASE = os.path.expanduser('~/workspace/audio-pipeline')
DATA = os.path.join(BASE, 'data')
LIKED_FILE = os.path.expanduser('~/workspace/spotify-liked-songs.txt')
HIST_FILE = os.path.expanduser('~/workspace/spotify-recommended-tracks.txt')

sys.path.insert(0, BASE)

failures = []
passes = []


def check(name, cond, detail=''):
    if cond:
        passes.append(name)
        print(f'PASS  {name}')
    else:
        failures.append(name)
        print(f'FAIL  {name}  {detail}')


FIVE_SEEDS = [
    ('1tO6hb9Ym7aTV8hJ31tLjr', 'oops!', 'ILLIT', 'E'),
    ('24Y63uAvj6Oc9jGuTlhlgf', 'HOT', 'LE SSERAFIM', 'E'),
    ('44r1T1DN4jHojCmYvYUs9H', 'GREEN LIGHT', 'SXMPRA', 'R'),
    ('5LrXxb87HY9uamlNiNugIb', 'I Think We Should Talk', 'Savage Ga$p', 'H'),
    ('0j2T0R9dR9qdJYsB7ciXhf', 'Stronger', 'Kanye West', 'H'),
]
# 20 fixed seeds for caps / determinism / timing (train likes with
# librosa, same-artist pool tracks present, by list position).
TWENTY_SEEDS = [
    '2ngaTpD87A9EL3TAMdTA9V', '44r1T1DN4jHojCmYvYUs9H',
    '3CNBmNfMJmqWMCbXO7dYH9', '77vcz54ZNapi1F9ea84rLs',
    '4NardHbUoTLDH54bOBJ9T8', '2gYj9lubBorOPIVWsTXugG',
    '5VtwUrgljtOyBa773AGSr4', '0Q5VnK2DYzRyfqQRJuUtvi',
    '5LrXxb87HY9uamlNiNugIb', '1Y4uBvlovpqcp3HPSqq5pr',
    '6LCqUn7jGSOBkBcRC3YMEa', '3T4N6wohtfOJN4okuvHiWT',
    '7GX5flRQZVHRAGd6B4TmDO', '3b3e8SeB4jzR0ZlNDwWxMK',
    '6S8oLA2FJ5AnR6XXMcOyZh', '2ePfpPUTnniAYFFtZMT81e',
    '0pLW5T2sp2N4YEEHtF7ilb', '6G0s3kdSxsawbQwnfE2jPg',
    '0j2T0R9dR9qdJYsB7ciXhf', '1VN3d3kSjNrhEx47IFZKKq',
]
NEGATIVE_IDS = {
    '6aJn7Cst74cj4lNIiPRgav', '6HacgXCExkzS552ILfJTXu',
    '1GfBLbAhZUWdseuDqhocmn', '5v7oj9PosKdE39x9uwDs6C',
    '4bkPAARkDov1lV05emEF7a', '2jv1ovpDpTJEPIlth5Wn07',
    '7zDu3l7DDlfSlt20YMKkWo', '2y57H6nA3vic5t8awwaS69',
    '2slLXFcITuhf0oSp2wVJLv', '30efCAopZCFcoi43sSToNr',
    '0FmrNFYyLLRtIS7XWJPLLk', '32gSPav2xqFylAeCjJCxWi',
    '1vvBUZseJ7fwYPX1NedOLd',
}


def id_set(path):
    ids = set()
    for line in open(path):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3 and not line.startswith('#'):
            ids.add(p[2].replace('spotify:track:', ''))
    return ids


def key_set(path):
    import pipeline as P
    keys = set()
    for line in open(path):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 2 and not line.startswith('#'):
            keys.add((P.clean(p[0]), P.clean(p[1].split(',')[0])))
    return keys


try:
    import embeat_port as EP
except ImportError as e:
    for t in [f'T-{i:02d}' for i in range(1, 17)]:
        check(f'{t} (RED: embeat_port missing)', False, str(e))
    print(f'\n{len(passes)} passed, {len(failures)} failed')
    print('FAILING:', ', '.join(failures))
    sys.exit(1)

# ---------------------------------------------------------------- T-01
_needle = 'qdrant' + '_client'
_hits = []
for _f in os.listdir(BASE):
    if _f.endswith('.py'):
        _txt = open(os.path.join(BASE, _f)).read()
        if _needle in _txt:
            _hits.append(_f)
check('T-01 Qdrant absent: no client import in *.py', not _hits,
      f'hits={_hits}')
check('T-01 Qdrant absent: not imported', _needle not in sys.modules)

# ---------------------------------------------------------------- T-02
try:
    _liked = EP.load_liked()
    _model = EP.get_model()
    _emb = EP.embed_records(_model, _liked)
    import numpy as np
    _norms = np.linalg.norm(_emb, axis=1)
    check('T-02 embedding dim 15', _emb.shape == (len(_liked), 15),
          f'shape={_emb.shape}')
    check('T-02 embedding L2 norm 1.0 +-1e-6',
          bool(np.all(np.abs(_norms - 1.0) <= 1e-6)),
          f'max dev={float(np.max(np.abs(_norms - 1.0))):.2e}')
    _self = float(_emb[0] @ _emb[0])
    check('T-02 cosine(seed, seed) = 1.0 +-1e-9', abs(_self - 1.0) <= 1e-9,
          f'got {_self}')
except Exception as e:
    check('T-02 embedding contract', False, repr(e))

# ---------------------------------------------------------------- T-03
try:
    import numpy as np
    from collections import defaultdict
    _tr = [t for t in _liked if t['spotify_id'] != 'None']
    _E = EP.embed_records(_model, _tr)
    _art = [EP.normalize_artist(t.get('artist') or '') for t in _tr]
    _bk = [t.get('genre_bucket') for t in _tr]
    _rng = np.random.RandomState(616)
    _groups = defaultdict(list)
    for _i, _a in enumerate(_art):
        if _a:
            _groups[_a].append(_i)
    _keys = [k for k, v in _groups.items() if len(v) >= 2]

    def _cos(i, j):
        return float(_E[i] @ _E[j])

    _sa = []
    while len(_sa) < 200:
        _k = _keys[_rng.randint(0, len(_keys))]
        _i, _j = _rng.choice(_groups[_k], 2, replace=False)
        _sa.append(_cos(_i, _j))
    _rp = []
    while len(_rp) < 200:
        _i, _j = _rng.randint(0, len(_tr), 2)
        if _i != _j:
            _rp.append(_cos(_i, _j))
    _sb, _cb = [], []
    while len(_sb) < 200 or len(_cb) < 200:
        _i, _j = _rng.randint(0, len(_tr), 2)
        if _i == _j or not _bk[_i] or not _bk[_j]:
            continue
        if _bk[_i] == _bk[_j]:
            if len(_sb) < 200:
                _sb.append(_cos(_i, _j))
        elif len(_cb) < 200:
            _cb.append(_cos(_i, _j))
    _gap_a = float(np.mean(_sa)) - float(np.mean(_rp))
    _gap_b = float(np.mean(_sb)) - float(np.mean(_cb))
    check('T-03 same-artist cosine beats random by >= 0.10',
          _gap_a >= 0.10,
          f'same-artist={np.mean(_sa):.3f} random={np.mean(_rp):.3f} '
          f'gap={_gap_a:.3f}')
    check('T-03 same-bucket cosine beats cross-bucket by >= 0.05',
          _gap_b >= 0.05,
          f'same-bucket={np.mean(_sb):.3f} cross={np.mean(_cb):.3f} '
          f'gap={_gap_b:.3f}')
except Exception as e:
    check('T-03 pair-rule sanity', False, repr(e))

# ---------------------------------------------------------------- shared index
try:
    _pool = EP.load_pool()
    _index = EP.build_index(_pool, os.path.join(BASE, 'model_v2.npz'))
    _by_id = {t['spotify_id']: t for t in _liked}
except Exception as e:
    check('shared index build', False, repr(e))
    _index = None

# ---------------------------------------------------------------- T-04
try:
    _ok = True
    _detail = ''
    for _sid, _title, _artist, _bucket in FIVE_SEEDS:
        _seed = _by_id[_sid]
        _cands = EP.recall_similar(_index, _seed, top_k=10)
        _voted, _src = EP.vote_track_bucket(_index, _seed)
        if len(_cands) > 100:
            _ok = False
            _detail += f'{_sid}: {len(_cands)} cands; '
        if any(EP.cand_key(c) == EP.cand_key(_seed) for c in _cands):
            _ok = False
            _detail += f'{_sid}: seed present; '
        if _src == 'vote' and any(
                (c.get('bucket') or 'O') == 'O' for c in _cands):
            _ok = False
            _detail += f'{_sid}: O-bucket candidate present; '
    # synthetic: an O-bucket candidate must be excluded by the ch-1 filter
    # when the seed bucket is voted (O is unreachable in the real pool
    # because the via_artist fallback assigns every pool track a bucket,
    # so the real-pool arm above is vacuous on O and this arm carries it).
    _fake = {'title': 'Synthetic Unknown', 'artist': 'Nobody',
             'bucket': 'O', 'isrc': None, 'deezer_id': -1,
             'spotify_id': None}
    _seed0 = _by_id[FIVE_SEEDS[0][0]]
    _synth_ok = not EP.passes_channel1_filter(
        _fake, _seed0, seed_bucket='E', bucket_voted=True)
    check('T-04 channel-1 limits/filter (<=100, no seed, no O when voted)',
          _ok and _synth_ok, _detail or 'synthetic O not excluded')
except Exception as e:
    check('T-04 channel-1 limits/filter', False, repr(e))

# ---------------------------------------------------------------- T-05
try:
    import numpy as np
    _ok, _detail = True, ''
    for _sid, _title, _artist, _bucket in FIVE_SEEDS:
        _seed = _by_id[_sid]
        _cands = EP.recall_popular(_index, _seed, top_k=10)
        _voted, _src = EP.vote_track_bucket(_index, _seed)
        if len(_cands) > 20:
            _ok = False
            _detail += f'{_sid}: {len(_cands)}; '
        if any((c.get('bucket') or 'O') != _voted for c in _cands):
            _ok = False
            _detail += f'{_sid}: off-bucket candidate; '
        if not _cands:
            _ok = False
            _detail += f'{_sid}: empty; '
            continue
        _pre = sorted(_cands, key=lambda c: c['popular_rank'])
        _prox = [_c['popularity_proxy'] for _c in _pre]
        if _prox != sorted(_prox, reverse=True):
            _ok = False
            _detail += f'{_sid}: pre-rerank not proxy-DESC; '
        _best_cos = max(_c['cosine'] for _c in _cands)
        if abs(_cands[0]['cosine'] - _best_cos) > 1e-9:
            _ok = False
            _detail += f'{_sid}: top is not max-cosine; '
    check('T-05 channel-2 mechanism (<=20, in-bucket, proxy order, '
          'cosine re-rank top)', _ok, _detail)
except Exception as e:
    check('T-05 channel-2 mechanism', False, repr(e))

# ---------------------------------------------------------------- T-06
try:
    _ok, _detail = True, ''
    for _sid, _title, _artist, _bucket in FIVE_SEEDS:
        _seed = _by_id[_sid]
        _cands = EP.recall_same_artist(_index, _seed, top_k=10)
        _want = EP.normalize_artist(_seed.get('artist') or '')
        if len(_cands) > 20:
            _ok = False
            _detail += f'{_sid}: {len(_cands)}; '
        if any(EP.normalize_artist(c.get('artist') or '') != _want
               for c in _cands):
            _ok = False
            _detail += f'{_sid}: wrong artist; '
        _prox = [_c['popularity_proxy'] for _c in _cands]
        if _prox != sorted(_prox, reverse=True):
            _ok = False
            _detail += f'{_sid}: not proxy-DESC; '
    check('T-06 channel-3 mechanism (same artist only, <=20, proxy DESC)',
          _ok, _detail)
except Exception as e:
    check('T-06 channel-3 mechanism', False, repr(e))

# ---------------------------------------------------------------- T-07
try:
    _ok, _detail = True, ''
    for _sid, _title, _artist, _bucket in FIVE_SEEDS:
        _seed = _by_id[_sid]
        _cands = EP.recall_related_artist(_index, _seed, top_k=10)
        _rel = EP.related_artist_set(_index, _seed)
        if len(_cands) > 20:
            _ok = False
            _detail += f'{_sid}: {len(_cands)}; '
        if any(EP.normalize_artist(c.get('artist') or '') not in _rel
               for c in _cands):
            _ok = False
            _detail += f'{_sid}: artist not in related set; '
        if not _rel and _cands:
            _ok = False
            _detail += f'{_sid}: non-empty with empty related set; '
    # seed whose artist has no related artists in the pool -> []
    _lonely = None
    for _t in _liked:
        if (_t.get('librosa') and _t['position'] >= 130
                and not EP.related_artist_set(_index, _t)):
            _lonely = _t
            break
    _lonely_ok = (_lonely is not None and
                  EP.recall_related_artist(_index, _lonely, top_k=10) == [])
    check('T-07 channel-4 mechanism (related artists only, <=20, [] when '
          'none)', _ok and _lonely_ok,
          _detail or ('no lonely seed found' if _lonely is None else ''))
except Exception as e:
    check('T-07 channel-4 mechanism', False, repr(e))

# ---------------------------------------------------------------- T-08
try:
    def _mk(key, title=None):
        return {'key': key, 'title': title or key, 'artist': 'A' + key,
                'isrc': None, 'deezer_id': None, 'spotify_id': None,
                'bucket': 'E', 'popularity_proxy': 0.5}
    _x = _mk('X')
    _y = _mk('Y')
    _z = _mk('Z')
    _lists = {'similar': [_x, _y, _z],
              'same_artist': [_x]}
    _merged = EP.merge_by_score(_lists, top_k=10, random_seed=616,
                                seed_artist='ZZZ')
    # merge keys are cand_key tuples: ('ta', title, artist) (no ISRC here)
    _byk = {m['key']: m for m in _merged}
    _kx, _ky, _kz = ('ta', 'x', 'ax'), ('ta', 'y', 'ay'), ('ta', 'z', 'az')
    # Linear remap, similar [1.7, 1.0] over 3 items: rank0=1.7, rank1=1.35,
    # rank2=1.0. Spec T-08 as written asserts Y=1.7, which contradicts its
    # own remap (Y is rank 1 of 3, not rank 0): the spec's number is wrong
    # for the list shape it implies; the correct remap value is 1.35.
    # The "rank-0 similar-only scores 1.7" claim is covered by case B.
    _ok = (abs(_byk[_kx]['score_raw'] - 3.6) < 1e-9
           and abs(_byk[_ky]['score_raw'] - 1.35) < 1e-9
           and abs(_byk[_kz]['score_raw'] - 1.0) < 1e-9
           and _merged[0]['key'] == _kx)
    _only = EP.merge_by_score({'similar': [_mk('Y')]}, top_k=10,
                              random_seed=616, seed_artist='ZZZ')
    _ok = _ok and abs(_only[0]['score_raw'] - 1.7) < 1e-9
    check('T-08 merge arithmetic (X=3.6, Y=1.35, last similar=1.0, X '
          'first; rank-0 similar-only=1.7)',
          _ok, f"got {[(m['key'], m['score_raw']) for m in _merged]}")
except Exception as e:
    check('T-08 merge arithmetic', False, repr(e))

# ---------------------------------------------------------------- T-09
try:
    _ok, _detail = True, ''
    for _sid in TWENTY_SEEDS:
        _rec = EP.recommend(_sid, top_k=10, random_seed=616)
        _seed = _by_id[_sid]
        _sa = EP.normalize_artist(_seed.get('artist') or '')
        _n_same = sum(1 for r in _rec
                      if EP.normalize_artist(r.get('artist') or '') == _sa)
        if len(_rec) != 10:
            _ok = False
            _detail += f'{_sid}: len={len(_rec)}; '
        if _n_same > 2:
            _ok = False
            _detail += f'{_sid}: same-artist={_n_same}; '
        from collections import Counter as _C
        _cnt = _C(EP.normalize_artist(r.get('artist') or '') for r in _rec)
        _bad = [a for a, n in _cnt.items() if a != _sa and n > 1]
        if _bad:
            _ok = False
            _detail += f'{_sid}: repeat non-seed artists {_bad}; '
        _ch2 = EP.recall_popular(_index, _seed, top_k=10)
        if _ch2 and not any('popular' in r.get('sources', [])
                            for r in _rec):
            _ok = False
            _detail += f'{_sid}: no popular-sourced track; '
    check('T-09 caps (<=2 same-artist, <=1 other artist, popular floor)',
          _ok, _detail)
except Exception as e:
    check('T-09 caps', False, repr(e))

# ---------------------------------------------------------------- T-10
try:
    import pipeline as P
    _pool_remix = [p for p in _pool
                   if 'remix' in (p.get('title') or '').lower()]
    _ok = len(_pool_remix) >= 1
    _detail = f'{len(_pool_remix)} pool remixes'
    # a non-seed remix must pass the common filters for a seed with a
    # different title stem
    _seed = _by_id[FIVE_SEEDS[0][0]]
    _eligible = [p for p in _pool_remix
                 if EP.passes_filters(p, _seed, seed_proxy=0.0)]
    if not _eligible:
        _ok = False
        _detail += '; no pool remix eligible'
    # a remix OF the seed itself (same stem + same artist) never passes
    _fake_remix = {'title': _seed['title'] + ' (Somebody Remix)',
                   'artist': _seed['artist'], 'isrc': None,
                   'deezer_id': -2, 'spotify_id': None, 'bucket': 'E'}
    if EP.passes_filters(_fake_remix, _seed, seed_proxy=0.0):
        _ok = False
        _detail += '; seed remix passed filters'
    check('T-10 remix correction (non-seed remix eligible, seed remix '
          'excluded)', _ok, _detail)
except Exception as e:
    check('T-10 remix correction', False, repr(e))

# ---------------------------------------------------------------- T-11
try:
    _buf = io.StringIO()
    _seed = _by_id[FIVE_SEEDS[0][0]]
    with redirect_stdout(_buf):
        _out = EP.recall_related_track_stub(_index, _seed, top_k=10)
    _logged = ('channel5=stub:no-playlist-corpus' in _buf.getvalue()
               or 'channel5=stub:no-playlist-corpus' in EP.CALL_LOG)
    check('T-11 channel-5 stub returns [] and logs',
          _out == [] and _logged, f'out={_out} logged={_logged}')
    _rec = EP.recommend(FIVE_SEEDS[0][0], top_k=10, random_seed=616)
    check('T-11 merge still returns 10 with stubbed channel 5',
          len(_rec) == 10, f'len={len(_rec)}')
except Exception as e:
    check('T-11 channel-5 stub', False, repr(e))

# ---------------------------------------------------------------- T-12
try:
    _ranked = EP.rank_eval_pool()
    _rank_of = {}
    for _i, _it in enumerate(_ranked, 1):
        _rank_of[_it['rec'].get('spotify_id')] = _i
    _in_pool = [s for s in NEGATIVE_IDS if s in _rank_of]
    _worst = max((_rank_of[s] for s in _in_pool), default=0)
    # 13th negative (Touch, no librosa) is excluded from the 311 pool by
    # evaluate.py itself (negatives_with_librosa=12); assert that, not
    # a rank for it.
    _neg_lb = [r for r in EP.load_negatives() if r.get('librosa')]
    _ok = (_worst > 10 and len(_in_pool) == 12
           and len(_neg_lb) == 12)
    _detail = f'in-pool negatives={len(_in_pool)} worst rank={_worst}'
    import pipeline as P
    _neg_keys = {(P.clean(r['title']), P.clean((r['artist'] or '').split(',')[0]))
                 for r in EP.load_negatives()}
    for _sid, *_ in FIVE_SEEDS:
        _rec = EP.recommend(_sid, top_k=10, random_seed=616)
        _keys = {(P.clean(r['title']), P.clean((r['artist'] or '').split(',')[0]))
                 for r in _rec}
        if _keys & _neg_keys:
            _ok = False
            _detail += f'; negative in top-10 for {_sid}'
    check('T-12 negative control (12 in-pool negatives rank >10; none in '
          'any 5-seed top-10)', _ok, _detail)
except Exception as e:
    check('T-12 negative control', False, repr(e))

# ---------------------------------------------------------------- T-13
try:
    _a = EP.recommend(FIVE_SEEDS[0][0], top_k=10, random_seed=616)
    _b = EP.recommend(FIVE_SEEDS[0][0], top_k=10, random_seed=616)
    _same = ([(r.get('deezer_id'), r.get('title'), r.get('score'))
              for r in _a]
             == [(r.get('deezer_id'), r.get('title'), r.get('score'))
                 for r in _b])
    check('T-13 determinism (byte-identical top-10, seed 616)', _same)
except Exception as e:
    check('T-13 determinism', False, repr(e))

# ---------------------------------------------------------------- T-14
try:
    _liked_ids = id_set(LIKED_FILE)
    _hist_ids = id_set(HIST_FILE)
    _liked_keys = key_set(LIKED_FILE)
    _hist_keys = key_set(HIST_FILE)
    import pipeline as P
    _ok, _detail = True, ''
    for _sid, *_ in FIVE_SEEDS:
        _rec = EP.recommend(_sid, top_k=10, random_seed=616)
        for _r in _rec:
            _rid = _r.get('spotify_id')
            if _rid and _rid != 'None' and (_rid in _liked_ids
                                            or _rid in _hist_ids):
                _ok = False
                _detail += f'{_sid}: id overlap {_rid}; '
            _k = (P.clean(_r['title']),
                  P.clean((_r['artist'] or '').split(',')[0]))
            if _k in _liked_keys or _k in _hist_keys:
                _ok = False
                _detail += f'{_sid}: key overlap {_k}; '
    check('T-14 exclusion (top-10 disjoint from Liked Songs + 60 history)',
          _ok, _detail)
except Exception as e:
    check('T-14 exclusion', False, repr(e))

# ---------------------------------------------------------------- T-15
try:
    _times = []
    for _sid in TWENTY_SEEDS:
        _t0 = time.perf_counter()
        EP.recommend(_sid, top_k=10, random_seed=616)
        _times.append(time.perf_counter() - _t0)
    _times.sort()
    _med = _times[len(_times) // 2]
    _rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    check('T-15 budget (median recommend <=500ms, peak RSS <=2GB)',
          _med <= 0.5 and _rss_mb <= 2048,
          f'median={_med * 1000:.0f}ms peak RSS={_rss_mb:.0f}MB')
except Exception as e:
    check('T-15 budget', False, repr(e))

# ---------------------------------------------------------------- T-16
try:
    _res = EP.evaluate_port(write=True)
    _ev = json.load(open(os.path.join(BASE, 'evaluation_v3_embeat.json')))
    _p = _ev['ported_medoid']
    _best_p = max(_ev['baselines']['v2']['P@10'],
                  _ev['baselines']['v6_per_bucket']['P@10'],
                  _ev['baselines']['v6_global']['P@10'])
    _best_n = max(_ev['baselines']['v2']['NDCG@10'],
                  _ev['baselines']['v6_per_bucket']['NDCG@10'],
                  _ev['baselines']['v6_global']['NDCG@10'])
    _best_neg_rank = _ev['baselines']['v6_global']['mean_rank_negative']
    _ok = (_p['P@10'] >= _best_p + 0.05
           and _p['NDCG@10'] >= _best_n + 0.05
           and _p['neg_top10'] == 0
           and _p['mean_rank_negative'] >= _best_neg_rank + 20)
    check('T-16 eval beats baseline (P+0.05, NDCG+0.05, neg_top10=0, '
          'neg rank +20 vs best v6)', _ok,
          f"port P@10={_p['P@10']} (best {_best_p}) NDCG={_p['NDCG@10']} "
          f"(best {_best_n}) neg_top10={_p['neg_top10']} "
          f"neg rank={_p['mean_rank_negative']} (v6g {_best_neg_rank})")
except Exception as e:
    check('T-16 eval beats baseline', False, repr(e))

print(f'\n{len(passes)} passed, {len(failures)} failed')
if failures:
    print('FAILING:', ', '.join(failures))
    sys.exit(1)
print('ALL ACCEPTANCE CHECKS GREEN')
