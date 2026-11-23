"""Replay recommend() from a private snapshot with no live lookups or playlist writes."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


REQUIRED = ('model_v2.npz', 'model_v2.json', 'features_v2.json', 'data/pool_features.json',
            'spotify-liked-songs.txt', 'spotify-recommended-tracks.txt', 'spotify_search_responses.json',
            'recommendations_v10.json')


def load(path):
    return json.loads(path.read_text())


def replay(snapshot):
    missing = [name for name in REQUIRED if not (snapshot / name).is_file()]
    if missing:
        raise ValueError('missing snapshot inputs: ' + ', '.join(missing))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import recommend_v2 as R
    import scoring

    responses = {}
    for row in load(snapshot / 'spotify_search_responses.json'):
        key = row['title'], row['artist']
        if key in responses or not isinstance(row['response'], (dict, list)):
            raise ValueError('ambiguous or invalid archived Spotify search')
        responses[key] = row['response']

    def search(title, artist):
        if (title, artist) not in responses:
            raise ValueError('a required Spotify search is missing from the snapshot')
        return R.pick_search_hit(title, artist, responses[title, artist])

    def fill(ids):
        # A present null documents a completed lookup with no features.
        cache_path = snapshot / 'data' / 'reccobeats.json'
        if not cache_path.is_file():
            raise ValueError('a ReccoBeats fill is required, but data/reccobeats.json is missing')
        cache = load(cache_path)
        if any(sid not in cache for sid in ids):
            raise ValueError('a required ReccoBeats lookup is missing from the snapshot')
        return {sid: cache[sid] for sid in ids}

    expected = load(snapshot / 'recommendations_v10.json')
    if not isinstance(expected, list) or len(expected) != 10 or not all(
            isinstance(r, dict) and r.get('spotify_id') for r in expected):
        raise ValueError('expected the original ten v10 output rows with Spotify IDs')
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copyfile(snapshot / 'features_v2.json', work / 'features_v2.json')
        with (patch.multiple(R, BASE=work, DATA=snapshot / 'data',
                             LIKED_FILE=snapshot / 'spotify-liked-songs.txt',
                             HIST_FILE=snapshot / 'spotify-recommended-tracks.txt', spotify_search_id=search),
              patch.object(scoring, 'BASE', snapshot), patch.object(R.F, 'fetch_reccobeats', fill),
              patch.object(R.time, 'sleep'), contextlib.redirect_stdout(io.StringIO())):
            actual = R.recommend()
        if actual != load(work / 'recommendations_v2.json'):
            raise ValueError('returned rows differ from the written output')
    old_ids = [r['spotify_id'] for r in expected]
    new_ids = [r['spotify_id'] for r in actual]
    return {'expected': len(expected), 'generated': len(actual),
            'common_ids': len(set(old_ids) & set(new_ids)),
            'same_positions': sum(a == b for a, b in zip(old_ids, new_ids)),
            'ordered_ids_identical': old_ids == new_ids, 'output_identical': actual == expected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    try:
        result = replay(args.snapshot.resolve())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f'incomplete replay: {exc}\n')
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['output_identical'] else 1)
