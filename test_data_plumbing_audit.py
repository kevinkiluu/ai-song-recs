"""Regressions for frozen-run inputs. No test contacts a service or changes a snapshot."""
import builtins
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import features_v2 as features
import fetch_genre_buckets as genres


class DataPlumbingTests(unittest.TestCase):
    def test_integration_verifies_this_checkout(self):
        script = Path(__file__).resolve().with_name("checks_v3_integration.py")
        paths = []
        open_file = builtins.open

        class SnapshotOpened(Exception):
            pass

        def observe(path, *args, **kwargs):
            if str(path).endswith("evaluation_v3.json"):
                paths.append(Path(path).resolve())
                raise SnapshotOpened
            return open_file(path, *args, **kwargs)

        # Simulate an old Muse workspace on the same machine as the checkout.
        with patch("os.path.expanduser", return_value="/unrelated/workspace"), \
                patch("sys.path", list(sys.path)), \
                patch("builtins.open", side_effect=observe), self.assertRaises(SnapshotOpened):
            runpy.run_path(str(script))
        self.assertEqual(paths, [script.with_name("evaluation_v3.json")])

    def test_stale_genre_mapping_cannot_replace_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(features, "BASE", tmp), \
                patch.object(features, "DATA", tmp), patch.object(features, "log"):
            mapping = {f"unrelated-{i}": "E" for i in range(635)}
            Path(tmp, "genre_buckets.json").write_text(json.dumps(mapping))
            snapshot = Path(tmp, "features_v2.json")
            snapshot.write_text("last complete snapshot")
            tracks = [{"spotify_id": str(i), "title": "a", "artist": "b", "position": i}
                      for i in range(650)]
            with patch.object(features, "load_liked", return_value=tracks), \
                    patch.object(features, "fetch_reccobeats", return_value={}), \
                    patch.object(features, "match_deezer", return_value={}), \
                    patch.object(features, "deep_features", return_value={}), \
                    self.assertRaisesRegex(SystemExit, "GENRE MAPPING"):
                features.main()
            self.assertEqual(snapshot.read_text(), "last complete snapshot")

    def test_incomplete_pagination_preserves_genre_mapping(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(genres, "DATA", tmp):
            snapshot = Path(tmp, "genre_buckets.json")
            snapshot.write_text('{"old":"E"}')
            page = {"items": [{"spotify_uri": "spotify:track:one"}], "next": "page-two"}
            with patch.object(genres, "PLAYLISTS", {"E": "playlist"}), \
                    patch.object(genres, "run_cli", side_effect=[page, RuntimeError("offline")]), \
                    self.assertRaisesRegex(RuntimeError, "offline"):
                genres.main()
            self.assertEqual(snapshot.read_text(), '{"old":"E"}')

    def test_duplicate_page_does_not_truncate_later_genre_members(self):
        first = {"items": [{"spotify_uri": "spotify:track:one"}], "next": "two"}
        middle = dict(first, next="three")
        last = {"items": [{"spotify_uri": "spotify:track:final"}]}
        with patch.object(genres, "run_cli", side_effect=[first, middle, last]):
            self.assertEqual(genres.collect_ids("playlist"), ["one", "final"])

    def test_transient_deezer_failure_is_retried_before_caching(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(features, "DATA", tmp), \
                patch.object(features, "PREV", tmp), patch.object(features, "log"), \
                patch.object(features.P, "log"), patch.object(features.time, "sleep"):
            payload = {"data": [{"id": 123, "title": "Song", "preview": "fixture"}]}
            with patch("urllib.request.urlopen", side_effect=[TimeoutError(),
                       io.BytesIO(json.dumps(payload).encode("utf-8"))]), \
                    patch.object(features, "download_preview", return_value=True):
                matches = features.match_deezer(
                    [{"spotify_id": "spotify-id", "title": "Song", "artist": "Artist"}], {})
            self.assertEqual(matches["spotify-id"]["deezer_id"], 123)

    def test_failed_preview_can_be_recovered_from_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(features, "DATA", tmp), \
                patch.object(features, "PREV", tmp), patch.object(features, "log"), \
                patch.object(features.time, "sleep"):
            # State left by the old downloader followed by one feature pass.
            Path(tmp, "deezer_matches.json").write_text(json.dumps({
                "sid": {"deezer_id": 123, "download_error": "preview download failed"}}))
            Path(tmp, "123.mp3").write_bytes(b"")
            Path(tmp, "librosa_deep.json").write_text('{"123":{"error":"analysis failed"}}')
            record = {"id": 123, "title": "Song", "preview": "https://example.invalid/audio"}
            responses = [io.BytesIO(json.dumps(record).encode()), io.BytesIO(b"preview")]
            with patch("urllib.request.urlopen", side_effect=responses), \
                    patch.object(features, "analyze_deep", return_value={"bpm": 91}):
                matches = features.match_deezer(
                    [{"spotify_id": "sid", "title": "Song", "artist": "Artist"}], {})
                recovered = features.deep_features(matches)
            self.assertEqual(recovered["123"], {"bpm": 91})
            self.assertNotIn("download_error", matches["sid"])

    def test_interrupted_preview_never_publishes_empty_audio(self):
        class BrokenRead(io.BytesIO):
            def read(self, *args):
                raise TimeoutError("incomplete response")

        with tempfile.TemporaryDirectory() as tmp, patch.object(features, "log"), \
                patch("urllib.request.urlopen", return_value=BrokenRead()):
            dest = Path(tmp, "123.mp3")
            ok = features.download_preview("https://example.invalid/audio", dest, attempts=1)
            self.assertFalse(ok)
            self.assertFalse(dest.exists())

    def test_replay_imports_code_from_reviewed_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "scoring.py").write_text('raise RuntimeError("stale checkout imported")')
            script = """
import os.path, sys
old_expanduser = os.path.expanduser
os.path.expanduser = lambda p: sys.argv[1] if p == '~/workspace/audio-pipeline' else old_expanduser(p)
import recommend_v2
"""
            result = subprocess.run([sys.executable, "-c", script, tmp],
                                    cwd=Path(__file__).resolve().parent,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_acceptance_names_missing_recommendation_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The recommendation JSON exists; the external liked/history files do not.
            root = Path(__file__).resolve().parent
            script = """
import os.path, runpy, sys
old_expanduser = os.path.expanduser
def expand(value):
    if value == '~/workspace/audio-pipeline':
        return sys.argv[1]
    if value.startswith('~/workspace/'):
        return os.path.join(sys.argv[2], value.rsplit('/', 1)[1])
    return old_expanduser(value)
os.path.expanduser = expand
runpy.run_path(os.path.join(sys.argv[1], 'test_pipeline_v2.py'))
"""
            result = subprocess.run([sys.executable, "-c", script, str(root), tmp],
                                    text=True, capture_output=True)
            self.assertIn(str(Path(tmp, "spotify-liked-songs.txt")), result.stdout)
            self.assertNotIn("recommendations_v2.json missing", result.stdout)
