"""Directory search against real tiny-model embeddings and scalar cosine."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tiny_model import write_model


class SearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="clap-search-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.model = cls.root / "model.gguf"
        write_model(cls.model, waveform=True)
        cls.folder = cls.root / "audio collection"
        cls.folder.mkdir()
        (cls.folder / "nested").mkdir()
        fixtures = Path(sys.argv[2])
        for source, target in (("mono.wav", "a.WAV"), ("stereo.mp3", "nested/b.mp3"),
                               ("44100.flac", "nested/c.flac"), ("mono.wav", "z.wav"),
                               ("mono.flac", "nested/音乐.flac")):
            shutil.copyfile(fixtures / source, cls.folder / target)
        (cls.folder / "ignored.txt").write_text("not audio", encoding="utf-8")
        (cls.folder / "broken.wav").write_bytes(b"not audio")
        cls.paths = sorted(p for p in cls.folder.rglob("*")
                           if p.is_file() and p.name not in ("ignored.txt", "broken.wav"))

    def run_cli(self, *args):
        return subprocess.run([sys.argv[1], *map(str, args)], capture_output=True,
                              text=True, encoding="utf-8", timeout=60)

    def embedding(self, *query):
        result = self.run_cli("embedding", "-m", self.model, "-t", 1, "--device", "cpu", *query)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def search(self, *query, top_k=10):
        result = self.run_cli("search", self.folder, "-m", self.model, "-t", 1,
                              "--device", "cpu", "--top-k", top_k, *query)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("broken.wav", result.stderr)
        return json.loads(result.stdout)

    def check_ranking(self, query):
        vector = self.embedding(*query)
        expected = []
        for path in self.paths:
            candidate = self.embedding("--audio", path)
            score = sum(a*b for a, b in zip(vector, candidate)) / math.sqrt(
                sum(a*a for a in vector) * sum(b*b for b in candidate))
            expected.append({"path": str(path).replace("\\", "/"), "score": score})
        expected.sort(key=lambda row: (-row["score"], row["path"]))
        actual = self.search(*query)
        self.assertEqual([r["path"] for r in actual], [r["path"] for r in expected])
        for row, reference in zip(actual, expected):
            self.assertEqual(set(row), {"path", "score"})
            self.assertAlmostEqual(row["score"], reference["score"], delta=2e-5)
        self.assertEqual(self.search(*query, top_k=2), actual[:2])

    def test_text_search(self):
        self.check_ranking(("--text", "hello"))

    def test_audio_search(self):
        self.check_ranking(("--audio", self.paths[0]))
        self.assertAlmostEqual(self.search("--audio", self.paths[0])[0]["score"], 1, delta=2e-5)

    def test_empty_text_is_an_explicit_query(self):
        self.assertEqual(len(self.search("--text", "")), len(self.paths))

    def test_cli_validation(self):
        base = ("search", self.folder, "-m", self.model)
        for args in ((), ("--text", "x", "--audio", self.paths[0]),
                     ("--text", "x", "-k", 0), ("--text", "x", "-k", -1),
                     ("--text", "x", "-t", -1), ("--audio", self.root / "missing.wav")):
            with self.subTest(args=args):
                result = self.run_cli(*base, *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
        for directory in (self.root / "missing", self.model):
            result = self.run_cli("search", directory, "-m", self.model, "--text", "x")
            self.assertNotEqual(result.returncode, 0)

    def test_no_decodable_candidates(self):
        folder = self.root / "empty"
        folder.mkdir(exist_ok=True)
        for corrupt in (False, True):
            if corrupt:
                (folder / "broken.mp3").write_bytes(b"invalid")
            result = self.run_cli("search", folder, "-m", self.model, "--text", "hello", "-t", 1)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")

    def test_symlink_directories_are_not_followed(self):
        external = self.root / "external"
        external.mkdir(exist_ok=True)
        shutil.copyfile(self.paths[0], external / "outside.wav")
        link = self.folder / "linked"
        try:
            link.symlink_to(external, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(str(error))
        self.addCleanup(link.unlink)
        self.assertEqual(len(self.search("--text", "hello")), len(self.paths))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
