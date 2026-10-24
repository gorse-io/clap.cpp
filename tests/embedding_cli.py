"""Real embedding CLI contract; optional second argument is a parity fixture root.

Usage: embedding_cli.py CLAP [BUILD_WITH_MODEL_FIXTURES]
The default suite generates a tiny valid GGUF with a scalar text oracle using
only the standard library. Optional existing model-fixture, clamp3-fixture and
mert2-fixture oracles additionally check native text/audio/frame representations.
"""
import errno
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

BINARY = str(Path(sys.argv[1]).resolve())
FIXTURES = Path(sys.argv[2]) if len(sys.argv) > 2 else None


class EmbeddingCLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tiny_model import write_model
        cls.temporary = tempfile.TemporaryDirectory(prefix="clap embedding ")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.tiny_path = Path(cls.temporary.name) / "model.gguf"
        cls.tiny_reference = write_model(cls.tiny_path)

    def run_cli(self, *args, command="embedding", **kwargs):
        return subprocess.run([BINARY, command, *map(str, args)],
                              capture_output=True, text=True, timeout=120, **kwargs)

    def success(self, *args):
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def model_args(self, kind="model"):
        if FIXTURES is None:
            if kind == "model":
                return ["--model", str(self.tiny_path), "--threads", "2"]
            self.skipTest("pass a build directory containing parity fixtures for native encoder checks")
        assert FIXTURES is not None
        path = FIXTURES / (kind + "-fixture") / "model.gguf"
        self.assertTrue(path.is_file(), f"missing fixture: {path}")
        return ["--model", str(path), "--threads", "2"]

    def vector(self, text, kind="model"):
        return json.loads(self.success(*self.model_args(kind), "--text", text))

    def assert_vector_close(self, actual, expected, tolerance=3e-5):
        self.assertEqual(len(actual), len(expected))
        self.assertTrue(all(isinstance(x, (int, float)) and math.isfinite(x) for x in actual))
        self.assertLessEqual(max(abs(a-b) for a, b in zip(actual, expected)), tolerance)

    def test_help(self):
        output = self.success("--help")
        for option in ("--model", "--threads", "--prompt", "--text", "--audio", "--file",
                       "--crop-offset", "--frames", "--clamp3-features",
                       "--embd-separator", "--embd-output-format"):
            self.assertIn(option, output)
        self.assertNotIn("dummy", output.lower())
        result = self.run_cli("--help", command="serve")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--model", "--threads", "--host", "--port"):
            self.assertIn(option, result.stdout)

    def test_model_required_for_embedding_and_serve(self):
        for command, args in (("embedding", ["--prompt", "hello"]), ("serve", [])):
            with self.subTest(command=command):
                # Do not hang if the old dummy server starts without a model.
                try:
                    result = subprocess.run([BINARY, command, *args], capture_output=True,
                                            text=True, timeout=1)
                except subprocess.TimeoutExpired:
                    self.fail(f"{command} started without its required --model")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--model", result.stderr)

    def test_invalid_model_is_not_a_dummy_success(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "invalid.gguf"
            model.write_bytes(b"not GGUF")
            for command, args in (("embedding", ["--prompt", "hello"]), ("serve", [])):
                result = self.run_cli("--model", model, "--threads", "1", *args, command=command)
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr)

    def test_text_single_vector_matches_model_oracle(self):
        args = self.model_args()
        cases = self.tiny_reference["cases"] if FIXTURES is None else [
            {"input": case["text"], "embedding": case["expected"]}
            for case in json.loads((FIXTURES / "model-fixture" / "fixture.json").read_text())["text_cases"]
        ]
        for case in cases:
            actual = json.loads(self.success(*args, "--text", case["input"]))
            self.assert_vector_close(actual, case["embedding"])

    def test_prompt_raw_default_and_explicit_formats(self):
        expected = self.vector("hello")
        args = self.model_args()
        for flags in ([], ["--embd-output-format", "raw"]):
            self.assert_vector_close([float(x) for x in self.success(*args, "-p", "hello", *flags).split()], expected)
        self.assertEqual(json.loads(self.success(*args, "--text", "hello", "--embd-output-format", "array")), [expected])
        for option in ("-p", "--prompt"):
            result = json.loads(self.success(*args, option, "hello", "--embd-output-format", "json"))
            self.assertEqual(result, {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": expected}]})

    def test_separator_and_empty_segments(self):
        args = self.model_args()
        for text, separator, parts in (("hello\nworld", "\n", ["hello", "world"]),
                                       ("hello<#sep#>world", "<#sep#>", ["hello", "world"]),
                                       ("hello\n\n", "\n", ["hello", "", ""])):
            actual = json.loads(self.success(*args, "--prompt", text, "--embd-separator", separator,
                                             "--embd-output-format", "array"))
            self.assertEqual(actual, [self.vector(part) for part in parts])
        # --text is one item, never separator-split (including a trailing newline).
        direct = self.vector("hello\nworld\n")
        self.assertTrue(all(isinstance(x, (int, float)) for x in direct))

    def test_file(self):
        args = self.model_args()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_bytes("你好\nworld".encode("utf-8"))
            for flag in ("-f", "--file"):
                actual = json.loads(self.success(*args, flag, path, "--embd-output-format", "array"))
                self.assertEqual(actual, [self.vector("你好"), self.vector("world")])

    def test_json_similarity_uses_true_cosine_for_unnormalized_vectors(self):
        args = self.model_args("clamp3")
        result = json.loads(self.success(*args, "--prompt", "ab c\na <mask> b", "--embd-output-format", "json+"))
        vectors = [item["embedding"] for item in result["data"]]
        norms = [math.sqrt(sum(x*x for x in vector)) for vector in vectors]
        self.assertTrue(any(abs(norm - 1) > 1e-3 for norm in norms), "fixture must expose unnormalized outputs")
        for i in range(2):
            for j in range(2):
                expected = sum(a*b for a, b in zip(vectors[i], vectors[j])) / (norms[i]*norms[j])
                self.assertAlmostEqual(result["cosineSimilarity"][i][j], expected, places=6)
        single = json.loads(self.success(*args, "-p", "hello", "--embd-output-format", "json+"))
        self.assertNotIn("cosineSimilarity", single)

    def test_audio_single_vector_and_native_frames(self):
        args = self.model_args("mert2")
        fixture = json.loads((FIXTURES / "mert2-fixture" / "fixture.json").read_text())
        case = next(case for case in fixture["waveform_cases"] if case["full_song"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            samples = struct.pack("<" + "f" * len(case["waveform"]), *case["waveform"])
            rate = fixture["sample_rate"]
            path.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt " +
                             struct.pack("<IHHIIHH", 16, 3, 1, rate, rate*4, 4, 32) +
                             b"data" + struct.pack("<I", len(samples)) + samples)
            pooled = json.loads(self.success(*args, "--audio", path))
            self.assert_vector_close(pooled, case["expected_pooled"], fixture["tolerance"])
            frames = json.loads(self.success(*args, "-a", path, "--frames"))
            self.assertEqual(set(frames), {"frames", "dimension", "values"})
            self.assertEqual(frames["frames"], case["frames"])
            self.assertEqual(frames["dimension"], fixture["hidden_size"])
            self.assert_vector_close(frames["values"], case["expected_frames"], fixture["tolerance"])

    def test_invalid_arguments(self):
        args = self.model_args()
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory) / "text.txt"
            text.write_text("hello")
            empty = Path(directory) / "empty.txt"
            empty.write_text("")
            audio = Path(directory) / "audio.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(48000)
                output.writeframes(b"\0\0" * 300)
            for flags in ([], ["-p", ""], ["--text", "x", "--prompt", "x"],
                          ["-p", "x", "-f", str(text)], ["-f", str(empty)],
                          ["--text", "x", "--audio", str(audio)],
                          ["--text", "x", "--crop-offset", "0"],
                          ["--text", "x", "--frames"], ["--text", "x", "--clamp3-features"],
                          ["-a", str(audio), "--frames", "--crop-offset", "0"],
                          ["-a", str(audio), "--frames", "--clamp3-features"],
                          ["-a", str(audio), "--clamp3-features", "--crop-offset", "0"],
                          ["-p", "x", "--embd-separator", ""],
                          ["-p", "x", "--embd-output-format", "invalid"],
                          ["-p", "x", "--threads", "-1"]):
                with self.subTest(flags=flags):
                    result = self.run_cli(*args, *flags)
                    self.assertGreater(result.returncode, 0)
                    self.assertTrue(result.stderr)
                    self.assertEqual(result.stdout, "")

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux procfs")
    def test_file_read_failure(self):
        args = self.model_args()
        try:
            with open("/proc/self/mem", "rb") as stream:
                stream.read(1)
        except OSError as error:
            if error.errno != errno.EIO:
                self.skipTest(str(error))
        else:
            self.skipTest("procfs does not produce a read error")
        result = self.run_cli(*args, "-f", "/proc/self/mem")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to read input file", result.stderr)
        self.assertEqual(result.stdout, "")

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_stdout_write_failure(self):
        args = self.model_args()
        for output_format in ("raw", "json"):
            with self.subTest(output_format=output_format), open("/dev/full", "wb") as output:
                result = subprocess.run([BINARY, "embedding", *args, "-p", "hello",
                                         "--embd-output-format", output_format], stdout=output,
                                        stderr=subprocess.PIPE, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Failed to write embedding output", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
