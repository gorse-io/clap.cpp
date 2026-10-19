"""Exercise the embedding subcommand without a model or external packages."""

import errno
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EmbeddingCLITest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.argv[1], "embedding", *args], capture_output=True, text=True, timeout=5
        )

    def success(self, *args):
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return result.stdout

    def test_help(self):
        output = self.success("--help")
        for option in ("--prompt", "--file", "--embd-separator", "--embd-output-format"):
            self.assertIn(option, output)

    def test_default_and_raw(self):
        for args in (("-p", "hello"), ("--prompt", "hello", "--embd-output-format", "raw")):
            with self.subTest(args=args):
                self.assertEqual([float(x) for x in self.success(*args).split()], [1, 0, 0])

    def test_array_and_separator(self):
        for prompt, separator in (("hello\nworld", "\n"), ("hello<#sep#>world", "<#sep#>")):
            output = self.success("-p", prompt, "--embd-separator", separator,
                                  "--embd-output-format", "array")
            self.assertEqual(json.loads(output), [[1, 0, 0], [1, 0, 0]])

    def test_json(self):
        output = self.success("-p", "hello\nworld", "--embd-output-format", "json")
        self.assertEqual(json.loads(output), {
            "object": "list", "data": [
                {"object": "embedding", "index": i, "embedding": [1, 0, 0]}
                for i in range(2)
            ]
        })

    def test_json_similarity(self):
        for prompt, count in (("hello", 1), ("hello\nworld", 2)):
            output = json.loads(self.success("-p", prompt, "--embd-output-format", "json+"))
            self.assertEqual(len(output["data"]), count)
            if count > 1:
                self.assertEqual(output["cosineSimilarity"], [[1, 1], [1, 1]])
            else:
                self.assertNotIn("cosineSimilarity", output)

    def test_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("你好\nworld", encoding="utf-8")
            for flag in ("-f", "--file"):
                output = self.success(flag, str(path), "--embd-output-format", "array")
                self.assertEqual(json.loads(output), [[1, 0, 0], [1, 0, 0]])

    def test_empty_segments_preserved(self):
        output = self.success("-p", "hello\n\n", "--embd-output-format", "array")
        self.assertEqual(json.loads(output), [[1, 0, 0]] * 3)

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux procfs")
    def test_file_read_failure(self):
        path = "/proc/self/mem"
        try:
            with open(path, "rb") as stream:
                stream.read(1)
        except OSError as error:
            if error.errno != errno.EIO:
                self.skipTest(f"procfs read failure unavailable: {error}")
        else:
            self.skipTest("procfs file does not produce a read error")
        result = self.run_cli("-f", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to read input file", result.stderr)
        self.assertEqual(result.stdout, "")

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_stdout_write_failure(self):
        for output_format in ("raw", "json"):
            with self.subTest(output_format=output_format), open("/dev/full", "wb") as output:
                result = subprocess.run(
                    [sys.argv[1], "embedding", "-p", "hello",
                     "--embd-output-format", output_format],
                    stdout=output, stderr=subprocess.PIPE, text=True, timeout=5
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Failed to write embedding output", result.stderr)

    def test_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.txt")
            valid = Path(directory) / "valid.txt"
            valid.write_text("hello", encoding="utf-8")
            empty = Path(directory) / "empty.txt"
            empty.write_text("", encoding="utf-8")
            for args in ((), ("-p", ""), ("-p", "x", "-f", str(valid)),
                         ("-f", str(empty)),
                         ("-f", missing), ("-f", directory),
                         ("-p", "x", "--embd-separator", ""),
                         ("-p", "x", "--embd-output-format", "invalid")):
                with self.subTest(args=args):
                    result = self.run_cli(*args)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(result.stderr)
                    self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
