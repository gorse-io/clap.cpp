"""Hermetic Google Benchmark CLI integration tests (stdlib only)."""
import csv
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

from tiny_model import write_model, write_mert_model

CLI = sys.argv[1]


class BenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="clap-bench-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.model = cls.root / "tiny.gguf"
        write_model(cls.model, waveform=True)
        cls.invalid = cls.root / "invalid.gguf"
        cls.invalid.write_bytes(b"not a model")
        cls.audio = cls.root / "audio.wav"
        with wave.open(str(cls.audio), "wb") as stream:
            stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            stream.writeframes(struct.pack("<2400h", *[
                int(4000 * math.sin(i * 0.071)) for i in range(2400)]))

    def run_cli(self, *args, ok=True):
        result = subprocess.run([CLI, *map(str, args)], capture_output=True,
                                text=True, timeout=60)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def bench(self, *args, native=()):
        return self.run_cli("bench", "-m", self.model, "-t", "1", *args, "--",
                            "--benchmark_min_time=0.001s", "--benchmark_repetitions=1",
                            "--benchmark_format=json", *native)

    def report(self, result, name, repetitions=1):
        document = json.loads(result.stdout)
        rows = [r for r in document["benchmarks"] if r.get("run_type") == "iteration"]
        self.assertEqual(len(rows), repetitions, document)
        for row in rows:
            self.assertTrue(row["name"].startswith(name + "/"), row)
            self.assertFalse(row.get("error_occurred"), row)
            self.assertGreater(row["iterations"], 0)
            self.assertGreater(row["real_time"], 0)
            self.assertEqual(row["time_unit"], "ms")
            self.assertGreater(row["items_per_second"], 0)
        return document

    def test_frames_real_inference(self):
        model = self.root / "mert.gguf"
        write_mert_model(model)
        result = self.run_cli("bench", "-m", model, "-t", "1", "-a", self.audio,
                              "--frames", "--", "--benchmark_min_time=0.001s",
                              "--benchmark_repetitions=1", "--benchmark_format=json")
        self.report(result, "audio_frames")
        self.run_cli("bench", "-m", model, "-t", "1", "-a", self.audio,
                     "--crop-offset", "0", "--", "--benchmark_min_time=0.001s", ok=False)

    def test_help(self):
        result = self.run_cli("bench", "--help")
        for option in ("--model", "--text", "--audio", "--threads", "--device",
                       "--frames", "--crop-offset", "--benchmark_"):
            self.assertIn(option, result.stdout)

    def test_text_native_json_and_repetitions(self):
        doc = self.report(self.bench("--text", "hello", "--device", "cpu",
                          native=("--benchmark_repetitions=2",)), "text", 2)
        self.assertEqual(doc["context"]["requested_device"], "cpu")

    def test_defaults_and_filter(self):
        result = self.run_cli("bench", "-m", self.model, "--text", "hello", "-t", "1", "--",
                              "--benchmark_format=json", "--benchmark_filter=^text/")
        self.report(result, "text", 3)
        # Native calibration is approximate and reuses iteration counts across
        # repetitions; do not impose a wall-duration threshold on shared runners.
        result = self.run_cli("bench", "-m", self.invalid, "--text", "", "--threads", "0", "--",
                              "--benchmark_list_tests=true")
        self.assertIn("text", result.stdout)
        result = self.run_cli("bench", "-m", self.invalid, "--text", "a", "--", "--help")
        self.assertIn("--benchmark_filter", result.stdout + result.stderr)

    def test_audio_resampling_and_crop(self):
        self.report(self.bench("-a", self.audio, "--crop-offset", "0"), "audio")

    def test_list_does_not_load_weights_or_decode(self):
        for args, name in [(["--text", "hello"], "text"),
                           (["--audio", self.invalid], "audio"),
                           (["--audio", self.invalid, "--frames"], "audio_frames")]:
            result = self.run_cli("bench", "-m", self.invalid, *args, "--",
                                  "--benchmark_list_tests=true")
            self.assertIn(name, result.stdout)
            self.assertNotIn("GGUF", result.stderr)

    def test_validation(self):
        cases = [[], ["--text", "a", "-a", self.audio], ["--text", "a", "--frames"],
                 ["--text", "a", "--crop-offset", "0"],
                 ["-a", self.audio, "--frames", "--crop-offset", "0"],
                 ["--text", "a", "-t", "-1"], ["-a", self.audio, "--crop-offset", "-1"],
                 ["--text", "a", "--unknown"],
                 ["--text", "a", "--benchmark_min_time=0.001s"],
                 ["--text", "a", "--", "--unknown"]]
        for args in cases:
            with self.subTest(args=args):
                self.run_cli("bench", "-m", self.model, *args, ok=False)
        self.run_cli("bench", "--text", "a", ok=False)
        self.run_cli("embedding", "-m", self.model, "--text", "a", "--",
                     "--benchmark_list_tests", ok=False)

    def test_unknown_native_flags_precede_model_load(self):
        result = self.run_cli("bench", "-m", self.invalid, "--text", "a", "--",
                              "--nonsense", ok=False)
        self.assertIn("unrecognized", result.stderr.lower())
        self.assertNotIn("GGUF", result.stderr)

    def test_no_match_and_output_failure(self):
        for flag in ("--benchmark_filter=does_not_match", "--benchmark_filter=[",
                     f"--benchmark_out={self.root / 'missing' / 'out.json'}"):
            with self.subTest(flag=flag):
                self.run_cli("bench", "-m", self.model, "--text", "a", "--", flag,
                              "--benchmark_min_time=0.001s", ok=False)
        if Path("/dev/full").exists():
            self.run_cli("bench", "-m", self.model, "-t", "1", "--text", "a", "--",
                          "--benchmark_out=/dev/full", "--benchmark_min_time=0.001s",
                          "--benchmark_repetitions=1", ok=False)

    def test_native_output_file(self):
        target = self.root / "report.json"
        self.bench("--text", "a", native=(f"--benchmark_out={target}",
                   "--benchmark_out_format=json"))
        self.report(subprocess.CompletedProcess([], 0, target.read_text()), "text")

    def test_native_csv_and_console(self):
        target = self.root / "report.csv"
        result = self.bench("--text", "a", native=("--benchmark_format=csv",
                            f"--benchmark_out={target}", "--benchmark_out_format=csv"))
        for contents in (result.stdout, target.read_text()):
            # Native CSV files include diagnostic context before their header.
            contents = contents[contents.index("name,iterations,"):]
            rows = list(csv.DictReader(io.StringIO(contents)))
            self.assertTrue(rows[0]["name"].startswith("text/"), contents)
            self.assertGreater(int(rows[0]["iterations"]), 0)
        result = self.bench("--text", "a", native=("--benchmark_format=console",))
        self.assertIn("text/real_time", result.stdout)

    def test_native_tabular_file(self):
        target = self.root / "table.txt"
        for flag, enabled in [("--benchmark_counters_tabular=true", True),
                              ("--benchmark_counters_tabular", True),
                              ("--benchmark_counters_tabular=false", False)]:
            with self.subTest(flag=flag):
                self.bench("--text", "a", native=(f"--benchmark_out={target}",
                           "--benchmark_out_format=console", flag))
                header = next(line for line in target.read_text().splitlines()
                              if "Benchmark" in line and "Time" in line)
                self.assertEqual("items_per_second" in header, enabled)

    def test_inference_errors_nonzero(self):
        result = self.run_cli("bench", "-m", self.model, "-a", self.audio, "--frames",
                              "-t", "1", "--", "--benchmark_min_time=0.001s",
                              "--benchmark_repetitions=1", "--benchmark_format=json", ok=False)
        self.assertTrue(json.loads(result.stdout)["benchmarks"][0]["error_occurred"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *sys.argv[2:]])
