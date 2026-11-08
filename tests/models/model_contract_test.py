#!/usr/bin/env python3
"""Download-free regressions for selection, failures and shared model checks."""
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from model_architectures import ARCHITECTURES
from model_contract import (Check, Command, Compare, Context, Diagnostic, Frames, Vector,
                            metadata_model, read_metadata_model, run_cases)

RUNNER = Path(__file__).with_name("model_tests.py")


def python_command(program, **kwargs):
    return Command("python-unit-probe", [sys.executable, "-c", program], **kwargs)


class SelectionTests(unittest.TestCase):
    def runner(self, *args):
        return subprocess.run([sys.executable, str(RUNNER), *args], text=True,
                              capture_output=True, timeout=10)

    def test_fixture_free_architecture_listing(self):
        result = self.runner("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            f"{name}/{suite}" for name, adapter in ARCHITECTURES.items() for suite in adapter.suites])

    def test_unknown_architecture_is_not_silently_skipped(self):
        result = self.runner("--arch", "unknown", "--suite", "cli")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_unsupported_suite_and_missing_requirements(self):
        for args, expected in [
                (("--arch", "mert", "--suite", "cli"), "does not support suite"),
                (("--arch", "mert2", "--suite", "cli"), "requires --fixture"),
                (("--arch", "mert", "--suite", "invalid"), "requires --cli"),
                (("--arch", "clap2", "--suite", "invalid", "--fixture", "."), "requires --build")]:
            with self.subTest(args=args):
                result = self.runner(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_missing_fixture_has_architecture_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.runner("--arch", "mert2", "--suite", "cli", "--cli", sys.executable,
                                 "--fixture", str(Path(temp) / "nonexistent"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL mert2/cli", result.stderr)
        self.assertIn("fixture.json", result.stderr)

    def test_case_filter_and_no_match_failure(self):
        called = []
        cases = [Check("first", lambda: called.append("first")),
                 Check("second", lambda: called.append("second"))]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_cases("clap", "cli", cases, "second"), 1)
        self.assertEqual(called, ["second"])
        with self.assertRaisesRegex(ValueError, "no cases matched clap/cli"):
            run_cases("clap", "cli", cases, "nonexistent")


class CommandTests(unittest.TestCase):
    def test_real_subprocess_json_vector(self):
        python_command("print('[1.0, 2.0]')", output=Vector(2, [1.0, 2.0], 1e-5)).run()

    def test_rejection_requires_both_exit_and_diagnostic(self):
        python_command("import sys; print('Invalid MERT', file=sys.stderr); sys.exit(3)",
                       success=False, positive_exit=True, diagnostics=("invalid mert",),
                       diagnostic_lower=True).run()
        for program in ("import sys; print('invalid mert', file=sys.stderr)",
                        "import sys; print('unrelated', file=sys.stderr); sys.exit(1)"):
            with self.subTest(program=program), self.assertRaises(AssertionError):
                python_command(program, success=False, diagnostics=("invalid mert",)).run()

    def test_signal_does_not_count_as_positive_rejection(self):
        result = subprocess.CompletedProcess(["native"], -11, "", "invalid MERT")
        with patch("model_contract.subprocess.run", return_value=result):
            with self.assertRaisesRegex(AssertionError, "not a signal"):
                Command("signal", ["native"], success=False, positive_exit=True).run()

    def test_any_diagnostic_and_nonempty_stderr(self):
        python_command("import sys; print('BPE', file=sys.stderr); sys.exit(1)", success=False,
                       diagnostics=("clap2", "bpe"), diagnostic_any=True, diagnostic_lower=True).run()
        with self.assertRaisesRegex(AssertionError, "expected stderr"):
            python_command("import sys; sys.exit(1)", success=False, nonempty_stderr=True).run()

    def test_mixed_case_sensitive_diagnostic_alternatives(self):
        options = dict(success=False, diagnostic_any=True,
                       diagnostics=(Diagnostic("tensor", ignore_case=True), "qkv_proj"))
        for diagnostic in ("TENSOR", "qkv_proj"):
            python_command(f"import sys; print({diagnostic!r}, file=sys.stderr); sys.exit(1)", **options).run()
        with self.assertRaises(AssertionError):
            python_command("import sys; print('QKV_PROJ', file=sys.stderr); sys.exit(1)", **options).run()

    def test_failure_includes_case_command_stdout_stderr_and_exit(self):
        command = python_command("import sys; print('unit stdout'); print('unit stderr', file=sys.stderr); sys.exit(9)")
        with self.assertRaises(AssertionError) as raised:
            run_cases("mert2", "cli", [command])
        for token in ("mert2/cli/python-unit-probe", "command:", "unit stdout", "unit stderr", "exit: 9"):
            self.assertIn(token, str(raised.exception))

    def test_invalid_json_and_dimension_have_native_output(self):
        for payload, expected in (("not-json", "not-json"), ("[1]", "[1]")):
            with self.subTest(payload=payload), self.assertRaises(AssertionError) as raised:
                python_command(f"print({payload!r})", output=Vector(2)).run()
            self.assertIn(expected, str(raised.exception))
            self.assertIn("stdout:", str(raised.exception))

    def test_timeout_has_command_diagnostic(self):
        with self.assertRaisesRegex(AssertionError, "timed out after"):
            python_command("import time; time.sleep(10)", timeout=0.1).run()

    def test_comparison_preserves_both_native_results(self):
        def compare(results):
            assert results[0].stdout == results[1].stdout, "different outputs"
        pair = Compare("pair", [python_command("print('one')"), python_command("print('two')")], compare)
        with self.assertRaisesRegex(AssertionError, "paired command results") as raised:
            pair.run()
        self.assertIn("one", str(raised.exception))
        self.assertIn("two", str(raised.exception))


class OutputTests(unittest.TestCase):
    def test_strict_and_inclusive_tolerances(self):
        Vector(1, [0.0], 0.125, inclusive=True)([0.125])
        with self.assertRaises(AssertionError):
            Vector(1, [0.0], 0.125)([0.125])
        with self.assertRaises(AssertionError):
            Vector(1, [0.0], 0.125, inclusive=True)([0.126])

    def test_dimension_reference_and_nan_failures(self):
        for check, actual in ((Vector(2), [1]), (Vector(2, [1], 0.1), [1, 2]),
                              (Vector(1, [0], 0.1), [float("nan")])):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                check(actual)

    def test_frames_checks_count_dimension_values_and_error(self):
        check = Frames(2, 1, [1, 2], 0.01)
        check({"frames": 2, "dimension": 1, "values": [1, 2]})
        for actual in ({"frames": 1, "dimension": 1, "values": [1, 2]},
                       {"frames": 2, "dimension": 2, "values": [1, 2]},
                       {"frames": 2, "dimension": 1, "values": [1]},
                       {"frames": 2, "dimension": 1, "values": [1, 2.1]}):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                check(actual)

    def test_string_metadata_rewrite_keeps_tensor_infos_and_payload(self):
        # Tensor-info bytes use a real GGUF tensor entry, with unchanged offset 0.
        name = b"weight"
        infos = struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 2, 2, 0, 0)
        payload = struct.pack("<4f", 1, 2, 3, 4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            metadata = {"general.architecture": "test", "test.config": "{}"}
            metadata_model(path, metadata, 1, infos, payload)
            self.assertEqual(read_metadata_model(path), (metadata, 1, infos, payload))
            metadata["test.config"] = json.dumps({"longer": "x" * 80})
            metadata_model(path, metadata, 1, infos, payload)
            self.assertEqual(read_metadata_model(path), (metadata, 1, infos, payload))

    def test_mert_and_clap2_architecture_case_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            context = Context(Path("unused-cli"), Path(temp), Path(temp), Path(temp))
            mert = list(ARCHITECTURES["mert"].cases("invalid", context))
            self.assertEqual(len(mert), 32)  # 31 native mutations + specialized converter checks
            self.assertTrue(all(case.positive_exit for case in mert[:-1]))
            clap2 = list(ARCHITECTURES["clap2"].cases("invalid", context))
            self.assertEqual(len(clap2), 29)  # metadata + 2 smoke + 12 converter/native pairs + 2 tensors
            self.assertEqual([case.name for case in clap2[:3]], ["conversion-metadata", "analytical-f32", "analytical-f16"])

    def test_mert2_full_song_case_waveform_and_deadlines(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            reference = dict(sample_rate=24000, hidden_size=1, tolerance=0.0003,
                             waveform_cases=[dict(full_song=False), dict(full_song=True,
                             waveform=[0.25, -0.5], expected_pooled=[0.125], frames=2,
                             expected_frames=[0.25, -0.5])])
            (directory / "fixture.json").write_text(json.dumps(reference))
            cases = list(ARCHITECTURES["mert2"].cases("cli", Context(Path("native"), fixture=directory)))
            self.assertEqual([case.name for case in cases], ["full-song-pooled", "full-song-frames"])
            self.assertTrue(all(case.timeout == 120 for case in cases))
            self.assertEqual(cases[0].output.tolerance, reference["tolerance"])
            self.assertTrue(cases[0].output.inclusive)
            self.assertEqual(cases[1].args[-1], "--frames")
            self.assertEqual((directory / "full-song.wav").read_bytes()[44:], struct.pack("<2f", 0.25, -0.5))

    def test_clap_and_clamp3_case_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            context = Context(Path("native"), fixture=directory)
            (directory / "fixture.json").write_text(json.dumps({"text_cases": [
                {"text": "one", "expected": [1, 2]}, {"text": "two", "expected": [2, 1]}]}))
            clap = list(ARCHITECTURES["clap"].cases("cli", context))
            self.assertEqual(len(clap), 6)  # 2 texts + 3 invalid arguments + resampling
            self.assertEqual(clap[0].output.tolerance, 3e-5)
            self.assertFalse(clap[0].output.inclusive)
            clamp3 = list(ARCHITECTURES["clamp3"].cases("cli", context))
            self.assertEqual(len(clamp3), 5)  # waveform + 3 obsolete options + bad architecture
            self.assertEqual(clamp3[0].output.dimension, 6)
            metadata_model(directory / "model.gguf", {"clamp3.config": json.dumps({
                "text_config": {"pad_token_id": 0, "is_decoder": False},
                "audio_config": {"layer_norm_eps": 1e-5}})})
            invalid = list(ARCHITECTURES["clamp3"].cases("invalid", context))
            self.assertEqual([case.name for case in invalid], ["negative-pad", "negative-epsilon", "causal-decoder"])
            self.assertEqual([case.diagnostics for case in invalid], [("padding token",), ("layer norm epsilon",), ("decoder",)])


if __name__ == "__main__":
    unittest.main()
