"""Real GGUF-backed HTTP integration tests; Python standard library only."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from tiny_model import write_model


@contextmanager
def running_server(binary, model):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            [binary, "serve", "--model", str(model), "--threads", "1",
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=output, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output.seek(0)
                    raise AssertionError("Server exited: " + output.read())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise AssertionError("Server did not become ready")
            yield f"http://127.0.0.1:{port}/v1/embeddings"
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def request(url, payload, raw=False):
    body = payload if raw else json.dumps(payload)
    req = urllib.request.Request(url, body.encode(), {"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        if response.headers.get_content_type() != "application/json":
            raise AssertionError(f"Expected JSON, got {response.status}: {response.read()!r}")
        return response.status, json.load(response)


class EmbeddingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="clap-http-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.model = Path(cls.directory.name) / "tiny-text.gguf"
        cls.fixture = write_model(cls.model)
        cls.server = running_server(sys.argv[1], cls.model)
        cls.url = cls.server.__enter__()
        cls.addClassCleanup(cls.server.__exit__, None, None, None)

    def assert_embedding(self, actual, expected):
        self.assertEqual(len(actual), self.fixture["dimensions"])
        self.assertTrue(all(math.isfinite(value) for value in actual))
        self.assertAlmostEqual(sum(value * value for value in actual), 1.0, delta=1e-5)
        for value, reference in zip(actual, expected):
            self.assertAlmostEqual(value, reference, delta=2e-5)

    def test_single_input_matches_independent_model_oracle(self):
        for case in self.fixture["cases"]:
            with self.subTest(input=case["input"]):
                status, result = request(self.url, {"input": case["input"], "model": self.model.name})
                self.assertEqual(status, 200)
                self.assertEqual(result["object"], "list")
                self.assertEqual(result["model"], self.model.name)
                self.assertEqual(len(result["data"]), 1)
                self.assertEqual(result["data"][0]["object"], "embedding")
                self.assertEqual(result["data"][0]["index"], 0)
                self.assert_embedding(result["data"][0]["embedding"], case["embedding"])
                self.assertEqual(result["usage"], {"prompt_tokens": case["tokens"], "total_tokens": case["tokens"]})

    def test_batch_order_dimensions_and_usage(self):
        cases = self.fixture["cases"]
        status, result = request(self.url, {
            "input": [case["input"] for case in cases], "encoding_format": "float",
            "dimensions": self.fixture["dimensions"]})
        self.assertEqual(status, 200)
        self.assertEqual(result["model"], self.model.name)
        self.assertEqual([entry["index"] for entry in result["data"]], list(range(len(cases))))
        for entry, case in zip(result["data"], cases):
            self.assert_embedding(entry["embedding"], case["embedding"])
        count = sum(case["tokens"] for case in cases)
        self.assertEqual(result["usage"], {"prompt_tokens": count, "total_tokens": count})
        self.assertNotEqual(result["data"][0]["embedding"], result["data"][1]["embedding"])

    def test_concurrent_requests_use_one_model_safely(self):
        cases = self.fixture["cases"] * 8
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda case: request(self.url, {"input": case["input"]}), cases))
        for (status, result), case in zip(results, cases):
            self.assertEqual(status, 200)
            self.assert_embedding(result["data"][0]["embedding"], case["embedding"])

    def test_invalid_requests_are_json_400(self):
        invalid = ["{", "null", "[]", "42", "{}", '{"input": 42}', '{"input": []}',
                   '{"input": ["hello", 42]}', '{"input": ""}', '{"input": [""]}']
        invalid.extend(json.dumps(payload) for payload in [
            {"input": "x", "model": 42}, {"input": "x", "model": "not-loaded.gguf"},
            {"input": "x", "model": ""}, {"input": ["x"] * 1025},
            {"input": "x", "encoding_format": "base64"},
            {"input": "x", "encoding_format": None},
            *[{"input": "x", "dimensions": value} for value in (3, 0, -1, 5.0, True, None, "5", 2**64)],
        ])
        for body in invalid:
            with self.subTest(body=body[:120]):
                status, result = request(self.url, body, raw=True)
                self.assertEqual(status, 400)
                self.assertEqual(result["error"]["type"], "invalid_request_error")
                self.assertTrue(result["error"]["message"])
        # Rejections must not poison the model/server.
        self.assertEqual(request(self.url, {"input": "hello"})[0], 200)

    def test_inference_failure_is_json_and_server_survives(self):
        model = Path(self.directory.name) / "broken-projection.gguf"
        write_model(model, broken_projection=True)
        with running_server(sys.argv[1], model) as url:
            for _ in range(2):
                status, result = request(url, {"input": "hello"})
                self.assertEqual(status, 500)
                self.assertEqual(result["error"]["type"], "server_error")
                self.assertIn("projection", result["error"]["message"].lower())

    def test_startup_load_failure_exits_nonzero_before_listen(self):
        invalid = Path(self.directory.name) / "invalid.gguf"
        invalid.write_bytes(b"not a GGUF model")
        for model in (invalid, Path(self.directory.name) / "missing.gguf"):
            with self.subTest(model=model.name):
                result = subprocess.run([sys.argv[1], "serve", "--model", str(model),
                                         "--host", "127.0.0.1", "--port", "8080"],
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Listening on", result.stdout)
                self.assertTrue(result.stderr)
                if model == invalid:
                    self.assertIn("GGUF", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
