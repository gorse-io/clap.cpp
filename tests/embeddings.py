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
def running_server(binary, model, *options):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            [binary, "serve", "--model", str(model), "--threads", "1",
             "--host", "127.0.0.1", "--port", str(port), *options],
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
                output.seek(0)
                raise AssertionError("Server did not become ready: " + output.read())
            yield f"http://127.0.0.1:{port}/v1/embeddings"
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise AssertionError("Server did not shut down orderly within 5 seconds")
            # Windows Popen.terminate uses TerminateProcess, not SIGTERM.
            if sys.platform != "win32" and process.returncode != 0:
                output.seek(0)
                raise AssertionError(f"Server shutdown returned {process.returncode}: " + output.read())


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

    def test_half_closed_client_receives_complete_embeddings(self):
        import http.client
        from urllib.parse import urlsplit

        endpoint = urlsplit(self.url)
        # Send FIN with the complete HTTP request. The receive direction remains
        # open: EOF on the server's read side is not an abandoned response.
        for count in (1, 1024):
            with self.subTest(count=count):
                cases = [self.fixture["cases"][i % len(self.fixture["cases"])] for i in range(count)]
                body = json.dumps({"input": [case["input"] for case in cases]}).encode()
                wire = (f"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\n"
                        f"Content-Type: application/json\r\nConnection: close\r\n"
                        f"Content-Length: {len(body)}\r\n\r\n").encode() + body
                with socket.create_connection((endpoint.hostname, endpoint.port), timeout=10) as client:
                    client.sendall(wire)
                    client.shutdown(socket.SHUT_WR)
                    with http.client.HTTPResponse(client) as response:
                        response.begin()
                        self.assertEqual(response.status, 200)
                        payload = response.read()
                        self.assertTrue(payload, "Half-closed client received an empty response")
                        self.assertEqual(response.headers.get_content_type(), "application/json")
                        self.assertEqual(len(payload), int(response.getheader("Content-Length", "-1")))
                        result = json.loads(payload)
                self.assertEqual(len(result["data"]), count)
                self.assertEqual([entry["index"] for entry in result["data"]], list(range(count)))
                for entry, case in zip(result["data"], cases):
                    self.assert_embedding(entry["embedding"], case["embedding"])
                tokens = sum(case["tokens"] for case in cases)
                self.assertEqual(result["usage"], {"prompt_tokens": tokens, "total_tokens": tokens})

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


class EmbeddingsFailureTest(unittest.TestCase):
    # Run startup/error cases without the healthy class-scoped server alive:
    # concurrent Metal library initialization can stall on paravirtual macOS GPUs.
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="clap-http-failure-")
        self.addCleanup(self.directory.cleanup)

    def test_bounded_queue_overload_and_recovery(self):
        from threading import Barrier

        model = Path(self.directory.name) / "tiny-queue.gguf"
        fixture = write_model(model)
        with running_server(sys.argv[1], model, "--queue-capacity", "1", "--device", "cpu") as url:
            barrier = Barrier(6)

            def batch(_):
                barrier.wait(timeout=10)
                return request(url, {"input": ["hello"] * 1024})

            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(batch, range(6)))
            statuses = [status for status, _ in results]
            self.assertIn(200, statuses)
            self.assertIn(503, statuses)
            for status, result in results:
                self.assertIn(status, (200, 503))
                if status == 503:
                    self.assertEqual(result["error"]["type"], "server_error")
                    self.assertEqual(result["error"]["code"], 503)
                    self.assertTrue(result["error"]["message"])
                else:
                    self.assertEqual(len(result["data"]), 1024)
                    self.assertEqual([item["index"] for item in result["data"]], list(range(1024)))
                    for item in result["data"]:
                        self.assertEqual(len(item["embedding"]), fixture["dimensions"])
            self.assertEqual(request(url, {"input": "hello"})[0], 200)

    def test_reset_batches_do_not_poison_model(self):
        import struct

        model = Path(self.directory.name) / "tiny-disconnect.gguf"
        fixture = write_model(model)
        with running_server(sys.argv[1], model, "--queue-capacity", "1", "--device", "cpu") as url:
            from urllib.parse import urlsplit

            endpoint = urlsplit(url)
            body = json.dumps({"input": ["hello"] * 1024}).encode()
            wire = (f"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\n"
                    f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
            # Abort with TCP RST rather than FIN, which could be a valid
            # write-half-close. Winsock's linger members are unsigned shorts;
            # POSIX uses ints. No response bytes are read or written as probes.
            linger = struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0)
            for _ in range(8):
                with socket.create_connection((endpoint.hostname, endpoint.port), timeout=5) as client:
                    client.sendall(wire)
                    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
            deadline = time.monotonic() + 10
            while True:
                status, result = request(url, {"input": "hello"})
                if status == 200:
                    break
                self.assertEqual(status, 503, result)
                self.assertLess(time.monotonic(), deadline, "Disconnected requests blocked recovery")
            expected = fixture["cases"][0]["embedding"]
            for actual, reference in zip(result["data"][0]["embedding"], expected):
                self.assertAlmostEqual(actual, reference, delta=2e-5)

    def test_bind_failure_joins_worker(self):
        model = Path(self.directory.name) / "tiny-bind.gguf"
        write_model(model)
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            result = subprocess.run([sys.argv[1], "serve", "--model", str(model), "--device", "cpu",
                                     "--host", "127.0.0.1", "--port", str(occupied.getsockname()[1])],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("Failed to bind", result.stderr)
            self.assertNotIn("Listening on", result.stdout)

    def test_queue_capacity_requires_positive_integer(self):
        model = Path(self.directory.name) / "tiny-queue.gguf"
        write_model(model)
        for capacity in ("0", "-1", "1.5"):
            result = subprocess.run([sys.argv[1], "serve", "--model", str(model),
                                     "--queue-capacity", capacity], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Listening on", result.stdout)
            self.assertIn("queue-capacity", result.stderr)

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
