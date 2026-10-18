"""Exercise the real embeddings HTTP endpoint without external packages."""

import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request


class EmbeddingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}/v1/embeddings"
        cls.process = subprocess.Popen(
            [sys.argv[1], "serve", "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise AssertionError("Server exited: " + cls.process.communicate()[0])
            try:
                cls.request({"input": "ready"})
                return
            except urllib.error.URLError:
                time.sleep(0.05)
        raise AssertionError("Server did not become ready")

    @classmethod
    def stop_server(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
        try:
            cls.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.communicate()

    @classmethod
    def request(cls, payload, raw=False):
        body = payload if raw else json.dumps(payload)
        request = urllib.request.Request(
            cls.url, body.encode(), {"Content-Type": "application/json"}
        )
        try:
            response = urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            assert response.headers.get_content_type() == "application/json"
            return response.status, json.load(response)

    def test_single_input(self):
        status, result = self.request({"input": "hello", "model": "test-model"})
        self.assertEqual(status, 200)
        self.assertEqual(result["object"], "list")
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["data"], [
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}
        ])
        self.assertEqual(result["usage"], {"prompt_tokens": 0, "total_tokens": 0})

    def test_batch_input(self):
        status, result = self.request({"input": ["hello", "world"], "encoding_format": "float"})
        self.assertEqual(status, 200)
        self.assertEqual(result["model"], "dummy")
        self.assertEqual([entry["index"] for entry in result["data"]], [0, 1])
        for entry in result["data"]:
            self.assertEqual(entry["embedding"], [1.0, 0.0, 0.0])

    def test_invalid_requests(self):
        invalid = [
            "{", "null", "[]", "42",
            "{}", '{"input": 42}', '{"input": []}',
            '{"input": ["hello", 42]}', '{"input": ""}',
            '{"input": [""]}', '{"input": "x", "model": 42}',
            '{"input": "x", "encoding_format": "base64"}',
            '{"input": "x", "encoding_format": null}',
            '{"input": "x", "dimensions": 2}',
        ]
        for body in invalid:
            with self.subTest(body=body):
                status, result = self.request(body, raw=True)
                self.assertEqual(status, 400)
                self.assertEqual(result["error"]["type"], "invalid_request_error")
                self.assertTrue(result["error"]["message"])

    def test_fixed_dimensions(self):
        status, result = self.request({"input": "hello", "dimensions": 3})
        self.assertEqual(status, 200)
        self.assertEqual(len(result["data"][0]["embedding"]), 3)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
