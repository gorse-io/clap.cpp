"""Real audio HTTP/CLI parity and strict request-validation regression tests."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request

from embeddings import running_server, request
from tiny_model import write_model


def audio_item(data, **metadata):
    return {"content": [{"type": "input_audio", "input_audio": {"data": data, **metadata}}]}


class AudioEmbeddingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="clap-audio-http-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.model = Path(cls.directory.name) / "tiny-waveform.gguf"
        cls.fixture = write_model(cls.model, waveform=True)
        cls.fixtures = Path(sys.argv[2])
        cls.wav = base64.b64encode((cls.fixtures / "mono.wav").read_bytes()).decode()
        cls.server = running_server(sys.argv[1], cls.model)
        cls.url = cls.server.__enter__()
        cls.addClassCleanup(cls.server.__exit__, None, None, None)

    def assert_vector(self, actual, expected):
        self.assertEqual(len(actual), 5)
        self.assertEqual(len(expected), 5)
        for value, reference in zip(actual, expected):
            self.assertAlmostEqual(value, reference, delta=2e-5)

    def cli_audio(self, name):
        result = subprocess.run([sys.argv[1], "embedding", "--model", str(self.model),
                                 "--audio", str(self.fixtures / name), "--threads", "1"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_audio_formats_match_real_cli_without_required_format(self):
        for name in ("mono.wav", "stereo.mp3", "mono.flac", "44100.flac"):
            with self.subTest(name=name):
                item = audio_item(base64.b64encode((self.fixtures / name).read_bytes()).decode())
                status, result = request(self.url, {"input": [item]})
                self.assertEqual(status, 200, result)
                self.assertNotIn("usage", result)
                self.assertEqual(result["model"], self.model.name)
                self.assertEqual(result["data"][0]["index"], 0)
                self.assert_vector(result["data"][0]["embedding"], self.cli_audio(name))
        self.assertEqual(request(self.url, {"input": [audio_item(self.wav, format="wav")]})[0], 200)

    def test_mixed_batch_order_and_concurrent_calls(self):
        cases = self.fixture["cases"]
        payload = {"input": [cases[0]["input"], audio_item(self.wav), cases[1]["input"]]}
        expected = [cases[0]["embedding"], self.cli_audio("mono.wav"), cases[1]["embedding"]]
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: request(self.url, payload), range(3)))
        for status, result in results:
            self.assertEqual(status, 200, result)
            self.assertNotIn("usage", result)
            self.assertEqual([row["index"] for row in result["data"]], [0, 1, 2])
            for row, vector in zip(result["data"], expected):
                self.assert_vector(row["embedding"], vector)

    def test_invalid_audio_is_json_400_and_server_survives(self):
        invalid = [None, {}, {"content": None}, {"content": []},
                   {"content": [audio_item(self.wav)["content"][0]] * 2},
                   {"content": [{"type": "text", "text": "hello"}]},
                   {"content": [{"type": "input_audio"}]},
                   {"content": [{"type": "input_audio", "input_audio": None}]},
                   {"content": [{"type": "input_audio", "input_audio": {}}]}]
        # Wrong types, illegal characters, misplaced/excess padding, missing padding,
        # whitespace, URL alphabet and noncanonical unused padding bits.
        invalid.extend(audio_item(data) for data in
                       (None, 42, [], "", "!!!!", "A", "AAA", "AAAA=", "=AAA", "AA=A",
                        "A===", "AA==AA==", "AA==\n", "____", "Zh==", "Zm9=", "éAAA"))
        # Mutate encodings of actual decodable WAV bytes so permissive base64
        # handling cannot pass these tests merely because the audio is garbage.
        invalid.extend(audio_item(data) for data in
                       (self.wav + "\n", self.wav + "====", self.wav.rstrip("="),
                        self.wav[:-2] + "1=", self.wav[:20] + " " + self.wav[20:],
                        self.wav.replace("/", "_"), "data:audio/wav;base64," + self.wav))
        invalid.extend(audio_item(base64.b64encode((self.fixtures / name).read_bytes()).decode())
                       for name in ("invalid.bin", "empty.wav", "truncated.wav", "truncated.flac", "nonfinite.wav"))
        for item in invalid:
            with self.subTest(item=str(item)[:100]):
                status, result = request(self.url, {"input": ["hello", item]})
                self.assertEqual(status, 400, result)
                self.assertEqual(result["error"]["type"], "invalid_request_error")
                self.assertTrue(result["error"]["message"])
        self.assertEqual(request(self.url, {"input": [audio_item(self.wav)]})[0], 200)
        self.assertEqual(request(self.url, {"input": "hello"})[0], 200)

    def test_audio_payload_larger_than_one_mib(self):
        # WAV readers ignore trailing bytes; exercises HTTP byte budget without
        # expensive long-waveform inference or another fixture generator.
        data = (self.fixtures / "mono.wav").read_bytes() + b"\0" * (1024 * 1024)
        status, result = request(self.url, {"input": [audio_item(base64.b64encode(data).decode())]})
        self.assertEqual(status, 200, result)
        self.assertNotIn("usage", result)
        oversized = urllib.request.Request(self.url, b" " * (32 * 1024 * 1024 + 1),
                                           {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(oversized, timeout=10)
        self.assertEqual(error.exception.code, 413)
        error.exception.close()
        self.assertEqual(request(self.url, {"input": "hello"})[0], 200)

    def test_valid_audio_model_failure_is_500(self):
        # Default reduced-mel fixture intentionally cannot run the waveform frontend.
        model = Path(self.directory.name) / "text-only-fixture.gguf"
        write_model(model)
        with running_server(sys.argv[1], model) as url:
            status, result = request(url, {"input": [audio_item(self.wav)]})
            self.assertEqual(status, 500, result)
            self.assertEqual(result["error"]["type"], "server_error")
            self.assertEqual(request(url, {"input": "hello"})[0], 200)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
