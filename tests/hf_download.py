"""Hermetic CLI integration tests; no Hugging Face access or Python runtime dependency."""
import hashlib
import http.server
import json
import os
from pathlib import Path
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

CLI = sys.argv[1]
TLS_FIXTURE = sys.argv[2]
sys.argv = sys.argv[:1] + sys.argv[3:]
COMMIT = "a" * 40
DATA = b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + b"\0" * 2048
OID = hashlib.sha256(DATA).hexdigest()
sys.dont_write_bytecode = True


class FixtureServer(http.server.ThreadingHTTPServer):
    fail = False
    bad_range = False
    redirect = ""
    paginate = False
    drop_once = False
    corrupt = False
    ignore_range = False
    data = DATA

    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.requests = []
        self.files = ["nested/model-Q8_0.gguf", "model-Q4_K_M.gguf"]


class Handler(http.server.BaseHTTPRequestHandler):
    server: FixtureServer

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # Certificate/hostname rejection can close TLS before an HTTP request.
            pass

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        data = getattr(self.server, "data", DATA)
        oid = hashlib.sha256(data).hexdigest()
        self.server.requests.append((self.path, dict(self.headers)))
        if self.path == "/api/models/test/model/revision/main":
            self.send(200, json.dumps({"sha": COMMIT}).encode())
        elif self.path.startswith(f"/api/models/test/model/tree/{COMMIT}"):
            rows = [{"type": "file", "path": p, "size": len(data),
                     "lfs": {"oid": oid, "size": len(data)}} for p in self.server.files]
            headers = {}
            if getattr(self.server, "paginate", False):
                if "cursor=second" in self.path:
                    rows = rows[1:]
                else:
                    rows = rows[:1]
                    scheme = "https" if isinstance(self.connection, ssl.SSLSocket) else "http"
                    headers["Link"] = f'<{scheme}://127.0.0.1:{self.server.server_port}/api/models/test/model/tree/{COMMIT}?cursor=second>; rel="next"'
            self.send(200, json.dumps(rows).encode(), headers)
        elif self.path == "/redirected":
            self.send(200, data)
        elif self.path.startswith(f"/test/model/resolve/{COMMIT}/"):
            if getattr(self.server, "redirect", ""):
                self.send(302, b"redirect body", {"Location": self.server.redirect})
                return
            if self.server.fail:
                self.send(503, b"retry")
                return
            if getattr(self.server, "drop_once", False):
                self.server.drop_once = False
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data[:30])
                return
            if getattr(self.server, "corrupt", False):
                self.send(200, b"x" * len(data))
                return
            offset = int(self.headers.get("Range", "bytes=0-")[6:-1])
            if getattr(self.server, "ignore_range", False):
                offset = 0
            if offset and self.server.bad_range:
                self.send(206, data[offset:], {"Content-Range": f"bytes 0-{len(data)-1}/{len(data)}"})
            else:
                headers = {"Content-Range": f"bytes {offset}-{len(data)-1}/{len(data)}"} if offset else {}
                self.send(206 if offset else 200, data[offset:], headers)
        else:
            self.send(404, b"missing")

    def send(self, status, body, headers=None):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

class Download(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tls = tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve())
        cls.addClassCleanup(cls.tls.cleanup)
        subprocess.run([TLS_FIXTURE, cls.tls.name], check=True, capture_output=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve())
        self.cache = Path(self.tmp.name) / "cache"
        self.repo = self.cache / "models--test--model"
        self.server = FixtureServer(("127.0.0.1", 0), Handler)
        self.server.requests, self.server.files = [], ["nested/model-Q8_0.gguf", "model-Q4_K_M.gguf"]
        self.server.fail = self.server.bad_range = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.env = dict(os.environ, CLAP_CACHE=str(self.cache), HF_ENDPOINT=f"http://127.0.0.1:{self.server.server_port}")
        self.env.pop("HF_TOKEN", None)
        self.env.pop("SSL_CERT_FILE", None)

    def start_server(self, tls=False):
        server = FixtureServer(("127.0.0.1", 0), Handler)
        if tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(Path(self.tls.name) / "server.pem",
                                    Path(self.tls.name) / "server-key.pem")
            server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def use_https(self):
        server = self.start_server(tls=True)
        self.env["HF_ENDPOINT"] = f"https://127.0.0.1:{server.server_port}"
        self.env["SSL_CERT_FILE"] = str(Path(self.tls.name) / "ca.pem")
        return server

    def test_https_download(self):
        server = self.use_https()
        server.paginate = True
        path = Path(self.download("-hft", "private-token").stdout.strip())
        self.assertEqual(path.read_bytes(), DATA)
        self.assertEqual(path.name, "model-Q4_K_M.gguf")
        self.assertTrue(any("cursor=second" in p for p, _ in server.requests))
        self.assertTrue(any("/resolve/" in p for p, _ in server.requests))
        self.assertTrue(all(h.get("Authorization") == "Bearer private-token"
                            for _, h in server.requests))

    def test_https_untrusted_certificate_rejected(self):
        server = self.use_https()
        del self.env["SSL_CERT_FILE"]
        self.download(ok=False)
        self.assertEqual(server.requests, [])
        self.assertFalse((self.repo / "refs/main").exists())

    def test_https_hostname_mismatch_rejected(self):
        server = self.use_https()
        self.env["HF_ENDPOINT"] = f"https://localhost:{server.server_port}"
        self.download(ok=False)
        self.assertEqual(server.requests, [])
        self.assertFalse((self.repo / "refs/main").exists())

    def test_https_cross_origin_redirect_does_not_leak_token(self):
        server = self.use_https()
        other = self.start_server(tls=True)
        server.redirect = f"https://127.0.0.1:{other.server_port}/redirected"
        self.assertEqual(Path(self.download("-hft", "private-token").stdout.strip()).read_bytes(), DATA)
        self.assertTrue(server.requests)
        self.assertTrue(all(h.get("Authorization") == "Bearer private-token"
                            for _, h in server.requests))
        self.assertTrue(other.requests)
        self.assertTrue(all("Authorization" not in h for _, h in other.requests))

    def test_https_downgrade_rejected(self):
        server = self.use_https()
        server.redirect = f"http://127.0.0.1:{self.server.server_port}/redirected"
        p = self.download("-hft", "private-token", ok=False)
        self.assertIn("Refusing HTTPS downgrade", p.stderr)
        self.assertTrue(any("/resolve/" in p for p, _ in server.requests))
        self.assertEqual(self.server.requests, [])
        self.assertFalse((self.repo / "blobs" / OID).exists())
        self.assertFalse((self.repo / "refs/main").exists())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def run_cli(self, *args, ok=True, exe=CLI):
        p = subprocess.run([exe, *args], env=self.env, text=True, capture_output=True, timeout=15)
        if ok:
            self.assertEqual(p.returncode, 0, p.stderr)
        else:
            self.assertNotEqual(p.returncode, 0, p.stdout)
        return p

    def download(self, *args, **kw):
        return self.run_cli("download", "-hf", "test/model", *args, **kw)

    def test_default_cache_offline_and_token(self):
        p = self.download("-hft", "secret")
        path = Path(p.stdout.strip())
        self.assertEqual(path, self.repo / "snapshots" / COMMIT / "model-Q4_K_M.gguf")
        self.assertEqual(path.read_bytes(), DATA)
        self.assertEqual((self.repo / "refs/main").read_text().strip(), COMMIT)
        self.assertTrue((self.repo / "blobs" / OID).is_file())
        self.assertTrue(all(h.get("Authorization") == "Bearer secret" for _, h in self.server.requests))
        self.server.requests.clear()
        self.assertEqual(self.download("--offline").stdout, p.stdout)
        self.assertEqual(self.server.requests, [])
        self.download()
        self.assertFalse(any("/resolve/" in p for p, _ in self.server.requests))

    def symlink_or_skip(self, target, link, **kwargs):
        try:
            link.symlink_to(target, **kwargs)
        except OSError as error:
            if os.name == "nt" and error.winerror == 1314:
                self.skipTest("Windows symlink privilege unavailable")
            raise

    def test_windows_special_tree_paths_rejected(self):
        for name in ["C:/outside/model.gguf", "C:model.gguf", "nested/model.gguf:stream",
                     "//server/share/model.gguf", "\\\\?\\C:\\outside\\model.gguf",
                     "NUL.gguf", "nested/COM1/model.gguf", "nested/LPT9.gguf",
                     "nested./model.gguf", "nested /model.gguf", "model?.gguf",
                     "COM¹.gguf", "nested/LPT²/model.gguf", "conout$.gguf"]:
            with self.subTest(name=name):
                self.server.files = [name]
                self.server.requests.clear()
                p = self.download(ok=False)
                self.assertIn("Unsafe repository filename", p.stderr)
                self.assertFalse(any("/resolve/" in p for p, _ in self.server.requests))
        self.assertFalse((self.repo / "refs/main").exists())

    def test_windows_absolute_explicit_filename_rejected(self):
        p = self.download("-hff", "C:/outside/model.gguf", ok=False)
        self.assertIn("Unsafe repository filename", p.stderr)
        self.assertEqual(self.server.requests, [])

    def test_userprofile_default_cache(self):
        for key in ["CLAP_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME",
                    "XDG_CACHE_HOME", "HOME"]:
            self.env.pop(key, None)
        home = Path(self.tmp.name) / "windows-home"
        self.env["USERPROFILE"] = str(home)
        path = Path(self.download().stdout.strip())
        self.assertEqual(path, home / ".cache/huggingface/hub/models--test--model" /
                         "snapshots" / COMMIT / "model-Q4_K_M.gguf")
        self.assertEqual(path.read_bytes(), DATA)

    def test_interrupted_snapshot_publication_recovers(self):
        path = Path(self.download("-hff", "nested/model-Q8_0.gguf").stdout.strip())
        blob = self.repo / "blobs" / OID
        path.unlink()
        temporary = path.with_name(path.name + ".downloadInProgress")
        self.symlink_or_skip(os.path.relpath(blob, path.parent), temporary)
        self.server.requests.clear()
        self.assertEqual(Path(self.download("-hff", "nested/model-Q8_0.gguf").stdout.strip()), path)
        self.assertEqual(path.read_bytes(), DATA)
        self.assertEqual(blob.read_bytes(), DATA)
        self.assertFalse(temporary.is_symlink())
        self.assertFalse(any("/resolve/" in p for p, _ in self.server.requests))

    def test_interrupted_snapshot_hardlink_recovers(self):
        path = Path(self.download().stdout.strip())
        blob = self.repo / "blobs" / OID
        path.unlink()
        temporary = path.with_name(path.name + ".downloadInProgress")
        os.link(blob, temporary)
        self.server.requests.clear()
        self.assertEqual(Path(self.download().stdout.strip()), path)
        self.assertTrue(os.path.samefile(path, blob))
        self.assertFalse(temporary.exists())
        self.assertFalse(any("/resolve/" in p for p, _ in self.server.requests))

    def test_unrelated_snapshot_temporary_file_refused(self):
        path = Path(self.download().stdout.strip())
        path.unlink()
        temporary = path.with_name(path.name + ".downloadInProgress")
        temporary.write_bytes(DATA)
        self.assertIn("Unsafe temporary snapshot", self.download(ok=False).stderr)
        self.assertEqual(temporary.read_bytes(), DATA)
        self.assertFalse(path.exists())

    def test_malicious_snapshot_temporary_link_refused(self):
        path = Path(self.download().stdout.strip())
        blob = self.repo / "blobs" / OID
        outside = Path(self.tmp.name) / "outside.gguf"
        outside.write_bytes(DATA)
        other_blob = blob.with_name("b" * 64)
        other_blob.write_bytes(DATA)
        chain = self.repo / "chain"
        self.symlink_or_skip(blob, chain)
        path.unlink()
        temporary = path.with_name(path.name + ".downloadInProgress")
        for target in [outside, other_blob, chain]:
            with self.subTest(target=target):
                self.symlink_or_skip(target, temporary)
                self.assertIn("Unsafe", self.download(ok=False).stderr)
                self.assertTrue(temporary.is_symlink())
                self.assertFalse(path.exists())
                self.assertEqual(outside.read_bytes(), DATA)
                self.assertEqual(other_blob.read_bytes(), DATA)
                self.assertEqual(blob.read_bytes(), DATA)
                temporary.unlink()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux symlink failure injection")
    def test_snapshot_hardlink_fallback(self):
        library = Path(self.tmp.name) / "deny-symlink.so"
        source = '#include <errno.h>\nint symlink(const char *a, const char *b) { errno = EPERM; return -1; }\n'
        subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(library)],
                       input=source, text=True, check=True, capture_output=True)
        self.env["LD_PRELOAD"] = str(library)
        path = Path(self.download().stdout.strip())
        self.assertFalse(path.is_symlink())
        self.assertTrue(os.path.samefile(path, self.repo / "blobs" / OID))
        self.assertEqual(path.read_bytes(), DATA)
        self.assertEqual(Path(self.download("--offline").stdout.strip()), path)
        path.rename(path.with_name(path.name + ".downloadInProgress"))
        self.server.requests.clear()
        self.assertEqual(Path(self.download().stdout.strip()), path)
        self.assertTrue(os.path.samefile(path, self.repo / "blobs" / OID))
        self.assertFalse(any("/resolve/" in p for p, _ in self.server.requests))

    def test_quant_and_explicit_filename(self):
        p = self.run_cli("download", "-hf", "test/model:Q8_0")
        self.assertTrue(Path(p.stdout.strip()).as_posix().endswith("nested/model-Q8_0.gguf"))
        self.assertEqual(self.download("-hff", "nested/model-Q8_0.gguf").stdout, p.stdout)

    def test_resume_and_invalid_range(self):
        blob = self.repo / "blobs" / (OID + ".downloadInProgress")
        blob.parent.mkdir(parents=True)
        blob.write_bytes(DATA[:20])
        self.download()
        self.assertTrue(any(h.get("Range") == "bytes=20-" for _, h in self.server.requests))
        self.assertEqual(blob.with_name(OID).read_bytes(), DATA)

    def test_bad_range_not_published(self):
        blob = self.repo / "blobs" / (OID + ".downloadInProgress")
        blob.parent.mkdir(parents=True)
        blob.write_bytes(DATA[:20])
        self.server.bad_range = True
        self.download(ok=False)
        self.assertFalse(blob.with_name(OID).exists())

    def test_split_unsupported(self):
        self.server.files = ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]
        p = self.download(ok=False)
        self.assertIn("split", p.stderr.lower())
        self.assertFalse(any("/resolve/" in p for p, _ in self.server.requests))

    def test_cross_origin_redirect_does_not_leak_token(self):
        other = FixtureServer(("127.0.0.1", 0), Handler)
        other.requests = []
        thread = threading.Thread(target=other.serve_forever, daemon=True)
        thread.start()
        self.server.redirect = f"http://127.0.0.1:{other.server_port}/redirected"
        try:
            self.download("-hft", "private-token")
            self.assertTrue(other.requests)
            self.assertTrue(all("Authorization" not in h for _, h in other.requests))
        finally:
            other.shutdown()
            other.server_close()
            thread.join()

    def test_offline_missing(self):
        self.assertIn("offline", self.download("--offline", ok=False).stderr.lower())
        self.assertEqual(self.server.requests, [])

    def test_traversal_and_symlink(self):
        self.download("-hff", "../outside.gguf", ok=False)
        self.repo.mkdir(parents=True, exist_ok=True)
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        self.symlink_or_skip(outside, self.repo / "blobs", target_is_directory=True)
        self.download(ok=False)
        self.assertEqual(list(outside.iterdir()), [])

    def test_paginated_tree_prefers_q4(self):
        self.server.paginate = True
        self.assertTrue(self.download().stdout.strip().endswith("model-Q4_K_M.gguf"))
        self.assertTrue(any("cursor=second" in p for p, _ in self.server.requests))

    def test_interrupted_download_resumes(self):
        self.server.drop_once = True
        self.assertEqual(Path(self.download().stdout.strip()).read_bytes(), DATA)
        self.assertTrue(any(h.get("Range") == "bytes=30-" for _, h in self.server.requests))

    def test_server_ignoring_range_restarts_safely(self):
        partial = self.repo / "blobs" / (OID + ".downloadInProgress")
        partial.parent.mkdir(parents=True)
        partial.write_bytes(DATA[:20])
        self.server.ignore_range = True
        self.assertEqual(Path(self.download().stdout.strip()).read_bytes(), DATA)

    def test_corrupt_download_not_published(self):
        self.server.corrupt = True
        self.assertIn("SHA256", self.download(ok=False).stderr)
        self.assertFalse((self.repo / "blobs" / OID).exists())
        self.assertFalse((self.repo / "refs/main").exists())

    def test_cache_precedence(self):
        self.env["HF_HUB_CACHE"] = str(Path(self.tmp.name) / "hub")
        self.assertTrue(self.download().stdout.startswith(str(self.cache)))
        del self.env["CLAP_CACHE"]
        self.assertTrue(self.download().stdout.startswith(self.env["HF_HUB_CACHE"]))
        del self.env["HF_HUB_CACHE"]
        self.env["HUGGINGFACE_HUB_CACHE"] = str(Path(self.tmp.name) / "legacy")
        self.assertTrue(self.download().stdout.startswith(self.env["HUGGINGFACE_HUB_CACHE"]))
        del self.env["HUGGINGFACE_HUB_CACHE"]
        self.env["HF_HOME"] = str(Path(self.tmp.name) / "home")
        self.assertTrue(self.download().stdout.startswith(str(Path(self.env["HF_HOME"]) / "hub")))
        del self.env["HF_HOME"]
        self.env["XDG_CACHE_HOME"] = str(Path(self.tmp.name) / "xdg")
        self.assertTrue(self.download().stdout.startswith(str(Path(self.env["XDG_CACHE_HOME"]) / "huggingface/hub")))

    def test_bounded_retry(self):
        self.server.fail = True
        self.download(ok=False)
        count = sum("/resolve/" in p for p, _ in self.server.requests)
        self.assertGreaterEqual(count, 2)
        self.assertLessEqual(count, 4)
        self.assertFalse((self.repo / "blobs" / OID).exists())

    def test_model_cli_real_inference(self):
        from hf_model_fixture import tiny_clap
        self.server.data = tiny_clap()
        p = self.run_cli("embedding", "-hf", "test/model", "--text", "hello")
        values = json.loads(p.stdout)
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(sum(x*x for x in values), 1.0, places=5)
        path = self.repo / "snapshots" / COMMIT / "model-Q4_K_M.gguf"
        direct = self.run_cli("embedding", "--model", str(path), "--text", "hello")
        self.assertEqual(p.stdout, direct.stdout)
        self.server.requests.clear()
        offline = self.run_cli("embedding", "-hf", "test/model", "--offline", "--text", "hello")
        self.assertEqual(p.stdout, offline.stdout)
        self.assertEqual(self.server.requests, [])

if __name__ == "__main__":
    unittest.main()
