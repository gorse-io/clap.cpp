# clap.cpp

<img width=160 src="assets/clap.png"/>

[![CI](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml)
[![GitHub Release](https://img.shields.io/github/v/release/gorse-io/clap.cpp)](https://github.com/gorse-io/clap.cpp/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/clap-cpp)](https://pypi.org/project/clap-cpp/)
[![npm](https://img.shields.io/npm/v/clap-cpp)](https://www.npmjs.com/package/clap-cpp)
[![crates.io](https://img.shields.io/crates/v/clap-cpp)](https://crates.io/crates/clap-cpp)

C++20/ggml implementations of audio embedding inference features:

- Supports WAV, MP3, and FLAC audio input.
- Automatically resamples audio to the model's required.
- Automatically mixes stereo and multichannel audio down to mono.

## Supported models

| Model | Text | Audio | Embedding dimensions | Sample rate |
| --- | --- | --- | --- | --- |
| [CLAP](https://huggingface.co/laion/larger_clap_music) | ✅ | ✅ | 512 | 48 kHz |
| [CLaMP3](https://github.com/sanderwood/clamp3) | ✅ | ✅ | 768 | 24 kHz |
| [MERT v1 95M](https://huggingface.co/m-a-p/MERT-v1-95M) | ❌ | ✅ | 768 | 24 kHz |
| [MERT v1 330M](https://huggingface.co/m-a-p/MERT-v1-330M) | ❌ | ✅ | 1,024 | 24 kHz |
| [MERT v2 FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) / [30s](https://huggingface.co/m-a-p/MERT-v2-30s) | ❌ | ✅ | 1,024 | 24 kHz |
| [MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) | ❌ | ✅ | 1,024 | 24 kHz |
| [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | ✅ | ✅ | 512 | 24 kHz |

GGUF models are available at [Audio Embedding GGUF](https://huggingface.co/collections/gorse-io/audio-embedding-gguf) collection.

## Bindings

| Binding | CPU | CUDA | Vulkan | Install |
| --- | --- | --- | --- | --- |
| [Python](python) | ✅ | ❌ | ❌ | `pip3 install clap-cpp` |
| [Rust](rust) | ✅ | ❌ | ❌ | `cargo install clap-cpp` |
| [Node.js](node) | ✅ | ❌ | ❌ | `npm install clap-cpp` |

## Usage

## Embeddings command

Load a converted GGUF and generate real native text or audio embeddings:

```sh
clap embedding --model model.gguf --text 'A jazz quartet.'
clap embedding --model model.gguf --audio music.flac
```

## Embeddings server

Start the server with a GGUF supporting the requested modality, such as CLAP,
CLaMP3 or MuQ-MuLan (default listen address: `127.0.0.1:8080`):

```sh
clap serve --model model.gguf --host 127.0.0.1 --port 8080
```

The model is loaded once at startup. Its API identifier is the GGUF **filename**
(`model.gguf` in this example), not its full path or a Hugging Face repository ID.

HTTP handlers validate JSON and decode audio before submitting owned inputs to
an inference FIFO. One dedicated worker owns the model, including tokenization
and destruction. All inputs in one HTTP request run together, in input order;
this is request scheduling, **not** true model batching or parallel inference.

`--queue-capacity N` sets the maximum number of **waiting inference requests**
(default: `64`, positive integer), excluding the one running request. A full or
stopping queue returns HTTP **503** with the existing JSON `error` shape and
`type: "server_error"`; callers can retry with backoff. This limit is not an HTTP
connection limit or a byte budget: HTTP parsing/decoding and the HTTP library's
connection backlog are separate, and each request may contain up to 1024 inputs
within the existing 32 MiB body limit.

Fatal transport failures (such as TCP resets) are checked while preparing inputs
and waiting (every 25 ms). A TCP FIN alone is **not** cancellation: it cannot
portably distinguish a client that closed completely from one that only closed
its write side and is still reading the response. Gracefully disconnected
requests may therefore finish inference. Explicitly abandoning a scheduler
waiter removes queued work immediately; detected fatal transport failures do
the same. Running work stops between inputs (or after the current C API call),
not inside a model operation. SIGINT/SIGTERM
request orderly shutdown: stop accepting connections, reject new inference,
cancel queued requests, let any current model call finish, and join the worker
before destroying the model on its owner thread. Shutdown can therefore take as
long as an in-flight model call and HTTP handler cleanup. Forced process kills
(including Windows `TerminateProcess`) cannot run this cleanup.

Send an OpenAI-compatible text embeddings request:

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"model.gguf","input":["hello","world"],"encoding_format":"float"}'
```

Audio inputs use one `input_audio` content block per item. Encode the complete
WAV, MP3 or FLAC file as standard base64 (no whitespace or data-URL prefix):

```sh
python3 -c 'import base64,json,sys; print(json.dumps({"input":[{"content":[{"type":"input_audio","input_audio":{"data":base64.b64encode(open(sys.argv[1],"rb").read()).decode()}}]}]}))' music.flac |
  curl http://127.0.0.1:8080/v1/embeddings \
    -H 'Content-Type: application/json' --data-binary @-
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Third-party dependencies retain their respective licenses.
