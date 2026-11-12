"""Optional HF oracle: python gemma_tokenizer_parity.py TOKENIZER_JSON OUT_DIR.

Only this test needs Python/tokenizers; native inference does not. Supply
Google embeddinggemma-2/tokenizer.json at revision
26711154b641697c443124e316f3d21587f9ff5d. Then run the native
clap-gemma-tokenizer-test OUT_DIR/metadata.json OUT_DIR/cases.json.
Primary sources: llama.cpp src/llama-vocab.cpp (gemma4), conversion/gemma.py;
HF google/embeddinggemma-2 tokenizer.json (BPE / Replace normalizer).
"""
import hashlib
import json
from pathlib import Path
import random
import sys
from tokenizers import Tokenizer

source, output = Path(sys.argv[1]), Path(sys.argv[2])
raw = source.read_bytes()
if hashlib.sha256(raw).hexdigest() != "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4":
    raise ValueError("Expected pinned google/embeddinggemma-2 tokenizer.json")
output.mkdir(parents=True, exist_ok=True)
j = json.loads(raw)
t = Tokenizer.from_file(str(source))
vocab = j["model"]["vocab"]
tokens = [""] * len(vocab)
for token, idx in vocab.items():
    tokens[idx] = token
types = [1] * len(tokens)
for token in j["added_tokens"]:
    types[token["id"]] = 3 if token["special"] else 4
for idx, token in enumerate(tokens):
    if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
        types[idx] = 6
metadata = {
    "tokenizer.ggml.model": "gemma4",
    "tokenizer.ggml.tokens": tokens,
    "tokenizer.ggml.token_type": types,
    "tokenizer.ggml.merges": [" ".join(m) if isinstance(m, list) else m for m in j["model"]["merges"]],
    "tokenizer.ggml.bos_token_id": vocab["<bos>"],
    "tokenizer.ggml.eos_token_id": vocab["<eos>"],
    "tokenizer.ggml.unknown_token_id": vocab["<unk>"],
    "tokenizer.ggml.add_bos_token": True,
    "tokenizer.ggml.add_eos_token": True,
    "tokenizer.ggml.add_space_prefix": False,
}
texts = ["", "hello world", " hello", "  hello  world  ", "Hello, WORLD! 12345", "e\u0301 é Ａ Straße ﬃ", "中文 日本語 한국어", "🙂🚀🧑🏽‍💻", "\t\n\r\n", "a\u00a0b\u2003c\u200bd", "a\x00b\x01c", "<bos>hello<eos>", "<unk><pad><mask>"]
texts += ["a" + "\n" * n + "b" for n in range(1, 100)]
texts += ["\n" * n for n in range(1, 100)]
texts += ["\t" * n + " " * n for n in range(1, 65)]
texts += [x["content"] + " hello " + x["content"] for x in j["added_tokens"]]
rng = random.Random(17)
alphabet = "abcABCéé▁\n\t\r 123!?<>中🙂\u0301\u200d\u00a0"
texts += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 120))) for _ in range(500)]
(output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
(output / "cases.json").write_text(json.dumps([{"text": s, "ids": t.encode(s).ids} for s in texts], ensure_ascii=False), encoding="utf-8")
print(f"Generated {len(texts)} HF tokenizer parity cases")
