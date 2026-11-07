#!/usr/bin/env python3
"""Default stdlib CLAP2 conversion-contract + native invalid-model regressions.

Usage: python clap2_invalid.py BUILD OUTPUT
Runs analytical F32/F16 public API smoke before negative loader checks.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys

from clap2_fixture import generate, model_data, write_gguf


def main():
    build, output = map(Path, sys.argv[1:])
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from conversion.clap2 import clap2_inputs, expected_tensor_shapes, validate_config, validate_tokenizer
    metadata, tensors = generate(output)
    config, processor, tokenizer, _ = model_data()
    validate_config(config, processor)
    validate_tokenizer(tokenizer, config)
    assert expected_tensor_shapes(config) == {name: shape for name, (shape, _) in tensors.items()}
    for name, data in (("config.json", config), ("processor_config.json", processor), ("tokenizer.json", tokenizer)):
        (output / name).write_text(json.dumps(data))
    # Metadata discovery is independent of safetensors payload parsing. The native
    # smoke below exercises a real complete GGUF, not these placeholder bytes.
    (output / "model.safetensors").write_bytes(b"")
    _, converted_metadata, _ = clap2_inputs(output)
    for field in ("clap2.config", "clap2.processor_config", "clap2.tokenizer"):
        assert json.loads(metadata[field]) == json.loads(converted_metadata[field])

    for dtype in ("f32", "f16"):
        subprocess.run([str(build / "clap2-model-test"), str(output / f"model-{dtype}.gguf"),
                        str(output / "fixture.json")], check=True)

    invalid = []
    def config_case(label, field, value, section="audio_config"):
        changed = copy.deepcopy(config)
        changed[section][field] = value
        invalid.append((label, changed, processor, tokenizer))
    config_case("zero heads", "num_attention_heads", 0)
    config_case("fractional width", "hidden_size", 8.5)
    config_case("unsupported activation", "hidden_act", "gelu")
    config_case("unsupported audio rope", "rope_parameters", dict(rope_type="default", rope_theta=10000.0))
    config_case("unsupported text rope", "rope_parameters", dict(rope_type="axial", rope_theta=10000.0), "text_config")
    config_case("audio QK normalization", "use_qk_norm", True)
    config_case("text QK normalization", "use_qk_norm", False, "text_config")
    config_case("out-of-range special id", "bos_token_id", 260, "text_config")
    changed = copy.deepcopy(processor)
    changed["feature_extractor"]["feature_size"] = 64
    invalid.append(("wrong mel height", config, changed, tokenizer))
    changed = copy.deepcopy(processor)
    changed["feature_extractor"]["max_num_patches"] = 1504
    invalid.append(("training token budget is not a crop policy", config, changed, tokenizer))
    changed = copy.deepcopy(tokenizer)
    changed["model"]["type"] = "WordPiece"
    invalid.append(("wrong tokenizer model", config, processor, changed))
    changed = copy.deepcopy(tokenizer)
    changed["post_processor"]["single"][0]["SpecialToken"]["id"] = "<|eos|>"
    invalid.append(("wrong special template", config, processor, changed))

    failures = []
    for index, (label, c, p, t) in enumerate(invalid):
        try:
            validate_config(c, p)
            validate_tokenizer(t, c)
        except (ValueError, KeyError, TypeError):
            pass
        else:
            raise AssertionError(f"converter accepted {label}")
        bad_metadata = dict(metadata, **{"clap2.config": json.dumps(c),
                                       "clap2.processor_config": json.dumps(p),
                                       "clap2.tokenizer": json.dumps(t)})
        path = output / f"invalid-{index}.gguf"
        write_gguf(path, bad_metadata, tensors)
        result = subprocess.run([str(build / "clap2-model-test"), str(path), str(output / "fixture.json")],
                                text=True, capture_output=True)
        if result.returncode == 0:
            failures.append(f"native accepted {label}")
            continue
        assert any(word in result.stderr.lower() for word in ("clap2", "token", "bpe", "template")), (label, result.stderr)
        print(f"Rejected {label}: {result.stderr.strip()}")
    for label in ("missing", "wrong-shape"):
        bad_tensors = dict(tensors)
        target = "text_model.encoder.layers.0.self_attn.qkv_proj.weight"
        if label == "missing":
            del bad_tensors[target]
        else:
            bad_tensors[target] = ((23, 8), [0.0] * (23 * 8))
        path = output / f"invalid-tensor-{label}.gguf"
        write_gguf(path, metadata, bad_tensors)
        result = subprocess.run([str(build / "clap2-model-test"), str(path), str(output / "fixture.json")],
                                text=True, capture_output=True)
        assert result.returncode != 0 and ("tensor" in result.stderr.lower() or "qkv_proj" in result.stderr), result.stderr
        print(f"Rejected {label} tensor: {result.stderr.strip()}")
    assert not failures, "\n".join(failures)
    print("CLAP2 analytical F32/F16 smoke and conversion/native invalid contracts passed")


if __name__ == "__main__":
    main()
