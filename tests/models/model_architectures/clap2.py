"""Analytical CLAP2 fixtures and conversion/native rejection case data."""
import copy
import json
from pathlib import Path
import sys

from clap2_fixture import generate, model_data, write_gguf
from model_contract import Check, Command, Diagnostic


def invalid(ctx):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from conversion.clap2 import clap2_inputs, expected_tensor_shapes, validate_config, validate_tokenizer

    output = ctx.fixture
    metadata, tensors = generate(output)
    config, processor, tokenizer, _ = model_data()

    def conversion_contract():
        validate_config(config, processor)
        validate_tokenizer(tokenizer, config)
        assert expected_tensor_shapes(config) == {name: shape for name, (shape, _) in tensors.items()}
        for name, data in (("config.json", config), ("processor_config.json", processor), ("tokenizer.json", tokenizer)):
            (output / name).write_text(json.dumps(data))
        # Metadata discovery is independent of safetensors payload parsing. The
        # native smoke below uses real complete GGUFs, not these placeholder bytes.
        (output / "model.safetensors").write_bytes(b"")
        _, converted_metadata, _ = clap2_inputs(output)
        for field in ("clap2.config", "clap2.processor_config", "clap2.tokenizer"):
            assert json.loads(metadata[field]) == json.loads(converted_metadata[field])

    yield Check("conversion-metadata", conversion_contract)
    # C API calls remain visible in clap2-model-test.cpp. No method facade.
    executable = ctx.build / ("clap2-model-test.exe" if sys.platform == "win32" else "clap2-model-test")

    def native(path):
        return [executable, path, output / "fixture.json"]

    for dtype in ("f32", "f16"):
        yield Command(f"analytical-{dtype}", native(output / f"model-{dtype}.gguf"))

    cases = []

    def config_case(label, field, value, section="audio_config"):
        changed = copy.deepcopy(config)
        changed[section][field] = value
        cases.append((label, changed, processor, tokenizer))

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
    cases.append(("wrong mel height", config, changed, tokenizer))
    changed = copy.deepcopy(processor)
    changed["feature_extractor"]["max_num_patches"] = 1504
    cases.append(("training token budget is not a crop policy", config, changed, tokenizer))
    changed = copy.deepcopy(tokenizer)
    changed["model"]["type"] = "WordPiece"
    cases.append(("wrong tokenizer model", config, processor, changed))
    changed = copy.deepcopy(tokenizer)
    changed["post_processor"]["single"][0]["SpecialToken"]["id"] = "<|eos|>"
    cases.append(("wrong special template", config, processor, changed))

    for index, (label, c, p, t) in enumerate(cases):
        def reject(c=c, p=p, t=t, label=label):
            try:
                validate_config(c, p)
                validate_tokenizer(t, c)
            except (ValueError, KeyError, TypeError):
                return
            raise AssertionError(f"converter accepted {label}")

        yield Check(f"converter-{label}", reject)
        bad_metadata = dict(metadata, **{"clap2.config": json.dumps(c),
                                       "clap2.processor_config": json.dumps(p),
                                       "clap2.tokenizer": json.dumps(t)})
        path = output / f"invalid-{index}.gguf"
        write_gguf(path, bad_metadata, tensors)
        yield Command(f"native-{label}", native(path), success=False,
                      diagnostics=("clap2", "token", "bpe", "template"),
                      diagnostic_any=True, diagnostic_lower=True)
    for label in ("missing", "wrong-shape"):
        bad_tensors = dict(tensors)
        target = "text_model.encoder.layers.0.self_attn.qkv_proj.weight"
        if label == "missing":
            del bad_tensors[target]
        else:
            bad_tensors[target] = ((23, 8), [0.0] * (23 * 8))
        path = output / f"invalid-tensor-{label}.gguf"
        write_gguf(path, metadata, bad_tensors)
        yield Command(f"tensor-{label}", native(path), success=False,
                      diagnostics=(Diagnostic("tensor", ignore_case=True), "qkv_proj"), diagnostic_any=True)
