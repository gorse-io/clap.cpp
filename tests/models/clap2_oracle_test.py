#!/usr/bin/env python3
"""Optional oracle-tooling unit tests; stdlib only, no model imports/downloads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import clap2_parity as oracle


class OracleConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.assets = Path(self.directory.name)
        self.hf = {
            "projection_dim": 224,
            "audio_config": dict(hidden_size=224, intermediate_size=640, num_attention_heads=4,
                                 num_hidden_layers=12, layerscale_value=1e-5),
            "text_config": dict(hidden_size=224, intermediate_size=640, num_attention_heads=4,
                                num_hidden_layers=12, max_position_embeddings=256),
        }
        self.upstream = {
            "embed_dim": 224,
            "audio_cfg": {"naflexvit_cfg": dict(embed_dim=224, depth=12, num_heads=4,
                                               mlp_ratio=640 / 224)},
            "text_cfg": dict(width=224, layers=12, heads=4, mlp_ratio=640 / 224, context_length=256),
        }
        self.identity = {"repo": "laion/clap2-xxs-audioxl-s1B", "revision": "a" * 40}
        self.write("config.json", self.hf)
        self.write("open_clip_config.json", {"model_cfg": self.upstream})
        self.write("identity.json", self.identity)

    def write(self, name, value):
        (self.assets / name).write_text(json.dumps(value))

    def test_published_config_and_identity_are_asset_specific(self):
        config, upstream, identity = oracle.load_oracle_assets(self.assets)
        self.assertEqual(config, self.hf)
        self.assertEqual(upstream, self.upstream)
        self.assertEqual(identity, self.identity)
        self.assertEqual(upstream["audio_cfg"]["naflexvit_cfg"]["embed_dim"], 224)

    def test_legacy_manifest_supplies_exact_identity(self):
        (self.assets / "identity.json").unlink()
        self.write("hf-manifest.json", {"id": self.identity["repo"], "sha": self.identity["revision"]})
        _, _, identity = oracle.load_oracle_assets(self.assets)
        self.assertEqual(identity, self.identity)

    def test_missing_identity_is_not_silently_labeled_large(self):
        (self.assets / "identity.json").unlink()
        with self.assertRaises(FileNotFoundError):
            oracle.load_oracle_assets(self.assets)

    def test_invalid_revision_is_rejected(self):
        self.write("identity.json", dict(self.identity, revision="main"))
        with self.assertRaisesRegex(ValueError, "revision"):
            oracle.load_oracle_assets(self.assets)

    def test_mismatched_official_config_is_rejected(self):
        self.upstream["text_cfg"]["layers"] = 11
        self.write("open_clip_config.json", {"model_cfg": self.upstream})
        with self.assertRaisesRegex(ValueError, "text.*layers"):
            oracle.load_oracle_assets(self.assets)

    def test_xxs_profile_preserves_head56_and_noninteger_mlp(self):
        original = copy.deepcopy((self.hf, self.upstream))
        config, upstream = oracle.tiny_configs(self.hf, self.upstream, "xxs-head56")
        for branch in ("audio", "text"):
            cfg = config[branch + "_config"]
            self.assertEqual(cfg["hidden_size"] // cfg["num_attention_heads"], 56)
            self.assertEqual(cfg["intermediate_size"], 160)
            self.assertEqual(cfg["num_hidden_layers"], 2)
        self.assertEqual(int(upstream["text_cfg"]["width"] * upstream["text_cfg"]["mlp_ratio"]), 160)
        self.assertEqual(config["projection_dim"], 56)
        self.assertEqual((self.hf, self.upstream), original)

    def test_asymmetric_profile_exercises_distinct_widths_and_depths(self):
        config, upstream = oracle.tiny_configs(self.hf, self.upstream, "asymmetric-head64")
        audio, text = config["audio_config"], config["text_config"]
        self.assertNotEqual(audio["hidden_size"], text["hidden_size"])
        self.assertNotEqual(audio["num_hidden_layers"], text["num_hidden_layers"])
        self.assertEqual(audio["hidden_size"] // audio["num_attention_heads"], 64)
        self.assertEqual(text["hidden_size"] // text["num_attention_heads"], 64)
        self.assertEqual(upstream["text_cfg"]["layers"], text["num_hidden_layers"])
        self.assertEqual(upstream["audio_cfg"]["naflexvit_cfg"]["depth"], audio["num_hidden_layers"])

    def test_default_profile_preserves_original_tiny_dimensions(self):
        config, upstream = oracle.tiny_configs(self.hf, self.upstream, "tiny")
        self.assertEqual(config["projection_dim"], 12)
        for branch in ("audio", "text"):
            cfg = config[branch + "_config"]
            self.assertEqual((cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"],
                              cfg["num_hidden_layers"]), (16, 32, 2, 2))
        self.assertEqual(upstream["text_cfg"]["context_length"], 32)


if __name__ == "__main__":
    unittest.main()
