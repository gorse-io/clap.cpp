"""Architecture fixtures/adapters for the common model contract runner.

Keep optional Torch/Transformers oracle imports out of this registry. Adding an
architecture means supplying case data here, not another CLI test entry point.
"""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Architecture:
    module: str
    suites: tuple
    fixture_suites: tuple = ()
    build_suites: tuple = ()

    def cases(self, suite, context):
        return getattr(import_module(f"model_architectures.{self.module}"), suite)(context)


ARCHITECTURES = {
    "clap": Architecture("clap", ("cli",), ("cli",)),
    "clap2": Architecture("clap2", ("invalid",), ("invalid",), ("invalid",)),
    "clamp3": Architecture("clamp3", ("cli", "invalid"), ("cli", "invalid")),
    "mert": Architecture("mert", ("invalid",)),
    "mert2": Architecture("mert2", ("cli",), ("cli",)),
}
