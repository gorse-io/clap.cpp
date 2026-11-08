#!/usr/bin/env python3
"""Run shared native model contracts, selected by architecture and suite.

Like llama.cpp's architecture/tokenizer runners, CTest invokes one entry point
with architecture-specific fixtures. No external model downloads are performed.
"""
import argparse
from pathlib import Path
import sys
import tempfile

from model_architectures import ARCHITECTURES
from model_contract import Context, run_cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHITECTURES)
    parser.add_argument("--suite", choices=("cli", "invalid"))
    parser.add_argument("--cli", type=Path, help="actual clap executable (including platform suffix)")
    parser.add_argument("--build", type=Path, help="directory containing native oracle executables")
    parser.add_argument("--fixture", type=Path, help="generated architecture fixture directory")
    parser.add_argument("--case", help="run cases whose name contains this substring; no matches is an error")
    parser.add_argument("--list", action="store_true", help="list supported architecture/suite pairs without fixtures")
    args = parser.parse_args(argv)
    if args.list:
        for name, adapter in ARCHITECTURES.items():
            for suite in adapter.suites:
                print(f"{name}/{suite}")
        return 0
    if not args.arch or not args.suite:
        parser.error("--arch and --suite are required unless --list is used")
    adapter = ARCHITECTURES[args.arch]
    if args.suite not in adapter.suites:
        parser.error(f"{args.arch} does not support suite {args.suite}; supported: {', '.join(adapter.suites)}")
    if args.suite in adapter.fixture_suites and args.fixture is None:
        parser.error(f"{args.arch}/{args.suite} requires --fixture")
    if args.suite in adapter.build_suites:
        if args.build is None:
            parser.error(f"{args.arch}/{args.suite} requires --build")
    elif args.cli is None:
        parser.error(f"{args.arch}/{args.suite} requires --cli")
    # Scratch output for metadata-only fixtures is isolated even under parallel CTest.
    with tempfile.TemporaryDirectory(prefix=f"model-{args.arch}-") as work:
        context = Context(args.cli.resolve() if args.cli else None,
                          args.build.resolve() if args.build else None,
                          args.fixture.resolve() if args.fixture else None, Path(work))
        try:
            run_cases(args.arch, args.suite, adapter.cases(args.suite, context), args.case)
        except Exception as error:
            print(f"FAIL {args.arch}/{args.suite}: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
