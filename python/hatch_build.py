"""Validate an externally copied runtime and mark its wheel as platform-specific."""

from pathlib import Path
import sys
import sysconfig

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        package = Path(self.root) / self.metadata.name.replace("-", "_")
        native = "clap.dll" if sys.platform == "win32" else (
            "libclap.dylib" if sys.platform == "darwin" else "libclap.so"
        )
        runtime = package / native
        if not runtime.is_file() or runtime.stat().st_size == 0:
            raise RuntimeError(
                f"Missing staged external runtime: {runtime}. "
                "Copy the externally compiled runtime's clap_cpp/ contents into "
                f"{package.name}/ before building a wheel. "
                "Python packaging does not compile native libraries."
            )
        notice = package / "licenses" / "clap-cpp.txt"
        if not notice.is_file() or notice.stat().st_size == 0:
            raise RuntimeError("Missing runtime notices; copy LICENSE to " + str(notice))
        platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        if sys.platform == "darwin" and "universal2" in platform:
            raise RuntimeError(
                "Cannot tag an unverified external runtime as universal2. "
                "Set _PYTHON_HOST_PLATFORM to a single-architecture macOS platform "
                "matching the staged runtime's architecture and deployment target "
                "(for example, macosx-11.0-arm64)."
            )
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform}"
