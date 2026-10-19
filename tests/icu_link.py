"""Bundled executables must not load system ICU libraries at runtime (POSIX)."""
import subprocess
import sys

command = ['otool', '-L'] if sys.platform == 'darwin' else ['ldd']
result = subprocess.run([*command, sys.argv[1]], text=True, capture_output=True)
assert result.returncode == 0, result.stdout + result.stderr
libraries = result.stdout.lower()
assert 'libicu' not in libraries, result.stdout
print('Executable has no ICU shared-library dependency')
