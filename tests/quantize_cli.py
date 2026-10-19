"""Quantize subcommand help and argument handling; standard library only."""
import subprocess
import sys

cli = sys.argv[1]


def run(*args):
    return subprocess.run([cli, *args], capture_output=True, text=True)


help_result = run('quantize', '--help')
assert help_result.returncode == 0, help_result.stderr
assert 'INPUT.gguf' in help_result.stdout and 'OUTPUT.gguf' in help_result.stdout
assert 'Q8_0' in help_result.stdout
assert 'quantize' in run('--help').stdout
assert run('quantize', '-h').returncode == 0
for args in [(), ('input.gguf',), ('a', 'b', 'c'), ('--unknown',)]:
    result = run('quantize', *args)
    assert result.returncode != 0, args
    assert result.stderr, args
result = run('quantize', 'missing-input.gguf', 'unused-output.gguf')
assert result.returncode != 0 and result.stderr
print('Quantize subcommand help and argument handling passed')
