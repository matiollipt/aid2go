"""Help and completion must work without initializing scientific libraries."""

import subprocess
import sys


def test_all_help_pages_and_completion_without_scientific_imports():
    # A fresh process catches imports even when other tests have loaded torch.
    result = subprocess.run(
        [sys.executable, "-c", '''
import importlib.abc
import sys

class BlockScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "torch_geometric", "numpy", "pandas", "matplotlib",
            "seaborn", "transformers", "mlflow", "h5py", "sklearn", "scipy",
        }:
            raise AssertionError(f"CLI help imported {fullname}")

sys.meta_path.insert(0, BlockScientificImports())

import os
from typer.main import get_command
from typer.testing import CliRunner
from aid2go.cli.app import app

runner = CliRunner()
os.environ["_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION"] = "1"

def check_help(command, args):
    result = runner.invoke(app, args + ["--help"])
    assert result.exit_code == 0, (args, result.output, result.exception)
    assert "Usage:" in result.output
    assert not result.stderr, (args, result.stderr)
    for name, child in getattr(command, "commands", {}).items():
        check_help(child, args + [name])

check_help(get_command(app), [])
result = runner.invoke(app, ["--show-completion", "bash"])
assert result.exit_code == 0, (result.output, result.exception)
assert "complete" in result.output
assert not result.stderr
'''],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
