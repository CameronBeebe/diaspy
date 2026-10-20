"""Distribution contracts, exercised without importing from the checkout."""
from email.parser import Parser
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    output = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(output)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return next(output.glob("*.whl"))


def test_wheel_ships_existing_experiment_namespace_and_offline_fixture(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "experiments/theory_driven/run_staged.py" in names
    assert "experiments/theory_driven/tests/fixtures/hotpotqa.json" in names
    assert "experiments/theory_driven/verification/pyproject.toml" in names
    assert not any("/results/" in name or "/notebooks/" in name for name in names)


def test_wheel_declares_tested_runtime_and_complete_dev_extra(wheel):
    from packaging.requirements import Requirement

    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
    from packaging.specifiers import SpecifierSet

    assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(">=3.12,<3.13")
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    runtime = {r.name: str(r.specifier) for r in requirements if r.marker is None}
    assert runtime["dspy"] == "==3.4.1"
    assert runtime["gepa"] == "==0.1.4"
    assert {"datasets", "rank-bm25", "numpy", "pydantic", "aiohttp"} <= runtime.keys()
    dev = {r.name for r in requirements if r.marker and r.marker.evaluate({"extra": "dev"})}
    assert {"pytest", "pytest-socket", "pytest-timeout", "ruff", "pyright", "build"} <= dev


@pytest.fixture(scope="module")
def unpacked_wheel(wheel, tmp_path_factory):
    directory = tmp_path_factory.mktemp("installed")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(directory)
    return directory


def run_offline_cli(unpacked_wheel, tmp_path, args, module="diaspy.cli"):
    env = {k: v for k, v in os.environ.items()
           if not k.endswith(("API_KEY", "API_TOKEN")) and k != "LITELLM_LOCAL_MODEL_COST_MAP"}
    env["PYTHONPATH"] = str(unpacked_wheel)
    code = """
import runpy
import socket
import sys
attempts = []
def forbidden(*args, **kwargs):
    attempts.append(args)
    raise AssertionError('network access during offline CLI')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
sys.argv = [sys.argv[1], *sys.argv[2:]]
try:
    runpy.run_module(sys.argv[0], run_name='__main__')
finally:
    assert not attempts, attempts
"""
    return subprocess.run([sys.executable, "-c", code, module, *args],
                          cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)


@pytest.mark.parametrize("args", [[], ["--help"], ["experiment", "--help"], ["legacy", "--help"]])
def test_installed_cli_help_needs_no_key_or_network(unpacked_wheel, tmp_path, args):
    result = run_offline_cli(unpacked_wheel, tmp_path, args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


def test_installed_cli_dry_run_is_existing_stage_plan(unpacked_wheel, tmp_path):
    result = run_offline_cli(unpacked_wheel, tmp_path, ["experiment", "--stage", "0", "--dry-run"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"total_runs": 4' in result.stdout
    assert "no LLM calls" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("args", [["legacy"], ["experiment", "--stage", "0"]])
def test_installed_cli_live_execution_requires_explicit_opt_in(unpacked_wheel, tmp_path, args):
    result = run_offline_cli(unpacked_wheel, tmp_path, args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--confirm" in result.stderr


def test_importing_package_is_lazy_and_preserves_public_modules(unpacked_wheel, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(unpacked_wheel), "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    result = subprocess.run([sys.executable, "-c", """
import sys
import diaspy
assert 'dspy' not in sys.modules, 'help imports the model stack eagerly'
from diaspy import agents, responders, signatures, utils
import diaspy.agents
assert agents is diaspy.agents
assert all((responders, signatures, utils))
try:
    diaspy.not_a_public_attribute
except AttributeError:
    pass
else:
    raise AssertionError('missing attribute did not raise')
"""], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_uses_shipped_configuration_not_ambient_checkout(monkeypatch):
    from experiments.theory_driven.preflight import PreFlightGate
    import experiments.theory_driven.preflight as preflight

    directory = Path(preflight.__file__).parent
    gate = PreFlightGate(directory)
    calls = {}
    monkeypatch.setattr(gate, "_run", lambda name, args: calls.setdefault(name, args) is not None)
    assert gate.run_all_checks(0)
    for name, flag in [("ruff", "--config"), ("pyright", "--project"), ("offline_tests", "-c")]:
        assert flag in calls[name]
        config = Path(calls[name][calls[name].index(flag) + 1])
        assert config.is_file()
        assert config.is_relative_to(directory)
