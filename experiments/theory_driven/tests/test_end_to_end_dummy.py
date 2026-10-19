"""CLI dry-run does not construct LMs, load datasets, or run optimizers."""
from experiments.theory_driven import run_staged


def test_cli_dry_run_never_constructs_harness(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run constructed a harness")
    monkeypatch.setattr(run_staged, "ExperimentHarness", forbidden)
    assert run_staged.main(["--stage", "0", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"cost_cap": 1.0' in output
    assert '"max_cost": null' in output
    assert "openrouter/qwen/qwen3.5-9b" in output
    assert '"task"' in output and '"reflection"' in output and '"critic"' in output
