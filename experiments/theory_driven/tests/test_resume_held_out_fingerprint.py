"""Resume must distinguish the held-out survival sampling configuration."""

import json
from dataclasses import replace

import pytest

from experiments.theory_driven.config import ExperimentConfig
from experiments.theory_driven.config.resume import completed_result, fingerprint


@pytest.mark.parametrize("variant", ["B@veto", "M@strict"])
@pytest.mark.parametrize("sample_count", [0, 1, 5])
def test_resume_rejects_results_with_different_held_out_samples(tmp_path, variant, sample_count):
    config = ExperimentConfig()
    stage = replace(config.get_stage_config(0), held_out_regression_samples=sample_count)
    original = fingerprint(config, stage, "hotpotqa", variant, 100, 0)
    row = {
        "benchmark": "hotpotqa", "arm": variant, "budget": 100, "seed": 0,
        "config_hash": original, "status": "success", "test_mean": 0.5,
        "test_size": stage.test_size, "test_scores": [0.5] * stage.test_size,
        "budget_stats": {"total_cost": 0.01},
    }
    (tmp_path / "completed.json").write_text(json.dumps(row))
    assert completed_result(tmp_path, "hotpotqa", variant, 100, 0, original, stage.test_size)

    changed = replace(stage, held_out_regression_samples=sample_count + 1)
    changed_hash = fingerprint(config, changed, "hotpotqa", variant, 100, 0)

    assert completed_result(
        tmp_path, "hotpotqa", variant, 100, 0, changed_hash, stage.test_size,
    ) is None
    assert changed_hash != original
    assert fingerprint(config, stage, "hotpotqa", variant, 100, 0) == original
