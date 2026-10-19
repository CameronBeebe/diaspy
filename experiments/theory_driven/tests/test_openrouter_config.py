"""Provider request-shape tests: offline optimizer tests alone cannot validate it."""
import pytest
from experiments.theory_driven.config import ModelConfig


@pytest.mark.parametrize("enabled", [False, True])
def test_openrouter_reasoning_is_object(enabled):
    config = ModelConfig(enable_reasoning=enabled)
    kwargs = config.get_litellm_kwargs("openrouter/google/gemma-4-26b-a4b-it:free")
    reasoning = kwargs["extra_body"]["reasoning"]
    assert isinstance(reasoning, dict)
    assert not isinstance(reasoning, bool)
    assert reasoning["enabled"] is enabled
    assert kwargs["extra_body"]["provider"] == {
        "allow_fallbacks": False, "require_parameters": True,
    }


def test_non_openrouter_does_not_get_reasoning_object():
    assert "extra_body" not in ModelConfig().get_litellm_kwargs("openai/example")
