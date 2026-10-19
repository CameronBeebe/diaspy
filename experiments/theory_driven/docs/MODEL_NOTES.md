# Model Configuration Notes

## Primary Model: Qwen3.5-9B

**LiteLLM string**: `openrouter/qwen/qwen3.5-9b`  
**API key**: `OPENROUTER_API_KEY`  
**Reasoning**: Disabled via OpenRouter's `reasoning=False` parameter (passed through LiteLLM's `extra_body`)

### Why Qwen3.5-9B?

Qwen3-8B (used in GEPA paper) is **no longer available pay-per-token** on OpenRouter. Qwen3.5-9B is the closest successor with similar capabilities.

### Non-Comparable Baselines

**Important**: GEPA paper likely ran Qwen3-8B in **thinking mode** with:
- Temperature: 0.6
- Top-p: 0.95
- Top-k: 20

Our experiments disable reasoning/thinking tokens for cost efficiency and interpretability. This means:

1. Our baseline numbers will **NOT** match GEPA paper's Qwen3-8B baselines
2. We're testing a different hypothesis: theory-driven critique at low budgets, not reproducing GEPA exactly
3. The comparison remains valid: all our arms use the same model settings

### Reasoning Control

Per OpenRouter documentation, reasoning-capable models (including Qwen3.5) support a `reasoning` parameter:

```python
# Disable reasoning/thinking tokens
litellm_kwargs = {
    "extra_body": {"reasoning": False}
}
```

This is configured in `ModelConfig.get_litellm_kwargs()` and applied when `enable_reasoning=False` (default).

## Alternative Models

### GPT-4.1-mini (Baseline Check)

**LiteLLM string**: `openrouter/openai/gpt-4.1-mini`  
**API key**: `OPENROUTER_API_KEY`  
**Purpose**: Published baseline check (GEPA paper reports GPT-4.1-mini results)

Use this for reproducing published baselines to verify our harness matches GEPA paper's methodology.

### Grok-4.3 (Strong Model)

**LiteLLM string**: `xai/grok-4.3`  
**API key**: `XAI_API_KEY`  
**Reasoning**: Controlled via `reasoning_effort` parameter

Per xAI documentation:
```python
reasoning_effort = "none"  # Options: "none", "low", "medium", "high"
```

Set to `"none"` by default for consistency with non-thinking Qwen3.5 runs.

## Environment Variables

```bash
# Primary model (Qwen3.5-9B via OpenRouter)
export OPENROUTER_API_KEY="your-openrouter-key"

# For GPT-4.1-mini baseline check (also via OpenRouter)
# Uses same OPENROUTER_API_KEY

# For Grok-4.3 strong-model comparison
export XAI_API_KEY="your-xai-key"

# Model selection (optional, defaults provided)
export TASK_MODEL="openrouter/qwen/qwen3.5-9b"
export REFLECTION_MODEL="openrouter/qwen/qwen3.5-9b"
export BASELINE_CHECK_MODEL="openrouter/openai/gpt-4.1-mini"
export STRONG_MODEL="xai/grok-4.3"

# Reasoning control (optional, default is disabled)
export ENABLE_REASONING="false"  # Set to "true" to enable thinking tokens
export STRONG_MODEL_REASONING_EFFORT="none"  # For Grok: none/low/medium/high
```

## Cost Tracking

### Provider-Reported Costs (Preferred)

OpenRouter returns usage/cost in the response:

```json
{
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  },
  "cost": 0.001234  // In dollars
}
```

Our `BudgetTracker` prefers provider-reported costs when available via the `provider_cost` parameter:

```python
tracker.record_task_call(
    input_tokens=1234,
    output_tokens=567,
    provider_cost=0.001234  # From OpenRouter response
)
```

### Fallback Estimates

If provider cost is unavailable, we fall back to token-based estimation:

- **Qwen3.5-9B** (approximate): $0.15/M input, $0.60/M output
- **GPT-4.1-mini**: Varies, use OpenRouter's reported cost
- **Grok-4.3**: Varies by reasoning effort, use xAI's reported cost

**Note**: Fallback estimates are for planning only. Always prefer provider-reported costs for actual accounting.

## Smoke Test Validation

Stage 0 (smoke test) validates that our cost tracking matches provider reports:

```python
# SmokeTestChecks.check_call_accounting() compares:
tracked_cost = budget_tracker.total_cost
provider_cost = sum(r["provider_cost"] for r in responses)

# Passes if within 10% tolerance
```

## References

- OpenRouter reasoning parameter: https://openrouter.ai/docs/reasoning
- xAI Grok API: https://docs.x.ai/api/endpoints
- LiteLLM extra_body passthrough: https://docs.litellm.ai/docs/completion/input
- GEPA paper: arXiv:2507.19457, Table 4 (Qwen3-8B settings)
