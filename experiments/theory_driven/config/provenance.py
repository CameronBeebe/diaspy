"""Provenance tracking for LM calls.

Logs full traces: prompt, completion, params, timestamps, model, provider,
usage, cost, OpenRouter generation IDs.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict, field
import hashlib


@dataclass
class LMCallTrace:
    """Full trace of a single LM call."""
    
    # Request
    call_id: str  # Unique ID for this call
    timestamp: str  # ISO 8601
    model: str
    prompt: str
    messages: List[Dict[str, Any]]
    
    # Parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    # Response
    completion: str = ""
    response_model: Optional[str] = None  # Actual model from response
    response_provider: Optional[str] = None  # Provider from response
    
    # Usage and cost
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None
    provider_reported_cost: bool = False
    
    # OpenRouter specific
    generation_id: Optional[str] = None  # OpenRouter generation ID
    provider_order: Optional[List[str]] = None  # Requested provider order
    allow_fallbacks: bool = True
    
    # Metadata
    call_type: str = "task_lm"  # task_lm, reflection_lm, jev
    call_context: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def hash_prompt(self) -> str:
        """Hash the prompt for deduplication."""
        return hashlib.sha256(self.prompt.encode()).hexdigest()
    
    def hash_completion(self) -> str:
        """Hash the completion for comparison."""
        return hashlib.sha256(self.completion.encode()).hexdigest()


class ProvenanceLogger:
    """Logger for LM call provenance."""
    
    def __init__(self, output_dir: Path):
        """Initialize provenance logger.
        
        Args:
            output_dir: Directory for storing trace logs
        """
        self.output_dir = Path(output_dir)
        self.traces_dir = self.output_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
        self.traces: List[LMCallTrace] = []
        self.call_counter = 0
    
    def log_call(
        self,
        model: str,
        prompt: str,
        messages: List[Dict[str, Any]],
        completion: str,
        response_data: Dict[str, Any],
        call_type: str = "task_lm",
        call_context: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
    ) -> LMCallTrace:
        """Log a single LM call.
        
        Args:
            model: Requested model
            prompt: Prompt text (for simple calls)
            messages: Full messages list
            completion: Completion text
            response_data: Raw response data from LM
            call_type: Type of call (task_lm, reflection_lm, jev)
            call_context: Additional context
            request_params: Request parameters (temp, top_p, etc)
            latency_ms: Call latency in milliseconds
            
        Returns:
            LMCallTrace object
        """
        self.call_counter += 1
        call_id = f"call_{self.call_counter:06d}_{int(time.time()*1000)}"
        
        # Extract usage from response
        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        # Extract cost (OpenRouter specific)
        cost = response_data.get("cost")
        provider_reported_cost = cost is not None
        
        # Extract model/provider from response
        response_model = response_data.get("model")
        response_provider = None
        
        # OpenRouter returns provider in various places
        if "provider" in response_data:
            response_provider = response_data["provider"]
        elif "x-provider" in response_data.get("headers", {}):
            response_provider = response_data["headers"]["x-provider"]
        
        # OpenRouter generation ID
        generation_id = response_data.get("id")
        if not generation_id:
            generation_id = response_data.get("x-generation-id")
        
        # Request params
        params = request_params or {}
        
        # Provider routing (OpenRouter specific)
        provider_order = None
        allow_fallbacks = True
        if "provider" in params:
            provider_config = params["provider"]
            if isinstance(provider_config, dict):
                provider_order = provider_config.get("order")
                allow_fallbacks = provider_config.get("allow_fallbacks", True)
        
        trace = LMCallTrace(
            call_id=call_id,
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            prompt=prompt,
            messages=messages,
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            max_tokens=params.get("max_tokens"),
            seed=params.get("seed"),
            extra_params={k: v for k, v in params.items() 
                         if k not in ["temperature", "top_p", "max_tokens", "seed", "provider"]},
            completion=completion,
            response_model=response_model,
            response_provider=response_provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            provider_reported_cost=provider_reported_cost,
            generation_id=generation_id,
            provider_order=provider_order,
            allow_fallbacks=allow_fallbacks,
            call_type=call_type,
            call_context=call_context or {},
            latency_ms=latency_ms,
        )
        
        self.traces.append(trace)
        
        return trace
    
    def save_traces(self, run_id: str) -> Path:
        """Save all traces for a run.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            Path to saved trace file
        """
        filepath = self.traces_dir / f"traces_{run_id}.jsonl"
        
        with open(filepath, 'w') as f:
            for trace in self.traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        
        return filepath
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of traces."""
        if not self.traces:
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
            }
        
        return {
            "total_calls": len(self.traces),
            "total_cost": sum(t.cost or 0 for t in self.traces),
            "total_tokens": sum(t.total_tokens for t in self.traces),
            "calls_by_type": {
                "task_lm": sum(1 for t in self.traces if t.call_type == "task_lm"),
                "reflection_lm": sum(1 for t in self.traces if t.call_type == "reflection_lm"),
                "jev": sum(1 for t in self.traces if t.call_type == "jev"),
            },
            "unique_models": list(set(t.response_model or t.model for t in self.traces)),
            "unique_providers": list(set(t.response_provider for t in self.traces if t.response_provider)),
        }


async def fetch_openrouter_generation_metadata(
    generation_id: str,
    api_key: str
) -> Optional[Dict[str, Any]]:
    """Fetch OpenRouter generation metadata for cost reconciliation.
    
    Per OpenRouter API: GET /api/v1/generation?id={generation_id}
    
    Args:
        generation_id: OpenRouter generation ID
        api_key: OpenRouter API key
        
    Returns:
        Generation metadata dict or None if unavailable
    """
    import aiohttp
    
    url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    print(f"Failed to fetch generation metadata: {resp.status}")
                    return None
    except Exception as e:
        print(f"Error fetching generation metadata: {e}")
        return None


def reconcile_costs(
    traces: List[LMCallTrace],
    openrouter_metadata: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Reconcile tracked costs with OpenRouter reported costs.
    
    Args:
        traces: List of call traces
        openrouter_metadata: Dict mapping generation_id -> metadata
        
    Returns:
        List of discrepancies
    """
    discrepancies = []
    
    for trace in traces:
        if not trace.generation_id:
            continue
        
        metadata = openrouter_metadata.get(trace.generation_id)
        if not metadata:
            continue
        
        # Compare costs
        tracked_cost = trace.cost or 0
        reported_cost = metadata.get("total_cost", 0)
        
        if abs(tracked_cost - reported_cost) > 0.0001:  # 0.01 cent tolerance
            discrepancies.append({
                "call_id": trace.call_id,
                "generation_id": trace.generation_id,
                "tracked_cost": tracked_cost,
                "reported_cost": reported_cost,
                "difference": reported_cost - tracked_cost,
                "provider": metadata.get("provider_name"),
            })
    
    return discrepancies
