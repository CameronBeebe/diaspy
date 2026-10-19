"""Canary suite for detecting model drift.

Runs fixed prompts at temperature 0 before each stage, compares outputs
to stored reference, flags drift if outputs change.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import difflib
import os
import re


def safe_canary_error(error: Exception) -> str:
    """Retain provider diagnostics without credential values or auth headers."""
    text = f"{type(error).__name__}: {error}"
    for name, value in os.environ.items():
        if value and any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)bearer\s+[^\s,;\"'}]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|authorization|access[_-]?token|password)[\"']?\s*[:=]\s*)[^\n;,}]+",
        r"\1[REDACTED]", text,
    )
    return re.sub(r"\bsk-[\w-]+", "[REDACTED]", text)[:2000]


@dataclass
class CanaryPrompt:
    """A single canary prompt."""
    id: str
    prompt: str
    expected_output: Optional[str] = None
    expected_hash: Optional[str] = None
    category: str = "general"  # general, math, reasoning, coding, etc.


@dataclass
class CanaryResult:
    """Result of running a canary prompt."""
    prompt_id: str
    timestamp: str
    model: str
    provider: Optional[str]
    
    prompt: str
    output: str
    output_hash: str
    
    # Comparison with reference
    reference_hash: Optional[str] = None
    exact_match: bool = False
    similarity_score: float = 0.0
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CanarySuite:
    """Suite of canary prompts for drift detection."""
    
    # Default canary prompts (fixed, deterministic)
    DEFAULT_CANARIES = [
        CanaryPrompt(
            id="canary_001",
            prompt="What is 2 + 2?",
            category="math"
        ),
        CanaryPrompt(
            id="canary_002",
            prompt="List the first three prime numbers.",
            category="math"
        ),
        CanaryPrompt(
            id="canary_003",
            prompt="Complete the pattern: A, B, C, D, __",
            category="reasoning"
        ),
        CanaryPrompt(
            id="canary_004",
            prompt="What is the capital of France?",
            category="factual"
        ),
        CanaryPrompt(
            id="canary_005",
            prompt="Translate 'hello' to Spanish.",
            category="language"
        ),
        CanaryPrompt(
            id="canary_006",
            prompt="Sort these numbers: 5, 2, 8, 1, 9",
            category="reasoning"
        ),
        CanaryPrompt(
            id="canary_007",
            prompt="What comes after Monday?",
            category="factual"
        ),
        CanaryPrompt(
            id="canary_008",
            prompt="Write a single sentence describing a cat.",
            category="generation"
        ),
    ]
    
    def __init__(
        self,
        reference_dir: Path,
        prompts: Optional[List[CanaryPrompt]] = None
    ):
        """Initialize canary suite.
        
        Args:
            reference_dir: Directory for storing reference outputs
            prompts: List of canary prompts (uses defaults if None)
        """
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
        self.prompts = prompts or self.DEFAULT_CANARIES
        self.results: List[CanaryResult] = []
    
    def get_reference_file(self, model: str) -> Path:
        """Get reference file path for a model."""
        # Sanitize model name for filename
        safe_model = model.replace("/", "_").replace(":", "_")
        return self.reference_dir / f"canary_reference_{safe_model}.json"
    
    def load_reference(self, model: str) -> Dict[str, Dict[str, Any]]:
        """Load reference outputs for a model.
        
        Args:
            model: Model name
            
        Returns:
            Dict mapping prompt_id -> reference data
        """
        ref_file = self.get_reference_file(model)
        
        if not ref_file.exists():
            return {}
        
        with open(ref_file, 'r') as f:
            return json.load(f)
    
    def save_reference(self, model: str, results: List[CanaryResult]) -> None:
        """Save results as new reference for a model.
        
        Args:
            model: Model name
            results: Canary results to save as reference
        """
        ref_file = self.get_reference_file(model)
        
        reference = {
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompts": {
                result.prompt_id: {
                    "prompt": result.prompt,
                    "output": result.output,
                    "output_hash": result.output_hash,
                }
                for result in results
            }
        }
        
        with open(ref_file, 'w') as f:
            json.dump(reference, f, indent=2)
    
    def run_canaries(
        self,
        lm,  # DSPy LM instance
        model: str,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
    ) -> List[CanaryResult]:
        """Run canary prompts, stopping and recording the first provider error.
        
        Args:
            lm: DSPy LM instance
            model: Model name (for reference lookup)
            temperature: Temperature (0.0 for deterministic)
            seed: Random seed (if supported)
            
        Returns:
            List of canary results
        """
        import dspy
        
        results = []
        self.results = results
        
        for canary in self.prompts:
            # Simple completion
            try:
                # Configure LM with temperature 0
                with dspy.context(lm=lm):
                    # Use a plain predictor with the fixed canary prompt.
                    class SimpleQA(dspy.Signature):
                        question: str = dspy.InputField()
                        answer: str = dspy.OutputField()
                    
                    predictor = dspy.Predict(SimpleQA)
                    generation = {"temperature": temperature, "cache": False}
                    if seed is not None:
                        generation["seed"] = seed
                    response = predictor(question=canary.prompt, config=generation)
                    output = response.answer
                
                # Hash output
                output_hash = hashlib.sha256(output.encode()).hexdigest()
                
                result = CanaryResult(
                    prompt_id=canary.id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model=model,
                    provider=None,  # Would extract from response
                    prompt=canary.prompt,
                    output=output,
                    output_hash=output_hash,
                )
                
                results.append(result)
                
            except Exception as e:
                # Record failure
                result = CanaryResult(
                    prompt_id=canary.id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model=model,
                    provider=None,
                    prompt=canary.prompt,
                    output="",
                    output_hash="",
                    error=safe_canary_error(e),
                )
                results.append(result)
                break
        
        self.results = results
        return results
    
    def compare_with_reference(
        self,
        results: List[CanaryResult],
        reference: Dict[str, Dict[str, Any]]
    ) -> List[CanaryResult]:
        """Compare results with reference outputs.
        
        Args:
            results: New canary results
            reference: Reference outputs
            
        Returns:
            Results with comparison fields populated
        """
        for result in results:
            ref_data = reference.get("prompts", {}).get(result.prompt_id)
            
            if not ref_data:
                # No reference available
                result.exact_match = False
                result.similarity_score = 0.0
                continue
            
            ref_output = ref_data["output"]
            ref_hash = ref_data["output_hash"]
            
            result.reference_hash = ref_hash
            
            # Exact match
            result.exact_match = (result.output_hash == ref_hash)
            
            # Similarity score (simple string similarity)
            result.similarity_score = self._compute_similarity(
                result.output, ref_output
            )
        
        return results
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple similarity score between two strings.
        
        Uses SequenceMatcher for character-level similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0 to 1)
        """
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def check_drift(
        self,
        results: List[CanaryResult],
        reference: Dict[str, Dict[str, Any]],
        exact_match_threshold: float = 0.8,  # 80% must match exactly
        similarity_threshold: float = 0.9,    # Others must be 90% similar
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if results indicate model drift.
        
        Args:
            results: Canary results
            reference: Reference outputs
            exact_match_threshold: Fraction that must match exactly
            similarity_threshold: Minimum similarity for non-exact matches
            
        Returns:
            Tuple of (drift_detected, issues)
        """
        if not reference.get("prompts"):
            # No reference, cannot detect drift
            return False, [{"type": "no_reference", "message": "No reference outputs available"}]
        
        issues = []
        
        # Compare results
        results_with_comparison = self.compare_with_reference(results, reference)
        
        # Count exact matches
        with_reference = [r for r in results_with_comparison if r.reference_hash]
        if not with_reference:
            return False, [{"type": "no_reference", "message": "No reference outputs"}]
        
        exact_matches = sum(1 for r in with_reference if r.exact_match)
        exact_match_rate = exact_matches / len(with_reference)
        
        # Check exact match threshold
        if exact_match_rate < exact_match_threshold:
            issues.append({
                "type": "low_exact_match_rate",
                "exact_match_rate": exact_match_rate,
                "threshold": exact_match_threshold,
                "message": f"Only {exact_match_rate:.1%} exact matches (threshold {exact_match_threshold:.1%})"
            })
        
        # Check similarity for non-exact matches
        non_exact = [r for r in with_reference if not r.exact_match]
        for result in non_exact:
            if result.similarity_score < similarity_threshold:
                issues.append({
                    "type": "low_similarity",
                    "prompt_id": result.prompt_id,
                    "similarity": result.similarity_score,
                    "threshold": similarity_threshold,
                    "prompt": result.prompt[:50] + "...",
                    "expected": reference["prompts"][result.prompt_id]["output"][:50] + "...",
                    "actual": result.output[:50] + "...",
                })
        
        drift_detected = len(issues) > 0
        
        return drift_detected, issues
    
    def save_results(self, output_file: Path) -> None:
        """Save canary results.
        
        Args:
            output_file: Path to save results
        """
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_report(self, results: List[CanaryResult], reference: Dict[str, Dict[str, Any]]) -> None:
        """Print a summary report.
        
        Args:
            results: Canary results
            reference: Reference outputs
        """
        print("\n" + "="*60)
        print("CANARY SUITE REPORT")
        print("="*60)
        
        print(f"Model: {results[0].model if results else 'N/A'}")
        print(f"Timestamp: {results[0].timestamp if results else 'N/A'}")
        print(f"Prompts run: {len(results)}")
        
        if not reference.get("prompts"):
            print("\n⚠ No reference outputs available")
            print("Run with --save-reference to create baseline")
            return
        
        # Compare
        results_with_comparison = self.compare_with_reference(results, reference)
        with_reference = [r for r in results_with_comparison if r.reference_hash]
        
        if not with_reference:
            print("\n⚠ No comparable results")
            return
        
        exact_matches = sum(1 for r in with_reference if r.exact_match)
        exact_match_rate = exact_matches / len(with_reference)
        
        print(f"\nExact matches: {exact_matches}/{len(with_reference)} ({exact_match_rate:.1%})")
        
        # Show non-matches
        non_matches = [r for r in with_reference if not r.exact_match]
        if non_matches:
            print(f"\nNon-exact matches ({len(non_matches)}):")
            for result in non_matches:
                print(f"\n  {result.prompt_id}: {result.prompt[:60]}")
                print(f"    Similarity: {result.similarity_score:.2%}")
                ref_output = reference["prompts"][result.prompt_id]["output"]
                print(f"    Expected: {ref_output[:80]}")
                print(f"    Actual:   {result.output[:80]}")
        
        print("\n" + "="*60)
