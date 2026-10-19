"""Results storage and retrieval."""
import json
import gzip
from pathlib import Path
from typing import Any, Iterator, Optional
import dataclasses


@dataclasses.dataclass
class RunResult:
    """Single run result."""
    arm: str
    n_labels: int
    seed: int
    train_size: int
    val_size: int
    
    # Optimization outcome
    optimizer_success: bool
    optimizer_error: Optional[str] = None
    
    # Budget tracking
    program_rollouts: int = 0
    reflection_calls: int = 0
    reflection_tokens: int = 0
    within_budget_tolerance: bool = True
    
    # Performance metrics
    test_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_perturbed_accuracy: Optional[float] = None
    
    # Per-item predictions (for bootstrap)
    test_predictions: Optional[list[dict]] = None
    
    # Arm-specific logs
    proposer_log: Optional[list[dict]] = None
    optimizer_stats: Optional[dict] = None
    
    # Metadata
    runtime_seconds: Optional[float] = None
    timestamp: Optional[str] = None


class ResultsStore:
    """JSONL-based results storage."""
    
    def __init__(self, base_path: Path, compress: bool = True):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.ext = ".jsonl.gz" if compress else ".jsonl"
    
    def save_run(self, result: RunResult, file_stem: str = "results"):
        """Append one run result to JSONL."""
        path = self.base_path / f"{file_stem}{self.ext}"
        
        data = dataclasses.asdict(result)
        line = json.dumps(data) + "\n"
        
        if self.compress:
            with gzip.open(path, "at", encoding="utf-8") as f:
                f.write(line)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    
    def load_runs(self, file_stem: str = "results") -> Iterator[RunResult]:
        """Load all runs from JSONL."""
        path = self.base_path / f"{file_stem}{self.ext}"
        
        if not path.exists():
            return
        
        if self.compress:
            f = gzip.open(path, "rt", encoding="utf-8")
        else:
            f = open(path, "r", encoding="utf-8")
        
        try:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    yield RunResult(**data)
        finally:
            f.close()
    
    def save_metadata(self, metadata: dict, filename: str = "metadata.json"):
        """Save experiment metadata."""
        path = self.base_path / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, filename: str = "metadata.json") -> Optional[dict]:
        """Load experiment metadata."""
        path = self.base_path / filename
        if not path.exists():
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
