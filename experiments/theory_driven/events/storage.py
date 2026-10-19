"""Storage for epistemic events."""

import json
from pathlib import Path
from typing import List
from datetime import datetime

from experiments.theory_driven.events.schema import EpistemicEvent


class EventStorage:
    """Stores epistemic events to JSONL files."""
    
    def __init__(self, output_dir: Path):
        """Initialize event storage.
        
        Args:
            output_dir: Directory for storing event files
        """
        self.output_dir = Path(output_dir)
        self.events_dir = self.output_dir / "epistemic_events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
    
    def save_events(
        self,
        events: List[EpistemicEvent],
        run_id: str,
        arm_name: str
    ) -> Path:
        """Save events to JSONL file.
        
        Args:
            events: List of events
            run_id: Run identifier
            arm_name: Arm name
            
        Returns:
            Path to saved file
        """
        filename = f"events_{run_id}_{arm_name}.jsonl"
        filepath = self.events_dir / filename
        
        with open(filepath, 'w') as f:
            for event in events:
                # Convert to JSON
                event_dict = event.dict()
                
                # Write as JSONL (one JSON object per line)
                f.write(json.dumps(event_dict, default=str) + "\n")
        
        return filepath
    
    def load_events(self, filepath: Path) -> List[EpistemicEvent]:
        """Load events from JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of events
        """
        events = []
        
        with open(filepath, 'r') as f:
            for line in f:
                event_dict = json.loads(line)
                event = EpistemicEvent(**event_dict)
                events.append(event)
        
        return events
    
    def get_run_events(self, run_id: str) -> List[EpistemicEvent]:
        """Get all events for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of events from all arms for this run
        """
        events = []
        
        # Find all event files for this run
        pattern = f"events_{run_id}_*.jsonl"
        for filepath in self.events_dir.glob(pattern):
            events.extend(self.load_events(filepath))
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def create_index(self) -> None:
        """Create index of all events for fast lookup.
        
        Creates an index.json file with metadata about all event files.
        """
        index = {
            "created_at": datetime.utcnow().isoformat(),
            "runs": {},
        }
        
        # Scan all event files
        for filepath in self.events_dir.glob("events_*.jsonl"):
            # Parse filename: events_{run_id}_{arm_name}.jsonl
            parts = filepath.stem.split("_")
            if len(parts) >= 3:
                run_id = parts[1]
                arm_name = "_".join(parts[2:])
                
                if run_id not in index["runs"]:
                    index["runs"][run_id] = {
                        "arms": [],
                        "event_files": [],
                    }
                
                index["runs"][run_id]["arms"].append(arm_name)
                index["runs"][run_id]["event_files"].append(str(filepath.name))
        
        # Save index
        index_path = self.events_dir / "index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
