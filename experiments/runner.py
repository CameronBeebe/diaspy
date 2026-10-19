"""Main experiment runner."""
import time
import datetime
from pathlib import Path
from typing import Optional, List
import dspy
import numpy as np

from .config import ExperimentConfig, BudgetTracker, BudgetedLM
from .arms import (
    PopperianProposer, CheckOnlyProposer, UntypedCriticProposer, MillianProposer,
    create_arm_config
)
from .task import IFBenchProgram, ifbench_metric, create_dummy_ifbench_data
from .store import ResultsStore, RunResult


class ExperimentRunner:
    """Runs the full experiment grid."""
    
    def __init__(self, config: ExperimentConfig, use_dummy_lm: bool = True):
        self.config = config
        self.use_dummy_lm = use_dummy_lm
        self.store = ResultsStore(Path(config.results_dir))
        
        # Save config
        import dataclasses
        import json
        config_dict = dataclasses.asdict(config)
        self.store.save_metadata({
            "config": config_dict,
            "use_dummy_lm": use_dummy_lm,
            "created_at": datetime.datetime.now().isoformat()
        })
    
    def setup_lms(self, seed: int):
        """Set up task and reflection LMs."""
        if self.use_dummy_lm:
            # Use custom FakeLM similar to skeleton - DummyLM format changed in 3.4.1
            from dspy.utils import DummyLM
            from dspy.lm15 import Message, Response, TextPart, Usage
            
            class SimpleFakeLM(DummyLM):
                """Simple fake LM for testing."""
                def __init__(self, answers):
                    super().__init__({})
                    self.answers = answers
                    self.calls = 0
                    lm = self
                    
                    def _complete(messages):
                        # Extract question/inputs from last message
                        last_msg = messages[-1]["content"] if messages else ""
                        
                        # Simple pattern matching
                        answer_text = "test answer"
                        for pattern, ans in lm.answers.items():
                            if pattern.lower() in last_msg.lower():
                                answer_text = ans
                                break
                        
                        # Format as field: value for DSPy parsing
                        # Check if this is looking for multiple fields
                        if "refutation" in last_msg.lower():
                            formatted = "refutations: []"
                        elif "new_instruction" in last_msg.lower():
                            formatted = "new_instruction: " + answer_text
                        elif "critique" in last_msg.lower():
                            formatted = "critique: " + answer_text
                        else:
                            # Default to answer field
                            formatted = "answer: " + answer_text
                        
                        lm.calls += 1
                        return Response(
                            id=None,
                            model="fake",
                            finish_reason="stop",
                            message=Message.assistant([TextPart(formatted)]),
                            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0)
                        )
                    
                    self._engine_spec._complete_messages = _complete
            
            # Simple Q&A for testing
            task_answers = {
                "what is 2+2": "4",
                "capital of france": "Paris",
                "legs does a spider": "8",
                "color is the sky": "blue",
                "what is 10-3": "7",
                "wrote hamlet": "Shakespeare",
                "what is 5*5": "25",
                "boiling point": "100",
                "days in a week": "7",
                "square root of 16": "4",
            }
            
            task_lm = SimpleFakeLM(task_answers)
            reflection_lm = SimpleFakeLM({
                "refutation": "[]",
                "new_instruction": "Improved instruction: answer clearly and concisely.",
                "critique": "The instruction could be more specific.",
            })
        else:
            if not self.config.task_model or not self.config.reflection_model:
                raise ValueError("Must specify task_model and reflection_model for real runs")
            
            task_lm = dspy.LM(model=self.config.task_model, temperature=0.0, seed=seed)
            reflection_lm = dspy.LM(model=self.config.reflection_model, temperature=0.7, seed=seed)
        
        return task_lm, reflection_lm
    
    def prepare_data(self, n_labels: int, seed: int):
        """Prepare train/val/test splits."""
        # For testing with DummyLM
        if self.use_dummy_lm:
            all_data = create_dummy_ifbench_data(n=10, seed=seed)
            
            # Split as per design: ceil(N/2) train, floor(N/2) val
            train_size = (n_labels + 1) // 2
            val_size = n_labels // 2
            
            train = all_data[:train_size]
            val = all_data[train_size:train_size + val_size]
            test = all_data  # Use all for test in dummy mode
            
            # Build gold dict
            gold_by_question = {ex.question: ex for ex in all_data}
            
            return train, val, test, gold_by_question
        else:
            raise NotImplementedError("Real IFBench data loading not implemented yet")
    
    def run_arm(
        self,
        arm: str,
        n_labels: int,
        seed: int,
        train: List,
        val: List,
        test: List,
        gold_by_question: dict,
        task_lm,
        reflection_lm
    ) -> RunResult:
        """Run one arm configuration."""
        print(f"Running {arm} with N={n_labels}, seed={seed}")
        
        start_time = time.time()
        tracker = BudgetTracker(self.config.budget)
        
        # For simplicity in smoke tests, use LMs directly
        # TODO: Add budget tracking wrapper that DSPy accepts
        tracked_task_lm = task_lm
        tracked_reflection_lm = reflection_lm
        
        try:
            program = IFBenchProgram()
            
            # A0: zero-shot, no optimization
            if arm == "A0":
                dspy.configure(lm=tracked_task_lm)
                optimized = program
                optimizer_stats = {"optimized": False}
                proposer_log = None
            
            # A: GEPA default
            elif arm == "A":
                dspy.configure(lm=tracked_task_lm)
                optimizer = dspy.GEPA(
                    metric=ifbench_metric,
                    reflection_lm=tracked_reflection_lm,
                    max_metric_calls=self.config.budget.max_program_rollouts,
                    reflection_minibatch_size=self.config.gepa_reflection_minibatch_size,
                    use_merge=self.config.gepa_use_merge,
                    seed=seed,
                    track_stats=True
                )
                optimized = optimizer.compile(program, trainset=train, valset=val)
                optimizer_stats = {"optimized": True, "optimizer": "gepa_default"}
                proposer_log = None
            
            # E: GEPA + check only
            elif arm == "E":
                dspy.configure(lm=tracked_task_lm)
                proposer = CheckOnlyProposer(
                    program=program,
                    metric=ifbench_metric,
                    gold_by_question=gold_by_question,
                    reflection_lm=tracked_reflection_lm,
                    task_lm=tracked_task_lm,
                    max_attempts=self.config.proposer_max_attempts,
                    archive_cap=self.config.proposer_archive_cap
                )
                optimizer = dspy.GEPA(
                    metric=ifbench_metric,
                    instruction_proposer=proposer,
                    max_metric_calls=self.config.budget.max_program_rollouts,
                    reflection_minibatch_size=self.config.gepa_reflection_minibatch_size,
                    use_merge=self.config.gepa_use_merge,
                    seed=seed,
                    track_stats=True
                )
                optimized = optimizer.compile(program, trainset=train, valset=val)
                optimizer_stats = {"optimized": True, "optimizer": "gepa_check_only"}
                proposer_log = proposer.log
            
            # C: GEPA + untyped critic
            elif arm == "C":
                dspy.configure(lm=tracked_task_lm)
                proposer = UntypedCriticProposer(critic_lm=tracked_reflection_lm)
                optimizer = dspy.GEPA(
                    metric=ifbench_metric,
                    instruction_proposer=proposer,
                    max_metric_calls=self.config.budget.max_program_rollouts,
                    reflection_minibatch_size=self.config.gepa_reflection_minibatch_size,
                    use_merge=self.config.gepa_use_merge,
                    seed=seed,
                    track_stats=True
                )
                optimized = optimizer.compile(program, trainset=train, valset=val)
                optimizer_stats = {"optimized": True, "optimizer": "gepa_untyped_critic"}
                proposer_log = proposer.log
            
            # B: GEPA + Popperian
            elif arm == "B":
                dspy.configure(lm=tracked_task_lm)
                proposer = PopperianProposer(
                    program=program,
                    metric=ifbench_metric,
                    gold_by_question=gold_by_question,
                    critic_lm=tracked_reflection_lm,
                    task_lm=tracked_task_lm,
                    max_attempts=self.config.proposer_max_attempts,
                    tau=self.config.proposer_tau,
                    archive_cap=self.config.proposer_archive_cap
                )
                optimizer = dspy.GEPA(
                    metric=ifbench_metric,
                    instruction_proposer=proposer,
                    max_metric_calls=self.config.budget.max_program_rollouts,
                    reflection_minibatch_size=self.config.gepa_reflection_minibatch_size,
                    use_merge=self.config.gepa_use_merge,
                    seed=seed,
                    track_stats=True
                )
                optimized = optimizer.compile(program, trainset=train, valset=val)
                optimizer_stats = {"optimized": True, "optimizer": "gepa_popperian"}
                proposer_log = proposer.log
            
            # D: GEPA + Millian
            elif arm == "D":
                dspy.configure(lm=tracked_task_lm)
                proposer = MillianProposer(
                    program=program,
                    metric=ifbench_metric,
                    gold_by_question=gold_by_question,
                    critic_lm=tracked_reflection_lm,
                    task_lm=tracked_task_lm,
                    max_attempts=self.config.proposer_max_attempts,
                    tau=self.config.proposer_tau,
                    archive_cap=self.config.proposer_archive_cap
                )
                optimizer = dspy.GEPA(
                    metric=ifbench_metric,
                    instruction_proposer=proposer,
                    max_metric_calls=self.config.budget.max_program_rollouts,
                    reflection_minibatch_size=self.config.gepa_reflection_minibatch_size,
                    use_merge=self.config.gepa_use_merge,
                    seed=seed,
                    track_stats=True
                )
                optimized = optimizer.compile(program, trainset=train, valset=val)
                optimizer_stats = {"optimized": True, "optimizer": "gepa_millian"}
                proposer_log = proposer.log
            
            # F: SIMBA
            elif arm == "F":
                dspy.configure(lm=tracked_task_lm)
                bsize = self.config.simba_bsize or min(32, len(train))
                
                # SIMBA metric needs to be a callable (gold, pred) -> float
                def simba_metric(gold, pred):
                    result = ifbench_metric(gold, pred, None, None, None)
                    return result.score
                
                optimizer = dspy.SIMBA(
                    metric=simba_metric,
                    prompt_model=tracked_reflection_lm,
                    max_demos=self.config.simba_max_demos,
                    bsize=bsize
                )
                # SIMBA uses trainset only
                optimized = optimizer.compile(program, trainset=train + val)
                optimizer_stats = {"optimized": True, "optimizer": "simba"}
                proposer_log = None
            
            else:
                raise ValueError(f"Unknown arm: {arm}")
            
            # Evaluate on test set
            dspy.configure(lm=tracked_task_lm)
            test_predictions = []
            test_correct = 0
            
            for ex in test:
                pred = optimized(question=ex.question)
                score_result = ifbench_metric(ex, pred)
                score = score_result.score
                test_correct += score
                
                test_predictions.append({
                    "question": ex.question,
                    "gold": ex.answer,
                    "pred": getattr(pred, "answer", ""),
                    "score": score
                })
            
            test_accuracy = test_correct / len(test)
            
            # Val accuracy
            val_correct = sum(
                ifbench_metric(ex, optimized(question=ex.question)).score
                for ex in val
            )
            val_accuracy = val_correct / len(val) if val else 0.0
            
            budget_report = tracker.report()
            runtime = time.time() - start_time
            
            # TODO: Get actual call counts from DSPy once wrapper works
            actual_rollouts = 0
            actual_reflection_calls = 0
            actual_reflection_tokens = 0
            
            result = RunResult(
                arm=arm,
                n_labels=n_labels,
                seed=seed,
                train_size=len(train),
                val_size=len(val),
                optimizer_success=True,
                program_rollouts=actual_rollouts,
                reflection_calls=actual_reflection_calls,
                reflection_tokens=actual_reflection_tokens,
                within_budget_tolerance=True,  # TODO: Check when tracking works
                test_accuracy=test_accuracy,
                val_accuracy=val_accuracy,
                test_predictions=test_predictions,
                proposer_log=proposer_log,
                optimizer_stats=optimizer_stats,
                runtime_seconds=runtime,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            print(f"  ✓ {arm} N={n_labels} seed={seed}: test_acc={test_accuracy:.2f}")
            
            return result
        
        except Exception as e:
            runtime = time.time() - start_time
            print(f"  ✗ {arm} N={n_labels} seed={seed} FAILED: {e}")
            
            return RunResult(
                arm=arm,
                n_labels=n_labels,
                seed=seed,
                train_size=len(train),
                val_size=len(val),
                optimizer_success=False,
                optimizer_error=str(e),
                runtime_seconds=runtime,
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def run_grid(self):
        """Run full experiment grid."""
        print(f"Starting experiment grid: {len(self.config.arms)} arms × "
              f"{len(self.config.n_labels)} N × {len(self.config.seeds)} seeds")
        
        total_runs = len(self.config.arms) * len(self.config.n_labels) * len(self.config.seeds)
        completed = 0
        
        for n_labels in self.config.n_labels:
            for seed in self.config.seeds:
                # Prepare data once per (N, seed)
                train, val, test, gold = self.prepare_data(n_labels, seed)
                task_lm, reflection_lm = self.setup_lms(seed)
                
                for arm in self.config.arms:
                    result = self.run_arm(
                        arm, n_labels, seed, train, val, test, gold,
                        task_lm, reflection_lm
                    )
                    
                    self.store.save_run(result)
                    
                    completed += 1
                    print(f"Progress: {completed}/{total_runs} runs completed\n")
        
        print(f"✓ Experiment complete: {completed} runs")
