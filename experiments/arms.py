"""Experiment arm definitions.

Arms A0-F implement the pre-registered research design:
- A0: zero-shot (unoptimized baseline)
- A: GEPA default with InstructionProposer
- E: GEPA + survival check + archive (no critic, filter-only control)
- C: GEPA + untyped prose critic (no mechanical check)
- B: GEPA + typed Popperian proposer (the hypothesis)
- D: GEPA + Millian debate (two rival critics, optional)
- F: dspy.SIMBA
"""
import json
import re
from typing import Literal, Optional
import pydantic
import dspy


# ========== Typed critique structures ==========

class Refutation(pydantic.BaseModel):
    """Typed refutation: falsifiable claim + counterexamples + severity."""
    claim: str
    counterexample_ids: list[int]
    severity: float = pydantic.Field(ge=0, le=1)
    proposed_fix: str


class Critique(dspy.Signature):
    """Conjecture-and-refutation: state each way the instruction fails as a falsifiable
    claim, cite the example ids that break it, and rate severity."""
    current_instruction: str = dspy.InputField()
    examples_with_feedback: str = dspy.InputField(desc="numbered examples: inputs, outputs, feedback")
    refutations: list[Refutation] = dspy.OutputField()


class Rewrite(dspy.Signature):
    """Rewrite the instruction so that it survives every refutation. Address each one."""
    current_instruction: str = dspy.InputField()
    refutations: list[Refutation] = dspy.InputField()
    new_instruction: str = dspy.OutputField()


# ========== Untyped critic signatures (arm C) ==========

class UntypedCritique(dspy.Signature):
    """Critique the current instruction based on failures in the examples."""
    current_instruction: str = dspy.InputField()
    examples_with_feedback: str = dspy.InputField()
    critique: str = dspy.OutputField(desc="prose analysis of what's wrong")


class UntypedRewrite(dspy.Signature):
    """Rewrite the instruction to address the critique."""
    current_instruction: str = dspy.InputField()
    critique: str = dspy.InputField()
    new_instruction: str = dspy.OutputField()


# ========== Millian debate signatures (arm D) ==========

class RivalDiagnosis(dspy.Signature):
    """Propose one specific diagnosis of why the instruction fails."""
    current_instruction: str = dspy.InputField()
    examples_with_feedback: str = dspy.InputField()
    diagnosis: str = dspy.OutputField(desc="one falsifiable hypothesis about the failure")
    predicted_counterexamples: str = dspy.OutputField(desc="which example indices this explains")


class DiagnosisJudge(dspy.Signature):
    """Compare two rival diagnoses and pick the one whose counterexamples hold up."""
    diagnosis_a: str = dspy.InputField()
    diagnosis_b: str = dspy.InputField()
    examples_with_feedback: str = dspy.InputField()
    winner: Literal["A", "B"] = dspy.OutputField()
    rationale: str = dspy.OutputField()


# ========== Proposers ==========

class PopperianProposer:
    """Arm B: typed Popperian proposer with survival check and archive."""
    
    def __init__(self, program, metric, gold_by_question, critic_lm, task_lm,
                 max_attempts=2, tau=0.5, archive_cap=6):
        self.program = program
        self.metric = metric
        self.gold = gold_by_question
        self.critic_lm = critic_lm
        self.task_lm = task_lm
        self.max_attempts = max_attempts
        self.tau = tau
        self.archive_cap = archive_cap
        
        self.critique = dspy.Predict(Critique)
        self.rewrite = dspy.Predict(Rewrite)
        self.archive = []
        self.log = []
        
    def survives(self, name, instruction, refs, batch):
        """Re-run counterexamples under new instruction; all must score 1."""
        prog = self.program.deepcopy()
        dict(prog.named_predictors())[name].signature = \
            dict(prog.named_predictors())[name].signature.with_instructions(instruction)
        
        # Gather counterexamples with severity >= tau
        ids = sorted({i for r in refs if r.severity >= self.tau 
                     for i in r.counterexample_ids if i < len(batch)})
        qs = [batch[i]["Inputs"]["question"] for i in ids]
        qs += [q for q in self.archive if q not in qs]
        
        with dspy.context(lm=self.task_lm):
            for q in qs:
                gold = self.gold[q]
                pred = prog(question=q)
                if self.metric(gold, pred, None, None, None).score < 1:
                    return False, qs
        
        # Survived: archive this round's counterexamples
        for q in qs[:len(ids)]:
            if q not in self.archive:
                self.archive = (self.archive + [q])[-self.archive_cap:]
        
        return True, qs
    
    def __call__(self, candidate, reflective_dataset, components_to_update):
        out = {}
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            
            batch = list(reflective_dataset[name])
            shown = "\n".join(f"[{i}] {json.dumps(ex, default=str)}" for i, ex in enumerate(batch))
            
            with dspy.context(lm=self.critic_lm):
                refs = self.critique(
                    current_instruction=candidate[name],
                    examples_with_feedback=shown
                ).refutations
            
            new = candidate[name]
            for attempt in range(1, self.max_attempts + 1):
                with dspy.context(lm=self.critic_lm):
                    new = self.rewrite(
                        current_instruction=candidate[name],
                        refutations=refs
                    ).new_instruction
                
                ok, checked = self.survives(name, new, refs, batch)
                self.log.append({
                    "attempt": attempt,
                    "n_refutations": len(refs),
                    "checked_ids": checked,
                    "survived": ok,
                    "new_instruction": new
                })
                
                if ok:
                    break
            
            out[name] = new
        
        return out


class CheckOnlyProposer:
    """Arm E: survival check + archive but no critic (filter-only control)."""
    
    def __init__(self, program, metric, gold_by_question, reflection_lm, task_lm,
                 max_attempts=2, archive_cap=6):
        self.program = program
        self.metric = metric
        self.gold = gold_by_question
        self.reflection_lm = reflection_lm
        self.task_lm = task_lm
        self.max_attempts = max_attempts
        self.archive_cap = archive_cap
        
        # Use GEPA's default instruction proposer signature
        self.propose = dspy.Predict("current_instruction, examples_with_feedback -> new_instruction")
        self.archive = []
        self.log = []
    
    def survives(self, name, instruction, batch):
        """Re-run all minibatch examples + archive; all must score 1."""
        prog = self.program.deepcopy()
        dict(prog.named_predictors())[name].signature = \
            dict(prog.named_predictors())[name].signature.with_instructions(instruction)
        
        qs = [ex["Inputs"]["question"] for ex in batch]
        qs += [q for q in self.archive if q not in qs]
        
        with dspy.context(lm=self.task_lm):
            for q in qs:
                gold = self.gold[q]
                pred = prog(question=q)
                if self.metric(gold, pred, None, None, None).score < 1:
                    return False, qs
        
        # Survived: archive minibatch questions
        for q in qs[:len(batch)]:
            if q not in self.archive:
                self.archive = (self.archive + [q])[-self.archive_cap:]
        
        return True, qs
    
    def __call__(self, candidate, reflective_dataset, components_to_update):
        out = {}
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            
            batch = list(reflective_dataset[name])
            shown = "\n".join(f"[{i}] {json.dumps(ex, default=str)}" for i, ex in enumerate(batch))
            
            new = candidate[name]
            for attempt in range(1, self.max_attempts + 1):
                with dspy.context(lm=self.reflection_lm):
                    new = self.propose(
                        current_instruction=candidate[name],
                        examples_with_feedback=shown
                    ).new_instruction
                
                ok, checked = self.survives(name, new, batch)
                self.log.append({
                    "attempt": attempt,
                    "checked_ids": checked,
                    "survived": ok,
                    "new_instruction": new
                })
                
                if ok:
                    break
            
            out[name] = new
        
        return out


class UntypedCriticProposer:
    """Arm C: untyped prose critic, no mechanical check."""
    
    def __init__(self, critic_lm):
        self.critic_lm = critic_lm
        self.critique_pred = dspy.Predict(UntypedCritique)
        self.rewrite_pred = dspy.Predict(UntypedRewrite)
        self.log = []
    
    def __call__(self, candidate, reflective_dataset, components_to_update):
        out = {}
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            
            batch = list(reflective_dataset[name])
            shown = "\n".join(f"[{i}] {json.dumps(ex, default=str)}" for i, ex in enumerate(batch))
            
            with dspy.context(lm=self.critic_lm):
                critique = self.critique_pred(
                    current_instruction=candidate[name],
                    examples_with_feedback=shown
                ).critique
                
                new = self.rewrite_pred(
                    current_instruction=candidate[name],
                    critique=critique
                ).new_instruction
            
            self.log.append({
                "critique": critique,
                "new_instruction": new
            })
            
            out[name] = new
        
        return out


class MillianProposer:
    """Arm D: two rival critics debate, winner's diagnosis used in Popperian check."""
    
    def __init__(self, program, metric, gold_by_question, critic_lm, task_lm,
                 max_attempts=2, tau=0.5, archive_cap=6):
        self.program = program
        self.metric = metric
        self.gold = gold_by_question
        self.critic_lm = critic_lm
        self.task_lm = task_lm
        self.max_attempts = max_attempts
        self.tau = tau
        self.archive_cap = archive_cap
        
        self.diagnose = dspy.Predict(RivalDiagnosis)
        self.judge = dspy.Predict(DiagnosisJudge)
        self.rewrite = dspy.Predict(Rewrite)
        self.archive = []
        self.log = []
    
    def survives(self, name, instruction, refs, batch):
        """Same as PopperianProposer."""
        prog = self.program.deepcopy()
        dict(prog.named_predictors())[name].signature = \
            dict(prog.named_predictors())[name].signature.with_instructions(instruction)
        
        ids = sorted({i for r in refs if r.severity >= self.tau 
                     for i in r.counterexample_ids if i < len(batch)})
        qs = [batch[i]["Inputs"]["question"] for i in ids]
        qs += [q for q in self.archive if q not in qs]
        
        with dspy.context(lm=self.task_lm):
            for q in qs:
                gold = self.gold[q]
                pred = prog(question=q)
                if self.metric(gold, pred, None, None, None).score < 1:
                    return False, qs
        
        for q in qs[:len(ids)]:
            if q not in self.archive:
                self.archive = (self.archive + [q])[-self.archive_cap:]
        
        return True, qs
    
    def __call__(self, candidate, reflective_dataset, components_to_update):
        out = {}
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            
            batch = list(reflective_dataset[name])
            shown = "\n".join(f"[{i}] {json.dumps(ex, default=str)}" for i, ex in enumerate(batch))
            
            # Two rival diagnoses
            with dspy.context(lm=self.critic_lm):
                diag_a = self.diagnose(
                    current_instruction=candidate[name],
                    examples_with_feedback=shown
                )
                diag_b = self.diagnose(
                    current_instruction=candidate[name],
                    examples_with_feedback=shown
                )
                
                # Judge picks winner
                verdict = self.judge(
                    diagnosis_a=diag_a.diagnosis,
                    diagnosis_b=diag_b.diagnosis,
                    examples_with_feedback=shown
                )
                
                # Convert winning diagnosis to typed refutation
                winner_diag = diag_a if verdict.winner == "A" else diag_b
                winner_ids_str = winner_diag.predicted_counterexamples
                ids = [int(x) for x in re.findall(r'\d+', winner_ids_str)]
                
                refs = [Refutation(
                    claim=winner_diag.diagnosis,
                    counterexample_ids=ids,
                    severity=0.9,
                    proposed_fix=f"Address: {winner_diag.diagnosis}"
                )]
            
            new = candidate[name]
            for attempt in range(1, self.max_attempts + 1):
                with dspy.context(lm=self.critic_lm):
                    new = self.rewrite(
                        current_instruction=candidate[name],
                        refutations=refs
                    ).new_instruction
                
                ok, checked = self.survives(name, new, refs, batch)
                self.log.append({
                    "attempt": attempt,
                    "debate_winner": verdict.winner,
                    "rationale": verdict.rationale,
                    "checked_ids": checked,
                    "survived": ok,
                    "new_instruction": new
                })
                
                if ok:
                    break
            
            out[name] = new
        
        return out


def create_arm_config(arm_name: str, **kwargs):
    """Factory for arm configurations."""
    configs = {
        "A0": {"optimizer": None, "description": "Zero-shot unoptimized baseline"},
        "A": {"optimizer": "gepa_default", "description": "GEPA with default InstructionProposer"},
        "E": {"optimizer": "gepa_check_only", "description": "GEPA with check + archive, no critic"},
        "C": {"optimizer": "gepa_untyped_critic", "description": "GEPA with untyped prose critic"},
        "B": {"optimizer": "gepa_popperian", "description": "GEPA with typed Popperian proposer (hypothesis)"},
        "D": {"optimizer": "gepa_millian", "description": "GEPA with Millian debate (optional)"},
        "F": {"optimizer": "simba", "description": "dspy.SIMBA"},
    }
    
    if arm_name not in configs:
        raise ValueError(f"Unknown arm: {arm_name}. Valid arms: {list(configs.keys())}")
    
    config = configs[arm_name].copy()
    config.update(kwargs)
    return config
