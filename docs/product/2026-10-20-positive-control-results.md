# Positive-control live result

Purpose: distinguish blanket abstention from ability to repair a deliberate instruction defect. This is a proposer-level mechanism diagnostic, not GEPA compile or a comparative efficacy benchmark.

## Design

Synthetic records contain a color and city plus the requested field. Defective instruction explicitly ignores requested_field and always returns city; clean instruction returns the requested field. Three examples inform reflection and survival (two color requests, one city request). Three fixed hidden examples use different entities/values with the same field pattern; they are evaluated only after the proposal is frozen and never sent to critics. Real Qwen3.5-9B via OpenRouter, temperature0, one attempt, strict survival. Four cells: B/M × defective/clean. No model judge; exact match scoring.

## Observed result

| Cell | Parent hidden scores | Selected hidden scores | Action | Calls |
|---|---|---|---|---|
| B defective | 0,0,1 | 1,1,1 | Admitted repair | 14 |
| B clean | 1,1,1 | 1,1,1 | Unchanged; no critique | 10 |
| M defective | 0,0,1 | 1,1,1 | Admitted repair | 16 |
| M clean | 1,1,1 | 1,1,1 | Unchanged; no critique/defense | 11 |

51 logged provider calls, no provider errors, reported cost $0.00156460. Cost metadata present for every call; not independently billing-reconciled. Full live structured outputs and admission evidence inspected.

Both critics identified the actual wrong-field directive and cited the two failing examples. Both replaced it with requested-field extraction while retaining short-answer constraints. The survival check preserved the already-correct city example. The hidden test confirms the change applies to the new record values rather than copying shown answers.

## What the debate did and did not establish

M's defender correctly distinguished obeying a defective instruction from satisfying the intended task, but marked the objection contested. Its dialectical rewrite emitted the same corrected instruction as the prosecution rewrite. Its rejected_refutations text says the refutation is rejected while its explanation and action support fixing the defect: this is a semantic inconsistency in the explanation, not faithful adjudication. The two candidate instructions were identical, so survival evaluation was reused; prosecution won the deterministic tie. This run provides no evidence that debate added value over B.

## Conclusion and limits

The updated critics are not simply refusing to criticize everything: both repair this clear defect and preserve the clean control. Real strict-JSON nonempty critique/rewrite/admission paths now execute successfully. This is a deliberately easy, evidence-complete, one-seed positive control, not proof of robust diagnosis, natural-task improvement, or generalization beyond the field pattern. It does not test a beliefs/coherence module.

Next scientific step: retain these controls and compare B/M against a simpler direct-rewrite baseline on several predefined defect types, with explicit cost accounting. Do not expand to a large HotpotQA run or claim superiority from this diagnostic.

Local artifacts: `/Users/sciphi/diaspy-smoke-results/positive-control-live-01/result.json` and `calls.jsonl`. Raw artifacts not included in this document.
