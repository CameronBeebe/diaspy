# Diaspy / exocortex: product-direction discussion

Date: 2026-10-19 (MDT)
Participants: Cameron (user) and Hermes (assistant).
Source: the recent conversation following the PR #5 implementation work, from the question about inspecting experiments through the request to take notes.
Status: working discussion notes, not an approved roadmap, preregistration, or implementation-status report. The discussion is ongoing; Cameron explicitly has more to add.

## Reading these notes

User positions are distinguished from assistant interpretations and suggestions. Quotation marks below indicate excerpts from the user's actual messages. Broader industry claims and research intuitions are recorded as hypotheses, not established facts. No new experiment, product pivot, or implementation task was authorized in this discussion beyond saving these notes.

## 1. The practical need: stay able to inspect and steer AI-assisted work

Cameron asked how experiments should be run so he can inspect code, see traces, judge the overall direction, and check that methods integrate properly and Pythonically with DSPy. Jupyter notebooks/Lab previously served this purpose, but AI makes more workflows possible without removing the need for human scrutiny.

User concern:
> "i still need to check stuff, like if it makes sense the overall direction we are going in and that the methods are integrating properly pythonic with dspy"

Assistant suggestion, not a settled tooling decision:
- Keep JupyterLab as an inspection bench, rather than the home of experimental implementation or hidden execution state.
- Keep method code in an ordinary Python package; run reproducibly; inspect saved artifacts through notebooks or a thin trace viewer.
- Separate scientific review (is the question useful?), methodological review (is the intended comparison implemented?), and engineering review (is it correct and idiomatic?). Passing tests does not settle the first two.
- Inspect a complete optimization round: parent, examples and answers, critiques/defenses, instruction diffs, scores, admission rationale, subsequent GEPA decision, and resource use.
- Avoid AI summaries becoming a barrier between Cameron and the evidence.

## 2. Original research intuition: explicit, typed, theory-driven improvement

Cameron finds it difficult to state a precise hypothesis yet, but has a substantive intuition: we can improve on asking an LLM to critique and rewrite prompts without an explicit structure for how to do that work.

> "we can do way better than just vanilla/naive rewrites of prompts, by being theory-driven in a more explicit and typed way"

> "rather than just hoping the llm doing the rewrite is in an appropriate latent space ('frame of mind') for critiquing and rewriting properly"

He also wants to avoid duplicating mechanisms that already exist in DSPy or related work.

Assistant interpretation:
- A rewrite implicitly bundles diagnosis, prioritization, explanation, intervention, anticipated regressions, and a judgment about acceptance.
- Explicit representations can make these operations addressable and testable.
- Distinguish representation (what must be expressed), procedure (what steps occur), and enforcement (what is independently checked).
- Types alone ensure neither truth nor sound evidence. A counterexample ID is useful when the system checks whether the cited example actually supports the claim.
- Separate improved proposal generation from improved proposal selection. A restrictive admission rule can produce discipline without better performance.

Provisional assistant hypothesis, not adopted as a final research claim:
"Explicit evidence-linked diagnosis and revision produces more useful prompt changes per unit of compute than unconstrained reflection on the same feedback."

## 3. GEPA, compute, and whether GEPA is the right center

Cameron questioned whether GEPA's authors considered more explicit theory or principally relied on compute. He wondered whether he was over-indexing on GEPA, whether other DSPy mechanisms were more appropriate, and whether full GEPA optimization was warranted when a harness increasingly performs repeated work in code.

> "I like the 'per unit of compute' framing."

He also recognized that adding adversarial debate itself adds compute and may therefore be another scaling strategy rather than an escape from the issue.

Research clarification supplied by the assistant:
- The retrieved GEPA paper explicitly motivates sample-efficient learning from rich natural-language execution/evaluation feedback rather than only scalar rewards.
- It identifies candidate validation as a major rollout expense and calls out "feedback engineering" as a future direction.
- It is therefore misleading to characterize GEPA simply as naive rewriting or to attribute an unverified 'compute only' motivation to its authors.
- This does not establish whether the authors considered or rejected Cameron's particular typed, theory-driven procedures.

Sources consulted in the conversation:
- GEPA paper, version 1: https://arxiv.org/html/2507.19457v1
- SIMBA: https://dspy.ai/api/optimizers/SIMBA/
- BootstrapFewShot: https://dspy.ai/api/optimizers/BootstrapFewShot/
- Refine: https://dspy.ai/api/modules/Refine/

Assistant suggestions, not selected alternatives:
- GEPA could remain a testbed rather than define the product.
- BootstrapFewShot supplies demonstrations; SIMBA develops reflective rules or demonstrations; Refine applies bounded refinement at execution time.
- A well-factored DSPy program with vetted examples and Python checks should be a serious baseline, even without an optimizer.
- Compare adversarial discussion against equal-resource alternatives such as independent proposals, ordinary reflection, better demonstrations, or more informative evaluation.
- Reusable improvements might be code, tests, retrieval knowledge, demonstrations, or instructions—not necessarily prompts.

## 4. Synthetic data and scarce genuine examples

Cameron raised the possibility that low-volume settings require more synthetic examples to bootstrap optimization than genuine examples available.

Assistant distinctions:
- Synthetic task instances, demonstrations, counterexamples, and labels have different evidential status.
- Theory could generate discriminating counterexamples: examples designed to distinguish competing explanations of failure.
- Generated labels are not automatically ground truth. Prefer executable rules, simulators, trusted sources, or human checks where available.
- Preserve provenance and validation status instead of silently promoting generated conjectures into evidence.

These were suggestions for investigating the concern, not a decision to build a synthetic-data pipeline.

## 5. Cameron's correction: bespoke requests are central, not an edge case

Cameron pushed back on the assistant's repeated-task framing. A large proportion of his requests are bespoke, and he suspects this is also true of other users and important industry applications.

> "a large amount of my requests, and i suspect from others as well, are completely bespoke"

Even when requests can be semantically matched, the actual wording, memory, surrounding context, model differences, and numerical GPU execution affect the process executing the request. Cameron's point is that semantic resemblance does not make the effective situation identical.

> "a lot more of the data in industry is actually 'low volume' in several important requests"

He contrasted this with constrained chatbot interactions—multiple choice or decision-tree navigation—which reduce the input space and can reduce the need for an LLM. This was an illustrative contrast, not a verified market claim about a named product.

Assistant acknowledgement and distinction:
- The earlier framing leaned too heavily on stable task distributions and reusable prompts.
- Bespoke requests can share decision structures without sharing the same task or template.
- Wording/context can change task meaning; sampling/numerical variability can change execution without changing intent. The system may need to handle both.

Steering implication: do not assume high repetition, stable distributions, or amortization across many identical future requests as the default product setting.

## 6. Intent representation → intent anticipation → agentic exocortex

Cameron's central formulation:
> "prompt iteration/optimization i think in the limit goes to 'intent representation' and then actually what is more important is 'intent anticipation'"

He described the desired agentic exocortex as something that can:
> "reliably predict and build in anticipation and in a reasonable direction consistent with the user's likely intent"

This goes beyond reactive answer quality. The system should advance work in a direction consistent with the user, including before the user fully articulates the next request.

Assistant interpretation, offered for further discussion:
- The reusable thing may be a procedure for forming, testing, and updating an account of what a person wants in a particular situation—not a fixed prompt.
- Cross-request learning may concern the user's priorities, constraints, tradeoffs, and changing circumstances.
- An intent model should remain revisable; it should not override current user instructions.
- Distinguish predicting relevance, preparing something useful, and having authority to act. Anticipation is not automatically authorization.
- Reversible preparation can be useful initiative without committing the user to publication, expenditure, or a change of direction.

These boundaries were introduced by the assistant; Cameron has not yet explicitly accepted a particular permission model or product architecture.

## 7. Decision traces, not claims of access to internal reasoning

Cameron explicitly liked the distinction between decision traces and reasoning traces.

> "the model can reason what it wants, but its behavior is approximated or dominated by what is EFFECTIVE which are the DECISIONS it makes in-harness"

His emphasis is on consequential behavior: what the model actually chooses and does through the harness, rather than requiring an authentic explanation of its internal cognition.

> "if the model wants to lie or obscure WHY it decides to do something, but it still did the 'correct' thing, that is important to know and track"

He sees this as helping:
> "screen off applications from major debates abut model authenticity, calibration, faithfulness, etc."

Faithful interpretation: the relationship between stated explanation and effective behavior still matters and should be tracked, but applications need not resolve all debates about internal reasoning before they can evaluate and govern decisions. Do not reduce this to "explanations never matter" or "correct output justifies any process."

Assistant qualification:
- Logs can establish a discrepancy between an explanation and observable behavior; they do not by themselves establish deliberate lying.
- Separate model-generated justifications from observed actions, constraints, outcomes, and user corrections.
- Correctness can include authority, privacy, and preserving options—not just immediate output usefulness.
- Decision accountability can avoid requiring faithful chain-of-thought, while uncertainty about intent and consequences remains operationally relevant.

## 8. Performance and inspectability: related but distinct goals

Cameron recognized that this is partly an explainability/interpretability project even though he originally framed it as performance/optimization.

> "yes this is a kind of explainable interpretability project on the side even though i was framing it in my head and to you as a performace/optimization project"

His tentative phrase was "typed reasoning traces rather than totally free form reasoning traces"; the subsequent discussion refined the focus toward decisions and evidence.

Assistant proposal:
- Performance objective: better useful improvements or actions per unit of total compute.
- Auditability objective: decisions a human can reconstruct, challenge, and correct.
- Do not silently treat improvement in one as evidence of improvement in the other.

A possible typed decision record, suggested by the assistant:
- Intent hypothesis and supporting context/provenance.
- Unresolved ambiguity.
- Proposed action, expected benefit, and possible downside.
- Authority to act.
- Observable outcome.
- User correction or acceptance.

This is a candidate design, not an agreed schema. Not every action necessarily warrants every field or another model call.

## 9. Product direction to preserve while the discussion continues

The strongest user-originated themes so far:
1. Human ability to inspect and steer the method must survive AI-assisted development.
2. Explicit typed theory-driven procedures may improve on relying on the model's implicit 'frame of mind.'
3. Bespoke, context-dependent, low-volume work is a central target setting.
4. Intent representation is a step toward the more important capability of intent anticipation.
5. An exocortex should reliably build ahead in a direction consistent with likely user intent.
6. Effective in-harness decisions are a practical object of observation and evaluation; private reasoning narratives need not be the foundation.
7. Compute efficiency matters, and debate itself must justify its resource use.
8. GEPA is not necessarily the final organizing abstraction.

Assistant synthesis offered, not a user-approved final hypothesis:
"Can an explicit decision protocol help an agent take more useful, appropriately authorized initiative on bespoke requests, with less user steering and less wasted compute?"

## 10. Open questions / not yet decided

- What exactly should an intent representation contain, and how should it change over time?
- What counts as successful anticipation versus unwanted initiative?
- How do we evaluate usefulness on genuinely bespoke requests without forcing them into an artificial repeated-task benchmark?
- Which observations establish a correct decision, including delayed outcomes and user correction?
- How much typed structure helps before it becomes overhead or constrains useful behavior?
- Which decisions should be deterministic Python, which model-mediated, and which human-approved?
- How should synthetic counterexamples and observations be mixed without confusing their evidential status?
- Which parts of existing DSPy/GEPA mechanisms should be reused, compared, or bypassed?
- How should performance, total compute, user steering effort, and auditability be measured separately?
- How should explanation/action discrepancies be recorded without assuming access to hidden motives?
- What role should notebooks, a trace inspector, and an experiment ledger play?

No final hypothesis, new benchmark, tooling choice, pivot away from GEPA, or implementation roadmap was agreed. The explicit commitment in the latest turn is to preserve faithful notes so the ongoing discussion can steer the product.
