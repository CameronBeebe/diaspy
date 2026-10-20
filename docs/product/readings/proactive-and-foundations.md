# Proactive assistance and executable epistemic procedures

Working research notes, not an approved roadmap. User clarification: formal epistemology may be implemented in the external harness; whether models later internalize it is a separate question.

## Reading scope

Read the extracted full text of the versioned PDFs below, including methods, evaluation, limitations, and appendices where present. This is textual reading, not visual inspection of figures or independent replication. Some PDF equations and tables have extraction artifacts. No repository implementation or empirical result was reproduced.

- π-Bench v1: 44 PDF pages.[9]
- Proactive Agent v3: 21 PDF pages.[15]
- Horvitz, Principles of Mixed-Initiative User Interfaces: 9 PDF pages, final page has no extracted substantive text.[12]
- Rao and Georgeff, BDI Agents: From Theory to Practice: 8 PDF pages.[13]
- Cooperative Inverse Reinforcement Learning v4: 13 PDF pages, including supplementary proofs.[14]

Full extracted texts are in /Users/sciphi/.hermes/cache/scratch/epistemic-papers/. Those scratch files are not durable archival copies.

## 1. π-Bench: closest evaluation setting, but not a complete objective

### What the paper does

Sections 3.1–3.5, PDF pp. 3–6: persistent workspaces, artifact-centered tasks, five personas, 20 sessions per persona, underspecified initial requests, and hidden intents recoverable through prior context, artifacts, or clarification. The user agent records whether each hidden requirement is completed without disclosure, elicited through a focused question (called 'inferred'), or proactively supplied by the user (called 'provided'). Proactivity gives equal credit to completion and focused elicitation; completeness separately grades concrete obligations.[9]

Section 4.3, p. 9: removing preceding dependency-group sessions reduces proactive intent resolution more than completeness, as the simulated user eventually supplies omitted requirements. This is evidence within the benchmark that history affects who does the requirement-discovery work, not proof of real-world burden reduction.[9]

Appendices B/C, pp. 20–26: intent assignment and artifact grading have distinct procedures. Rule-based graders inspect tool evidence for exact outcomes; rubric graders use selected trace context. Appendix F.1 reports an audit of 120 trajectories, with human-expert disagreements of 2.66% for checklist judgments and 1.48% for intent assignments.[9]

### Important qualifications from close reading

- 'Inferred' means targeted elicitation, not that the model independently recovered the missing answer. The metric is initiative in requirement discovery, not mental-state identification accuracy.[9]
- Completion-check prompt, p. 21, explicitly instructs the judge to trust assistant wording and supplied file context and not check factual accuracy beyond that context. Thus proactivity credit is not itself verified correctness. Separate completeness checks matter.[9]
- The simulator eventually discloses every hidden requirement, and terminal status is assigned once. This is a closed-set requirement-recovery protocol, not a model of a real person reconsidering priorities during work.[9]
- The proactivity metric does not explicitly price every unnecessary question or penalize all invented requirements. Turn count is complementary, not equivalent to experienced burden (p. 9).[9]
- Appendix J's cross-session failure example summarizes an agent asking for the research theme, yet labels the subsequent theme disclosure 'provided' rather than 'inferred' (p. 44). The condensed trace may omit relevant detail; it deserves checking against raw traces before treating all assignments as unambiguous.[9]
- Human audit agreement supports consistency against this protocol; it does not establish that the protocol captures all real-world usefulness, authority, or evolving-intent concerns. This is our interpretation of the evaluation boundary, not a claimed author result.

### Reuse for Diaspy (proposal)

Reuse the separation of initiative and completion, recoverable hidden requirements, persistent workspaces, and objective artifact checks. Add permission violations, unnecessary interruption, correction/rework cost, invented constraints, stale-memory use, and changed-intent episodes. Keep proposal, preparation, and externally consequential execution distinct.

## 2. Proactive Agent: useful warning against optimizing for activity

### What the paper does

Sections 3.1–3.5, pp. 3–6: predicts a task or no task from observed events, user activities, and environmental state. Uses human acceptance labels to train a reward model; uses an LLM gym, simulated user, and proactive agent to generate training events. Section 3.4 and the agent prompt explicitly place execution after user acceptance, not merely after prediction.[15]

Tables 1–3, pp. 2, 6–7: 6,790 agent-training events; 233 test events; reward-model split of 1,640 training and 120 test entries. Table 3 reports fine-tuned Qwen2-7B F1 of 66.47%, precision of 49.78%, recall of 100%, and reported false-alarm of 50.22%. Agent evaluation uses the trained reward model to simulate user judgments.[15]

Section 4.3, p. 8: external reward-model feedback can reduce unwanted suggestions but can also drive the agent into silence and lower recall. Multiple-task evaluation accepts a bundle if any candidate is accepted; this is not a direct measurement of the cognitive cost of reading all suggestions.[15]

### Reporting caveats

The prose accompanying Table 3 gives figures inconsistent with the table: LLaMA improvement 44.78%→61.74% in prose versus 55.06%→66.25% in Table 3; Qwen best F1 appears as both 66.07% and 66.47%. Treat exact performance claims cautiously until artifacts/code are checked.[15]

Appendix C.5 defines TP as an offered task accepted, FP as an offered task rejected, TN as no offer when none needed, and FN as no offer when help needed. The main-text recall description is less precise than these definitions. In these definitions, an inappropriate suggestion when help is needed is recorded as FP, not necessarily also FN for the missed actual need. That limits a plain-English interpretation of high recall as 'all true user needs were met'.[15]

Our assessment: this is promising evidence that task suggestion can be trained, not evidence of a reliable autonomous exocortex. The small held-out reward-model set, surrogate-judge evaluation, high rejected-suggestion share, and lack of extensive longitudinal real-user evaluation are important limits. Do not equate acceptance, usefulness over time, and authority.

## 3. Horvitz: a direct precedent for an external decision layer

PDF pp. 1–2 state design principles including uncertainty about user goals, attention/timing, costs of mistaken initiative, efficient clarification, direct invocation/termination, and reducing precision under uncertainty. LookOut can fall back from scheduling a specific appointment to showing a relevant calendar span.[12]

PDF pp. 5–7 separate a learned estimate of user intent from expected-utility action selection. The system chooses among doing nothing, asking, and acting. Context-dependent utilities change decision thresholds; users can adjust utilities or thresholds directly. PDF p. 7 describes adapting the timing model from observations of user behavior.[12]

This is a strong conceptual and implemented precedent for the user's harness interpretation: the predictive component does not need to internalize the whole decision policy. However, the paper concerns a narrow scheduling service and does not offer broad modern controlled evidence for open-ended agent tasks.[12]

Our proposed extension: put authorization outside utility maximization as a hard constraint. Among permitted options, compare useful action, reversible preparation, clarification, and waiting. Do not allow predicted benefit to stand in for permission.

## 4. BDI: useful theory must survive computational constraints

PDF pp. 2–3 motivate beliefs as incomplete environment information, desires as objectives, and intentions as currently selected commitments. Intentions avoid the extremes of recomputing a plan at every environmental change and blindly executing to completion.[13]

PDF pp. 4–5 discuss different commitment and termination policies rather than one uniquely correct agent logic. The architecture uses explicit state and constrained update/query operations. The paper explicitly distinguishes an ideal logical architecture from an efficient implementation: logically closed states and general provability are not a practical runtime interpreter.[13]

PDF pp. 5–6 implement a restricted representation with current ground beliefs, plan invocation conditions/preconditions, and intention stacks. Event handling determines when commitments are reconsidered. The OASIS application is described as undergoing parallel evaluation with live radar data; the paper is not a modern randomized performance comparison.[13]

Our assessment: this is a precedent for formal ideas shaping executable state transitions, not a prescription to run a general modal theorem prover on every model turn. Also, BDI 'intention' is the agent's adopted plan; the user's latent intent is a separate object. A Diaspy design should not overload the same field for both.

## 5. CIRL: user interaction is evidence and teaching, with restrictive assumptions

Sections 3.1–3.2, pp. 4–5: a cooperative human–robot game shares the human's reward, while only the human initially observes the static reward parameter. An optimal joint-policy computation reduces to a POMDP; under the stated assumptions, current state plus the robot's reward belief suffices for an optimal policy pair.[14]

Sections 3.3–3.4 and appendix: demonstration optimal for accomplishing the immediate task need not be optimal for teaching the robot. The human can trade immediate reward for a demonstration that better communicates preferences. Section 4 studies grid navigation with linear feature rewards and constructed demonstration policies, not a longitudinal personal assistant with evolving human goals.[14]

Section 5 explicitly warns that computing the joint policy effectively assumes both actors can be made to follow a coordination policy, which is infeasible for real humans.[14]

Our assessment: infer user goals from communicative context, not just surface imitation of behavior. A correction can be a deliberate teaching act. But the paper assumes static reward parameters known to the human; it does not resolve goals that are uncertain even to the user or formed during interaction. 'Maintain revisable hypotheses' is a design implication, not a claim that CIRL solves bespoke intent.

## 6. ACE: incremental context adaptation is useful, but not belief revision

Read the full extracted 23-page v1 PDF, including prompts; no code or reported results reproduced. Sections 3.1–3.2 use itemized bullets with identifiers and helpful/harmful metadata, model-generated deltas, and deterministic merging. This separates semantic lesson generation from bookkeeping and avoids full-context rewriting.[10]

Table 2 provides a useful counterexample to unconditional self-improvement: online FiNER without ground-truth feedback scores 67.3 versus base 70.7. The authors explicitly warn that unreliable feedback can pollute context (Section 4.4 and Appendix B). Table 3 shows that the headline online AppWorld result includes offline warmup; the no-warmup variant is reported separately. Do not present the headline as cold-start learning from one bespoke interaction.[10]

The published curator prompts expose ADD operations, while the main text discusses refinement, counter updates, and deduplication. Neither the paper's structured bullets nor helpful/harmful counts establish a dependency-aware, truth-preserving belief-revision system. This is our interpretation: adopting delta updates is sensible, but scope, contradictory evidence, revision, and authorization need their own semantics.

## 7. Agent Execution Records: close schema precedent, not established faithfulness

Read the full extracted 8-page v1 PDF. Section 4 records identity/authority, retrieved context, versioned plans, per-step intent/observation/inference, and verdict evidence links. Retrieval provenance means what actually entered context, not merely what was accessible.[16]

Section 3.3 explicitly acknowledges self-report, rationalization, format gaming, and model-style differences. Section 7 is a planned evaluation methodology, not completed evidence that this instrumentation improves decisions. Section 6.3 labels its storage comparison stylized and preliminary.[16]

Our interpretation: contemporaneous typed annotations are easier to query than unstructured logs, but neither timing nor typing proves causal faithfulness. The example's DNS resolution evidence is also used to support broader reachability/network conclusions (Sections 4.4–4.5); a well-formed pointer does not by itself justify the inference. An executable harness should validate relevant evidence and preconditions rather than trusting the annotation alone.

Mock replay in Section 5 reuses recorded tool results under a new model/prompt.[16] Our proposed caution: if a revised policy selects different actions, recorded outputs may not answer those actions. Use recorded-output replay for compatible branches; stop at divergence or execute a sandbox transition model rather than inventing counterfactual observations.

## 8. PROV-AGENT: input/output lineage across agentic and ordinary computation

Read the full extracted 7-page v1 PDF. Section III extends W3C PROV with AI agents, model invocations, prompts/responses, tool activities, and domain data. Section IV describes Flowcept decorators and model wrappers that connect these to ordinary workflow tasks. Its query examples trace upstream inputs and downstream affected outputs.[17]

The evaluation is preliminary and uses a distributed workflow mimicking an additive-manufacturing loop. The live sensor-to-simulation link is explicitly still under development. This is a provenance and queryability demonstration, not a controlled demonstration of fewer hallucinations or better epistemic decisions.[17]

Our interpretation: reuse lineage concepts to distinguish source observations, model interpretations, and resulting artifacts. This supplies a substrate for revision and audit, not the policy deciding which claims to believe or actions to permit.

## Synthesis: epistemology as a control protocol, not a decorative trace

This section is an assistant proposal, not an implemented result or an agreed roadmap.

Separate three layers:

1. Semantic proposals: model-generated candidate intent interpretations, alternative actions, possible tests, and predictions. These remain fallible.
2. Executable epistemic procedures: evidence provenance, scope and freshness, dependency tracking, explicit contradictions/defeaters, commitment revision, and policies for seeking information.
3. Authority and execution: permitted actions, state-changing tool calls, independent readback, and recorded outcomes. Permission must not be inferred from confidence or predicted benefit.

Candidate invariants:

- An inferred preference does not silently become an explicit user instruction.
- New authoritative correction supersedes a conflicting old hypothesis within its scope; dependent plans are flagged for reconsideration rather than leaving the correction isolated in a memory note.
- Synthetic examples can test a policy or expose a failure mode, but do not become observations about this user's preferences.
- Multiple dependent retellings of one source do not count as independent corroboration.
- Consequential execution requires scoped authorization regardless of a proposal's apparent utility.
- A completion claim must point to the evidence appropriate for that claim, not merely to the agent's report that it succeeded.

These make formal commitments inspectable without claiming to expose private model cognition. They do not make semantic interpretation infallible: a correctly typed but false observation, mistaken entailment judgment, or omitted alternative can still defeat the system. Separate schema validity, evidence authenticity, semantic adequacy, and behavioral benefit.

The crucial experimental distinction is logging versus control. Compare the same agent without structured protocol, with structured records but no additional control, and with records whose contents actually constrain updates and actions. Hold model, tools, information access, memory access, and budget comparable; include a compute-matched extra-deliberation control. Test both stale/contradictory intent and legitimate changes of mind, not only fixed hidden requirements. Evaluate useful outcomes, user correction/rework, missed opportunities, unnecessary questions, authority violations, latency, and cost separately rather than prematurely choosing a single score.

A direct causal test is to change one evidence or permission item while holding the rest fixed and check whether the corresponding decision changes. If records can be deleted or contradicted with no behavioral consequence, the purported epistemic harness may be only an annotation layer.

Sources:
[9] https://arxiv.org/pdf/2605.14678v1
[10] https://arxiv.org/pdf/2510.04618v1
[12] https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/chi99horvitz-1.pdf
[13] https://cdn.aaai.org/ICMAS/1995/ICMAS95-042.pdf
[14] https://arxiv.org/pdf/1606.03137v4
[15] https://arxiv.org/pdf/2410.12361v3
[16] https://arxiv.org/pdf/2603.21692v1
[17] https://arxiv.org/pdf/2508.02866v1
