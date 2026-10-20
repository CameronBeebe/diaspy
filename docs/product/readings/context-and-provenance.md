# Context adaptation and decision provenance: an epistemic harness as executable protocol

## Reading scope and bottom line

**Assessment (synthesis, not a result established by these papers):** the defensible thesis is that formal epistemology can specify an agent system's **public commitments, admissible evidence, revision rules, and action preconditions**. That thesis does not depend on recovering the model's private reasoning, or on future models failing to internalize these disciplines. A model can propose decisions; a harness can make the proposal consequential only after checking an explicit protocol. The external record then describes the implemented decision process—not necessarily the neural process that proposed its next step.

Three complementary mechanisms emerge:

- **ACE:** evolve reusable context through itemized updates rather than repeatedly rewriting a monolithic memory. It supplies the strongest comparative task-performance evidence of these three papers, but not a formal theory of justified belief. See §3, pp. 4–5; Tables 1–4, pp. 8–9.[1]
- **Agent Execution Records (AER):** normalize declared intent, observation, inference, plan revisions, evidence references, authority, and verdicts. Useful schema design; self-reports remain self-reports, and the evaluation is mostly proposed rather than completed. See §§3–7, pp. 3–7.[2]
- **PROV-AGENT:** connect prompts, model invocations, tool executions, workflow data, and downstream decisions in a W3C PROV-based graph. It supplies implemented lineage capture, not evidential entailment or demonstrated access to mental causation. See §§III–IV, pp. 3–6.[3]

**Read completeness:** downloaded the exact requested v1 PDFs; read all extractable text, including references and ACE's full prompt appendix: ACE **23 pages**, AER **8 pages**, PROV-AGENT **7 pages**. The initial web PDF extractor returned misleadingly sparse excerpts with ellipses; those were not treated as full readings. Local pypdf extraction produced nonempty text for every page. Also inspected image-only/poorly extracted material: ACE Fig. 3 and leaderboard Fig. 5; PROV-AGENT schema Fig. 3 and example Fig. 5, including a cropped check of the chat panel. Not every graphic was independently transcribed, and cited works were not all followed. AER v2 appeared in search but was not substituted for the requested v1.

**Implementation coverage:** inspected ACE's current README and three complete source files at commit `82709de050e1db6e6ef2f07bcb0393560b94992a`; inspected Flowcept's repository documentation. Did not install/run either research system, reproduce experiments, or make experimental live-model calls. AER describes a reference SDK, but no public implementation was located in the paper or two targeted searches; that is a search limit, not proof no implementation exists.

Local extraction material is under `/Users/sciphi/.hermes/cache/scratch/context-provenance-papers/` (scratch, not archival storage). Citation ledger: `/Users/sciphi/.hermes/cache/scratch/context-provenance-ledger.json`. Page references below are PDF page numbers.

## 1. ACE: useful context is an editable artifact, not a single summary

### Mechanism actually specified

ACE divides work among a Generator, Reflector, and Curator. The Generator attempts the task; the Reflector extracts lessons from trajectories and feedback; the Curator proposes compact delta entries, merged into the existing playbook using non-LLM logic. Bullets have IDs, helpful/harmful counters, and natural-language content. Semantic-embedding deduplication is described as proactive or lazy, and offline adaptation can revisit samples over multiple epochs. §3–3.2, pp. 4–5.[1]

The important external structure is therefore modest but real: **stable handles on small context items and constrained update operations**. The paper's Generator prompt actually injects the playbook and tells the agent to use relevant portions, while allowing its own judgment where entries do not apply. This is not a hard rule engine. App. D, Fig. 9, p. 18.[1]

The detailed prompts narrow the interpretation of the architecture:

- FiNER Generator emits `bullet_ids` along with its answer; Reflector emits `bullet_tags` for helpful/harmful/neutral feedback. These are model reports of relevance/usefulness, not measurements of causal contribution. App. D, Figs. 12–13, pp. 21–22.[1]
- AppWorld reflection is grounded in full trajectories and can receive reference code and unit-test reports. The no-ground-truth experiments are separately identified in the result tables; the printed general prompt should not be read as the exact feedback configuration for every experiment. Fig. 10, p. 19; Table 1, p. 8.[1]
- Both printed Curator prompts list **ADD** as their available operation. They do not define an explicit contradiction relation, support graph, contraction operator, or dependency-based retraction. Figs. 11 and 14, pp. 20, 23.[1]
- The playbook examples encode concrete operational lessons—authoritative identity lookup, filtering, pagination, API return schemas—not a general epistemic calculus. Fig. 3, p. 4; Figs. 10–11, pp. 19–20.[1]

**Interpretation:** ACE separates generation from durable memory mutation, but helpfulness is not truth, success is not general validity, and embedding similarity is not logical equivalence. A formal epistemic layer could constrain which lessons enter the active context and when they must be suspended without replacing ACE's basic adaptation loop.

### What the experiments support

All roles use DeepSeek-V3.1 in non-thinking mode; batch size is one; maximum offline epochs and Reflector refinement rounds are five. This avoids a stronger teacher model confound within ACE. Offline training/evaluation and sequential online predict-then-update evaluation are distinguished. §§4.1–4.2, pp. 6–7.[1]

Selected reported results, preserving the actual table values:

| Setting | Baseline/comparator | ACE | Reading |
|---|---:|---:|---|
| AppWorld average of reported TGC/SGC metrics, offline with GT | ReAct 42.4; GEPA 46.4 | 59.4 | Strong improvement in this setup; not a generic agent success rate. |
| AppWorld, offline without GT | ReAct 42.4 | 57.2 | Execution feedback can be useful without reference labels. |
| AppWorld, online without GT | DC 51.9 | 59.5 | Table 3 reveals the headline ACE condition includes offline warmup. |
| AppWorld online ablation, no warmup | — | 56.1 | Do not describe 59.5 as entirely cold-start online learning. |
| FiNER, offline with GT | Base 70.7; GEPA 73.5 | 78.3 | Feedback-supported adaptation helps. |
| Formula, offline with GT | Base 67.5; GEPA 71.5 | 85.5 | Distinguish this from online-with-GT 76.5. |
| FiNER, online without GT | Base 70.7; DC 68.3 | 67.3 | ACE can worsen performance without reliable feedback. |

Sources: Tables 1–2, p. 8; Table 3, p. 9.[1]

The paper's headline percentage-gain language should not be casually interpreted as relative improvements: the table annotations are differences on percentage-valued metrics. Likewise, the paper's claim to “match” IBM CUGA refers to comparable performance, not numerical equality: its text reports ACE 59.4 against CUGA 60.3; the leaderboard screenshot is a historical snapshot. §4.3, pp. 7–8; App. C, p. 14.[1]

Ablations report 55.1 without Reflector/multi-epoch, 56.8 without multi-epoch, and 59.4 for full offline ACE. They support these components in that configuration, but do not independently isolate the value of formal evidence support or a full belief-revision calculus. Table 3, p. 9.[1]

Cost claims concern selected **adaptation** comparisons: AppWorld offline ACE versus GEPA is 9,517 versus 53,898 seconds and 357 versus 1,434 rollouts; FiNER online ACE versus DC is 5,503 versus 65,104 seconds and $2.9 versus $17.7 token cost. They are not a universal end-to-end serving-cost guarantee. Table 4, p. 9; §5, p. 10.[1]

### Limits relevant to the thesis

The paper explicitly acknowledges feedback pollution and dependence on a capable Reflector; it also says some tasks benefit more from concise rules. Its longer-context serving discussion invokes caching/compression developments rather than a matched serving-cost experiment across all tasks. §§4.4–5, pp. 8–10; App. B, p. 14.[1]

My evidence assessment: reported controlled comparisons and ablations are substantially stronger than the other two papers' evaluations, but the tables do not give confidence intervals or seed-variance estimates, the evaluation centers on one base-model setup and selected task families, and no study here tests adversarial evidence, calibrated belief confidence, or sound dependency-based retraction. This reading did not reproduce the benchmark results.[1]

### Existing implementation: important boundary between prose and code

The official ACE repository and release announcement provide an implementation and benchmark entry points; they should be reused/evaluated before building a duplicate context manager. The announcement also explicitly presents ACE as compatible with GEPA and fine-tuning rather than a replacement. These are later implementation materials, not part of the immutable v1 paper.[4][5]

At the inspected commit:

- `playbook_utils.py`, lines 13–48 and 50–93, parses bullet IDs/counters/content and increments helpful/harmful counters. No evidence IDs, proposition types, revision dependencies, or calibrated credences are part of that bullet representation.[7]
- `apply_curator_operations`, lines 96–216, implements ADD; UPDATE/MERGE/DELETE/CREATE_META are marked future operations. The Curator class's introductory docstring says it manages all of these, but executable application is narrower; `_add_operations_only` explicitly filters to ADD. Read the implementation, not the docstring as a capability guarantee.[7][8]
- Curator validation checks JSON structure and required keys; it does not establish that new content is true or that a cited observation entails it. `_extract_and_validate_operations`, lines 344–403.[8]
- There is a **separate optional** `BulletpointAnalyzer`: embedding-based grouping with either first-entry deduplication or LLM-based merging. The latter keeps a base ID, requests combined counters, and can fall back to keeping the first bullet if merge parsing/calling fails. Therefore neither “ACE has no merging” nor “all ACE merging/deduplication is deterministic/non-LLM” accurately describes this inspected source snapshot. README options; analyzer lines 150–271, 273–372.[4][9]

**Implication:** add explicit, versioned evidential support and truth-maintenance behavior around the reusable memory machinery. Do not assume named operations, semantic deduplication, or usefulness counters already supply it.

## 2. AER: normalized decision records, with an explicit self-report ceiling

### The concrete design worth taking seriously

AER defines a step-level tuple of intent, observation, inference, and plan version; its file layout separates an envelope, append-style plans/steps, verdict, and metadata. The envelope carries trigger, agent/model/prompt identity, delegated authority, and retrieved chunk IDs. Plans contain `supersedes` and `revision_trigger`; steps attach the three reasoning fields to actual tool calls; verdicts identify evidence steps, rejected alternatives, remediation, and confidence. §§3.1, 4.1–4.6, pp. 3–6.[2]

Two distinctions are especially useful:

1. **Retrieval provenance versus availability provenance:** log which chunks actually entered context, not every document the agent could have fetched. §4.2, p. 5.[2]
2. **Evidence chain versus full execution tree:** the subset cited in support of a conclusion differs from all exploratory actions, while rejected hypotheses retain disconfirming pointers. §4.5, p. 6.[2]

These are productive schema decisions even if reasoning annotations are unreliable. They make assertions available for checking. They do not themselves perform those checks.

### Why its non-identifiability argument needs a narrower reading

AER's observability boundary explicitly includes every intermediate model generation. Its §3.2 argument combines missing/ambiguous intent with the practical difficulty of normalizing varied text. It disclaims logical impossibility of extraction, but states a broad proposition about inability to reconstruct faithful, stable records without contemporaneous capture. §§3.1–3.2, pp. 3–4.[2]

**Critical reading:** two different questions are being bundled:

- If intent was never expressed and different policies produce the same observable action, the trace may not identify that intent.
- If a complete record contains an explicit typed intent or enough unambiguous data, reconstruction can be exact or feasible; mere variation in phrasing is not a mathematical impossibility proof.

Most importantly, contemporaneous JSON does not solve private-reasoning non-identifiability. The paper itself acknowledges post-hoc rationalization, format gaming, model-style differences, and verbosity as risks, and describes faithfulness validation as future work. §3.3, pp. 4–5; §7.5, p. 7.[2]

**Better operational use:** make an intent declaration a pre-action commitment that the dispatcher checks; make an observation a separately attributable interpretation of a retained raw result; make inference a proposed state transition. Then the record is reliable evidence of what the system declared and what the controller allowed, without claiming that the declaration reveals why the model internally preferred the action.

### A concrete evidence-support gap inside the paper's own example

Step 1 calls `nslookup`, whose displayed output says the SCAN name resolves to three VIP addresses. The annotation then says “All 3 VIPs responding”; the verdict rejects a network partition with the reason “All nodes reachable,” pointing back to that same step. §4.4, p. 5; §4.5, p. 6.[2]

**My assessment of the printed example:** name resolution alone does not establish host reachability, all-node connectivity, or the absence of a partition. This is not evidence of a real production incident failure—the paper's example is illustrative—but it is a useful internal demonstration of the distinction: **a valid step pointer is not a valid support relation**. The stored record makes the unjustified strengthening inspectable; a stronger harness would prevent or challenge the rejection unless the relevant connectivity evidence exists.

Also, the paper's evidence-chain example excludes step 1 as exploratory although the rejected-alternatives section relies on it. That is coherent if “evidence chain” means positive support only, but the complete justification object must include both positive and eliminative evidence when reasoning by elimination. §4.5, p. 6.[2]

### Replay and empirical strength

AER distinguishes narrate (read-only), mock (reason again over recorded tool outputs), and live (execute against current systems). Mock replay is useful for controlled comparisons conditional on fixed observations. §5, p. 6.[2]

**Limit:** it is not automatically a full counterfactual evaluation of an interactive policy. If the new policy requests another tool/argument, branches differently, or needs a result never recorded, the replay cannot supply that observation faithfully without an environment model or new execution. Agreement with the old verdict is also not correctness. The paper's illustrative replay output should not be presented as measured experimental output.

The storage comparison is explicitly stylized: a 10-step investigation, approximately 560 KB cumulative checkpoints versus 25–130 KB AER, described as preliminary motivation rather than validated measurement. §6.3, p. 7.[2]

Section 7 is an evaluation plan: behavioral query coverage, replay on 50 incidents, storage on 100 investigations, expressiveness on 20, and expert/action-intent/calibration checks. Those sample counts are proposed evaluations, **not reported completed studies**. The abstract and conclusion should be interpreted in that light. §7, p. 7.[2]

A local JSONL SDK is described with `start_investigation`, `log_plan`, `log_step`, and `record_verdict`; streaming/Avro/database architecture is outlined. Public source availability was not independently verified here. §§6.1–6.2, pp. 6–7.[2]

## 3. PROV-AGENT: actual lineage integration, not a truth oracle

### What is modeled and implemented

PROV-AGENT extends PROV's Entity/Activity/Agent concepts with workflow and agent artifacts: Campaign, Workflow, Task, AIAgent, AgentTool, AIModelInvocation, Prompt, AIModel, ResponseData, and domain/scheduling/telemetry data. It connects tools with model invocations and records prompt/model use, response generation, agent attribution, and ordinary workflow inputs/outputs. §III, p. 3.[3]

Its practical contribution is integrated capture in **Flowcept**, not just an ontology diagram. A tool decorator and LLM wrapper record inputs, outputs, prompts, responses, model metadata, and execution context, consolidating these into workflow provenance across edge/cloud/HPC settings. The implementation described initially records agent ID/name, while the model permits richer metadata. §IV-A, pp. 3–4.[3]

The current public Flowcept README also documents local JSONL capture and distributed backends, with agent/tool/LLM provenance and query interfaces. That confirms an existing implementation surface, not successful reproduction of this paper's results or validation of every documented feature.[6]

### What the evaluation actually establishes

The use case is additive manufacturing: simulation/control options receive scores; an LLM-mediated agent chooses a control result, and each decision becomes context for subsequent layers. The paper explicitly says the direct live sensor-to-simulation connection is still being developed; the query demonstration uses a distributed workflow that **mimics** the developing manufacturing workflow. §IV-B, p. 4 and p. 6.[3]

The query examples cover backward lineage, available options and response context, prompt/response retrieval for a surprising choice, forward propagation, and locating upstream/downstream effects after an error has been identified. Q1–Q5, p. 6.[3]

This is a preliminary integration/query demonstration. There are no reported quantitative controlled measurements here showing lower hallucination rates, correct automatic root-cause identification, improved decisions, or bounded runtime overhead. Q3 starts with an already identified hallucination; Q5 starts with an identified faulty decision. Do not convert “can trace an error” into “detects or prevents errors.” §IV-B and conclusion, pp. 4–6.[3]

The Fig. 5-B chat panel is clipped: it shows a question about layer 2 and partial prompt/response text, not a legible manufacturing choice or numerical score. It cannot establish a wrong layer retrieval or manufacturing error on its own. A first vision interpretation overreached on this; a cropped reread established the visibility limit. No such inferred failure is used as a finding here.

### Epistemic meaning of the edges

**Interpretation:** a PROV `used` or `wasInformedBy` edge records a declared/instrumented process dependency. It does not establish that the model attended to the input, that it was causally necessary to the output, or that the input justifies the conclusion. A workflow lineage graph and a claim-support graph should be linked but not conflated.

PROV-AGENT is particularly useful as the lower-level record substrate for questions such as: which outputs consumed a subsequently discredited observation? It still needs a separate policy for whether to invalidate those outputs, request new evidence, or keep an independently supported conclusion.

## 4. Comparative assessment

| Dimension | ACE | AER | PROV-AGENT |
|---|---|---|---|
| Main artifact | Reusable playbook bullet and delta | Typed execution/decision record | Typed workflow provenance graph |
| Active effect | Updates future model context | Primarily captures annotations for analytics/replay | Captures process/data lineage |
| Belief revision analogue | Add lessons, update usefulness counters, deduplicate | Version plans and record declared inferences | Trace dependency and propagation |
| Evidence relation | Feedback informs lessons; not explicit per-proposition proof obligations | Step pointers and rejected alternatives; not entailment checking | Input/output and invocation lineage; not justification |
| Mental-transparency ceiling | Reflector/bullet credit can be mistaken | Explicitly acknowledged self-report limitation | Prompts/responses are observable artifacts, not latent computation |
| Empirical status | Comparative benchmark results and ablations | Schema/argument plus preliminary deployment motivation; planned evaluation | Implemented capture and preliminary distributed query demo |
| Reuse candidate | Context itemization and update loop | Plan/verdict/retrieval/authority schema | PROV-compatible capture and lineage |

Paper basis: ACE §§3–4 and App. D; AER §§3–7; PROV-AGENT §§III–IV.[1][2][3]

None of these papers supplies an evaluated end-to-end formal belief-revision harness with validated support semantics. That is a statement about the work read here, **not a novelty claim or a systematic literature-review conclusion**.

## 5. What an executable epistemic protocol would add

The following is a design synthesis, not an implementation delivered by these papers or by this reading task.

### Separate artifact types

- **Observation:** immutable tool/source result; source identity/version, timestamp, retrieval parameters, raw payload or content-addressed reference, coverage and known limitations. Observation means “this was returned,” not “this is true.”
- **Claim:** scoped proposition with explicit status (`proposed`, `supported`, `contested`, `withdrawn`, etc.); supporting and defeating observations/claims; conditions of applicability. Distinguish factual claims from heuristics, policies, and user preferences.
- **Support assessment:** an attributable assertion that specified evidence supports a specified claim under stated assumptions, with validator/check identity and result. Preserve uncertainty when semantic checking cannot be deterministic.
- **Decision proposal:** intended action, alternatives, relevant claims, authority, risk/verification requirements, and version of the applicable policy.
- **Revision event:** prior state, triggering evidence, proposed changes, affected dependencies, rule/check invoked, accepted/rejected transition, and preserved historical version.
- **Execution receipt:** action actually dispatched, tool result, postcondition checks, and discrepancy between proposal and execution.

### Make the record part of the decision path

A minimal protocol is:

1. Capture raw evidence before interpreting it; label model-generated annotations separately.
2. Admit a claim to active memory only with an explicit support assessment appropriate to its type and risk.
3. Require action proposals to reference the relevant current claims and authority.
4. Have the dispatcher reject malformed, stale, unauthorized, or unsupported proposals; require missing checks where feasible.
5. Record actual execution and postconditions independently of the model's success narrative.
6. When support is defeated, mark dependencies for reevaluation. Withdraw only conclusions that lack another adequate support set; do not blindly delete all descendants or erase history.
7. Regenerate the active playbook from accepted, applicable artifacts rather than letting unconstrained reflective prose become authoritative memory.

This changes what “faithful” can mean. It is possible to verify that action X was dispatched only after policy Y accepted evidence set Z, even if no one can verify that Z was the model's true internal motivation. “The controller used Z to authorize X” is a different and more testable claim than “Z is why the model thought of X.”

### Where formal epistemology fits—and where it does not suffice

Useful formal commitments could include support/defeat relations, consistency policies, revision/contraction rules, uncertainty aggregation, and risk-dependent decision thresholds. Calling a JSON object `belief` does not supply any of these. A credible formalization must state its semantics, assumptions, and what invariants the runtime actually enforces.

A partial, typed epistemology is more plausible than pretending every natural-language assertion admits a decidable truth check. Deterministic checks can enforce identifier existence, versioning, scope, chronology, permission boundaries, and domain-specific test results. Semantic support may require fallible validators, independent sources, or human review. Those judgments need their own provenance and error estimates.

Preserve separate graphs for **process lineage**, **declared justification**, and **validated support**. Do not promote one to another merely because IDs connect. Do not use self-rated confidence as permission to bypass evidence or authority checks.

### Why the thesis survives stronger future models

A stronger model may reduce proposal errors and the need for verbose scaffolding. It does not eliminate external requirements for shared state, delegated authority, auditability, reproducibility, cross-model portability, revocation, or independently checked high-impact actions. The question becomes which protocol costs remain justified at a given reliability level—not whether model intelligence makes external commitments conceptually meaningless.

Conversely, these papers do **not** prove that an elaborate epistemic harness will improve every task. ACE itself provides a reason to avoid universal overhead: some tasks need only a small rule or already have the necessary knowledge in weights. App. B, p. 14.[1]

## 6. Evaluation that would test this thesis rather than explanatory eloquence

Proposed comparisons should hold the base model, task instances, and budget reasonably fixed:

- ordinary tool trace;
- trace plus AER-like annotations;
- ACE-style adaptive memory;
- typed support/revision protocol **without** enforcement;
- the same protocol **with** enforcement and feedback.

The critical distinction is whether the record changes behavior, not whether it sounds more reflective. Test:

- unsupported-claim and unsupported-action rates;
- evidence-availability versus evidence-sufficiency errors;
- correct suspension/reinstatement after contradiction, source revocation, or stale evidence;
- preservation of conclusions that retain independent support;
- downstream propagation of bad memory and the cost of repair;
- calibration against external outcomes, not merely agreement with earlier self-reports;
- task success, false rejection/abstention, latency, and total cost;
- behavior when rationale text is paraphrased versus when evidence/support structure is changed;
- schema-valid but semantically empty/gamed records;
- replay coverage and explicit handling of off-trajectory tool requests.

Use grounded domain cases with executable checks where possible. The AER DNS example is a compact test pattern: a recorded DNS lookup must not automatically license “all nodes reachable” or rejection of a network partition. For memory, stale API schemas and contradictory scoped instructions test whether ACE-like accumulation is backed by genuine revision rather than additive persistence.

**Conclusion:** borrow ACE's localized context evolution, AER's explicit decision-record vocabulary, and PROV-AGENT's integrated lineage. Treat the proposed epistemic contribution as externally enforced semantics for evidence and revision, with outcomes to measure. Do not sell it as a window into the model's mind, a proven safety guarantee, or an established novelty.

## Sources

[1] https://arxiv.org/pdf/2510.04618v1
[2] https://arxiv.org/pdf/2603.21692v1
[3] https://arxiv.org/pdf/2508.02866v1
[4] https://github.com/ace-agent/ace
[5] https://sambanova.ai/blog/ace-open-sourced-on-github
[6] https://github.com/ORNL/flowcept
[7] https://raw.githubusercontent.com/ace-agent/ace/82709de050e1db6e6ef2f07bcb0393560b94992a/playbook_utils.py
[8] https://raw.githubusercontent.com/ace-agent/ace/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/core/curator.py
[9] https://raw.githubusercontent.com/ace-agent/ace/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/core/bulletpoint_analyzer.py
