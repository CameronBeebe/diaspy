# Diaspy: direction, components, and next experiment

Participants: Cameron and Hermes. Started 2026-10-19; updated through the request to consolidate Diaspy, beliefs, judges, and Jev-like components.
Status: ongoing discussion. User positions and assistant recommendations are separated. Notes and recommendations are authorized; no implementation, publication, or roadmap approval is implied.

## 1. North star — Cameron's positions

- **Bespoke work first:** wording, memory, context, and execution conditions matter; semantic similarity does not make requests interchangeable. Do not assume a stable distribution of repeated tasks.
- **Intent representation → intent anticipation:** an exocortex should "reliably predict and build in anticipation and in a reasonable direction consistent with the user's likely intent."
- **Inspectability:** retain direct access to code, evidence, traces, and scientific direction; use DSPy/Python naturally. Passing tests and AI summaries are not substitutes for scrutiny.
- **Explicit theory:** typed, theory-driven procedures may improve on unconstrained critique and rewriting. GEPA is a possible component/testbed, not necessarily the product definition.
- **Effective decisions:** examine consequential in-harness behavior without requiring access to authentic private reasoning. Performance, interpretability, and compute efficiency remain distinct objectives.

Correction preserved: the assistant initially overemphasized repeated tasks and later overemphasized realized outcomes as prerequisites for epistemic improvement.

## 2. Components — conceptual roles, not current implementation claims

### Diaspy: orchestration and experimental setting
- Integrate proposal generation, critique, revision, admission, execution, and trace inspection.
- Existing discussion includes Popperian criticism and Millian/adversarial alternatives; these are candidate procedures, not demonstrated performance advantages.
- Compare structured procedures with simpler DSPy programs, ordinary reflection, and compute-matched alternatives. Improvements may become code, tests, evidence, demonstrations, or instructions—not just prompt rewrites.

### Beliefs: a small DSPy-like module — Cameron's proposal
- Elicit the relevant propositions, purported facts, and credences for a particular execution.
- Make those commitments available for inspection, comparison, and possibly revision before action.
- Optional comparison with a Jev-like estimator; disagreement may provide a diagnostic of belief structure.
- The proposed connection to latent-space belief structure is a research conjecture, not an established interpretation. Reported beliefs are not automatically verified facts or faithful neural measurements.

### Jev-like / Clairic-like adapters: structured estimation
- Jev-like component: estimate probabilities over specified propositions/options from context, alongside the main model's reported credences.
- Clairic-like component: supply typed extracted values, source locations, and numerical estimates/distributions.
- Keep extraction confidence, proposition probability, predicted action success, and action utility distinct. Align proposition, evidence, and time before comparing estimators.
- External estimators are comparators, not truth oracles. Permit substitution; no vendor is a required foundation.

### Formal epistemic checks: constraints on commitments
- Check supported logical relationships among beliefs and consider their consequences across admissible possible worlds.
- Separate honest elicitation, coherence, and empirical calibration. Internal expected-loss reasoning can improve reports before any event resolves.
- Track evidence, scope, contradictions, and revision dependencies. Synthetic worlds are hypothetical, not observations about the real user.
- Valid types and coherent probabilities do not establish truth, complete evidence, or adequate semantic interpretation.

### Judges / judgekit: evaluation and adjudication
- Cameron identifies judges and judgekit as part of the wider set of ideas.
- **Proposed role:** evaluate evidence support, semantic adequacy, relevance, and task outcomes where deterministic checks are insufficient; preserve rubric, evidence, verdict, and disagreement.
- Prefer code for logical/numerical checks and human review for ambiguous intent. Judge outputs remain fallible assessments, not ground truth.
- Earlier detailed judgekit discussions were not recovered in the available history search. This role is an assistant recommendation, not a claim about judgekit's existing APIs or agreed architecture.

### Adversarial structures: targeted challenges
- Critics propose counterexamples, alternative interpretations, defeaters, or rival actions; defenders identify what remains supported.
- Critique should attach to explicit claims and evidence rather than reward persuasive debate or majority agreement alone.
- Escalate selectively when ambiguity or consequence warrants the cost; the hypothesis that this improves decisions remains to be tested.

## 3. de Finetti correction — central, not a footnote

**Cameron's point:** "ACTUAL contact with the world is not needed in order to minimize the score by ones own lights." Expected scoring consequences can motivate honest reporting and coherent commitments before outcomes occur; realized penalties are not required.

**Distinctions preserved:**
- Proper scoring: under one's own probability assessment, honest reporting minimizes expected loss when that score is the relevant objective.
- Coherence: some inconsistent sets of reports can be improved across every admissible outcome, without learning which outcome occurs.
- Empirical calibration: agreement of reported probabilities with observed frequencies is a different question; internal coherence does not establish it.
- Simulated-world deliberation requires explicit constraints and, for expectation calculations, appropriate probability weights. Arbitrary generated examples are not automatically samples from the agent's beliefs.
- Repeating one's own credences through a scoring formula is not new evidence. The research opportunity lies in eliciting distributed commitments, exposing incompatibilities, and changing decisions—not decorative arithmetic.

Assistant correction: the previous record → resolve → score proposal was incomplete. A beliefs module can do useful ex ante work; later outcome scoring is complementary, not its sole justification.

## 4. Recommended next step — assistant proposal, not adopted roadmap

**Build one narrow vertical slice: execution-local beliefs plus a coherence challenge, inside Diaspy.** Do not start a new general agent framework or simultaneously expand every component.

### First slice
- Pick a bounded decision from a bespoke work episode, with a clear point before action and inspectable supporting context.
- Record a small set of decision-relevant propositions, credences, source references, dependencies, and the proposed action; version revisions rather than overwrite them.
- Begin with confirmed relationships: negation, implication, mutually exclusive/exhaustive alternatives. Separate testing the numerical checker from testing whether a model identified the relationships correctly.
- Return a concrete inconsistency or dominance witness to the model; let it revise the propositions, assumptions, credences, or action. Do not silently project numbers and call that understanding.
- Keep authorization checks separate from confidence and expected utility.

### Minimal comparison
- Ordinary execution.
- Belief elicitation only, to measure the effect and cost of recording.
- Belief elicitation plus executable coherence feedback and revision.
- Compute-matched ordinary reflection, to separate structure from extra deliberation.

Use small controlled cases with known semantics to verify mechanisms, plus a curated set of heterogeneous personal episodes to test usefulness. The former is not a proxy for the whole product; the latter is not enough for broad generalization claims.

### What to inspect
- Validity of the propositions/relations; inconsistency before and after feedback.
- Whether evidence interventions and user corrections change the appropriate commitments and actions.
- Artifact/task quality, user correction effort, unnecessary questions, missed useful actions, latency, tokens, and cost.
- Outcome-based scores only where resolution is valid; do not confuse coherence gains with empirical calibration.
- Preserve raw → checked → challenged → revised → executed traces in ordinary artifacts, with a notebook or thin viewer for review.

### Add other components only after this slice is informative
- **Jev-like comparator:** run in shadow mode on aligned propositions and context, initially without seeing the main model's credences. Study where disagreement predicts errors or useful revisions.
- **judgekit:** reuse existing capabilities after inspecting its actual interfaces; evaluate semantic judgments against code/human checks, not solely against another model.
- **Adversarial roles:** target failures the simple checker cannot address; compare against ordinary reflection before adding full debate.
- **GEPA/other optimization:** optimize a component only after the behavior and evaluation target are stable. Do not optimize the entire evolving product idea at once.
- **Persistent user beliefs/anticipation:** defer until execution-local scope, correction, and revision work reliably.

**Go/no-go:** proceed beyond the slice if it improves meaningful decisions or diagnoses correctable failures at acceptable cost—not merely if it emits cleaner JSON or coherent numbers. If numerical coherence improves but decisions do not, investigate relevance and decision coupling before scaling complexity.

## 5. Evidence and open questions

Detailed reading notes:
- [Proactive assistance, foundations, and synthesis](readings/proactive-and-foundations.md).
- [Context adaptation and provenance](readings/context-and-provenance.md).

Later sources discussed (source claims, not independently reproduced):
- de Finetti/proper scoring: https://statmath.wu.ac.at/~kastner/ISBA/dawid01.pdf
- Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Clairic: https://blog.cloudflare.com/clairic-extraction-model/
- ConfTuner: https://arxiv.org/abs/2508.18847v2
- Behavioral calibration: https://arxiv.org/abs/2512.19920v3
- ForecastBench: https://www.forecastbench.org/docs/

Open: belief selection and elicitation timing; formal semantics and subjective world weights; relevance to actual action; comparator interpretation; handling changing intent; independent judge validation; and whether any diagnostic supports claims about internal representation beyond observable behavior.

The long original discussion remains in [the archive](archive/2026-10-19-product-direction-before-beliefs.md). This is the concise steering summary, synchronized with the local notes for PR #6.
