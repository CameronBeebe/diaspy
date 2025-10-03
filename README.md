[Pair coded prototype with grok, minimum functionality and testing -CB]

# Masterpiece DSPy Script with Grok

This project contains a Python script that demonstrates a novel multi-agent dialectic workflow using DSPy with xAI's Grok model. The script creates an optimized program for generating witty, truthful responses to user queries through a thesis-antithesis-synthesis process, inspired by Grok's personality and Hegelian dialectics.

## Prerequisites

- Python 3.8+
- xAI API Key: Set the environment variable `XAI_API_KEY` with your key.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your API key:
   ```
   export XAI_API_KEY='your-api-key-here'
   ```

## Running the Script

Run the script:
```
python masterpiece.py
```

Follow the prompts to enter queries and receive responses.

## Novelty

The script implements a novel multi-agent system in DSPy with three agents: ThesisAgent for initial response, AntithesisAgent for counterpoints, and SynthesisAgent for synthesis. It employs Chain-of-Thought reasoning for each and optimizes the entire dialectic responder using BootstrapFewShot with staged examples. This creates a self-improving, dialectic responder that leverages Grok's truth-seeking nature in a unique way by simulating philosophical debate for more nuanced responses. 

## Detailed Specification of the Multi-Agent Dialectic Workflow

### Philosophical Foundations
This workflow draws inspiration from philosophical traditions emphasizing open dialogue and debate as mechanisms for truth-seeking:

- **Hegelian Dialectics**: Thesis-antithesis-synthesis process for reconciling opposing views.
- **John Stuart Mill's On Liberty**: Advocates free expression and debate, arguing that truth emerges from the collision of adverse opinions, refining partial truths through confrontation.
- **Liberal Thought and Argumentation Theory**: Thinkers like Karl Popper emphasize falsifiability and critical discussion; modern argumentation theory (e.g., Toulmin model) structures claims with grounds, warrants, and rebuttals to foster robust reasoning.
- **Cybernetics and Systems Theory**: Concepts like negative feedback (e.g., in Norbert Wiener's cybernetics) relate to falsification and corrective processes in a control-theoretic way. The antithesis acts as negative feedback, correcting or challenging the thesis to stabilize towards a more accurate synthesis, mirroring self-regulating systems.

These foundations underpin the system's use of multi-agent debate to generate nuanced, truthful responses.

### Purpose
The multi-agent dialectic workflow is designed to generate nuanced, witty, and truthful responses to user queries by simulating a dialectic process inspired by these traditions. This approach produces more balanced and insightful responses than a single-agent system, leveraging Grok's personality for humor and truth-seeking.

### How It Works
1. **Thesis Agent**: Receives the user's query and generates an initial witty, truthful response (thesis) using Chain-of-Thought reasoning.
2. **Antithesis Agent**: Takes the query and thesis as input, producing a counterpoint (antithesis) that offers an alternative perspective, maintaining wit and truthfulness.
3. **Synthesis Agent**: Combines the query, thesis, and antithesis to create a final synthesized response that balances both views.
4. **Critic Agent** (New): Critiques the synthesis for accuracy, wit, and balance, providing negative feedback. If quality is low, triggers refinement via another antithesis iteration (cybernetic feedback loop).

Each agent is a DSPy module with its own signature defining inputs and outputs. The DialecticResponder orchestrates the agents in sequence, with up to 2 iterations for refinement.

### Relation to Philosophical Foundations
- **Hegelian Dialectics**: Core thesis-antithesis-synthesis flow.
- **Mill's On Liberty**: Simulated debate refines truths through adversarial collision.
- **Popper/Argumentation Theory**: CriticAgent enforces falsifiability by checking claims.
- **Cybernetics**: Feedback loop uses critique as negative feedback for self-correction.

A custom metric optimizes for truthfulness, wit, and balance during compilation.

### API Calls and Historical Context
The workflow involves multiple API calls to the Grok model (via OpenAI-compatible API):
- One call for thesis generation.
- One for antithesis, incorporating the thesis as historical context.
- One for synthesis, using query, thesis, and antithesis as context.

This ensures each step builds on previous outputs, maintaining historical context across the dialectic process. The entire system is optimized using BootstrapFewShot with few-shot examples for each stage, allowing self-improvement.

### Testing Considerations
When testing, compare outputs against this spec:
- Does the thesis directly address the query wittily?
- Does the antithesis provide a genuine counterpoint?
- Does the synthesis balance both while remaining truthful and engaging?
- Verify multiple API calls occur with proper context passing. 

## Expansion to Multi-Agent Debate and Group of Experts

Based on the current DSPy prototype in `masterpiece.py` (which implements a binary dialectic with Thesis, Antithesis, Synthesis, and Critic agents using Grok via DSPy.LM), we can leverage DSPy's modular architecture to simulate more advanced multi-agent behaviors. This enables emergent \"multi-agent\" dynamics through sequential or parallel chaining, iteration, and feedback loops—mimicking deliberation, debate, and expert consensus.

### Key Principles for Achievement in DSPy
1. **Modularity and Chaining**: Each \"agent\" is a DSPy Module (e.g., using ChainOfThought or Predict). We can chain them sequentially (for debate) or in parallel (for experts) within a higher-level module like an expanded DialecticResponder.
2. **Iteration for Debate**: Simulate back-and-forth debate by looping between \"pro\" and \"con\" modules, using previous outputs as context (via input fields). Add a Critic to score and decide when to stop (e.g., after convergence or a max iterations).
3. **Parallelism for Group of Experts**: Run multiple specialized expert modules concurrently on the same query, then aggregate outputs via a synthesizer module. DSPy's `dspy.Predict` can handle this by invoking modules in a list or loop.
4. **Feedback and Optimization**: Extend the existing CriticAgent with multi-output scoring. Use BootstrapFewShot (as in the current code) to compile all new modules with few-shot examples, ensuring they learn from dialectic-style training data.
5. **Integration**: Build on the existing DialecticResponder by adding modes (e.g., \"binary\", \"debate\", \"experts\") via parameters. This keeps the system extensible while reusing the Grok LM backend.
6. **Efficiency**: Limit iterations (e.g., 3-5 for debate) to avoid excessive API calls. Use caching in DSPy.settings if needed.
7. **Advantages Over Native Multi-Agent Systems**: DSPy allows explicit optimization via metrics (e.g., extending `philosophical_metric` to score debate coherence or expert diversity), making it \"better\" in controllability and trainability.

### Potential Challenges and Mitigations
- **Context Management**: Long chains can exceed token limits; mitigate by summarizing prior outputs in input fields.
- **Training Data**: Need to expand the `trainset` with examples for debate rounds and expert inputs.
- **Evaluation**: Extend the metric to include \"diversity\" (for experts) or \"resolution\" (for debates).

### Sketched Solution: Additional Modules, Signatures, and Functions

#### 1. Debate Extension (For Multi-Round Back-and-Forth)
This adds a \"debate mode\" with multiple pro/con agents iterating until convergence, building on the binary thesis-antithesis.

- **New Signatures**:
  - `ProArgumentSignature`: Generates supporting arguments for a position.
    - Inputs: `query` (str), `current_position` (str, e.g., thesis or prior argument), `opposing_arguments` (str, summary of cons).
    - Output: `pro_argument` (str, witty/truthful support).
  - `ConArgumentSignature`: Generates counterarguments.
    - Inputs: `query` (str), `current_position` (str), `supporting_arguments` (str).
    - Output: `con_argument` (str, alternative perspective).

- **New Modules**:
  - `ProDebateAgent(dspy.Module)`: Uses ChainOfThought on ProArgumentSignature.
  - `ConDebateAgent(dspy.Module)`: Uses ChainOfThought on ConArgumentSignature.

- **Pipeline Function** (Add to DialecticResponder):
  ```python
  def run_debate(self, query, initial_thesis, max_rounds=3):
      # Start with initial thesis
      current_position = initial_thesis
      debate_history = [f\"Thesis: {initial_thesis}\"]
      
      for round in range(max_rounds):
          # Generate con (antithesis-like)
          con_arg = self.con_debate_agent(query=query, current_position=current_position, supporting_arguments=\"\".join(debate_history))
          debate_history.append(f\"Con {round+1}: {con_arg}\")
          
          # Critique and check convergence
          critique, score = self.critic_agent(query=query, thesis=initial_thesis, antithesis=con_arg, synthesis=current_position)  # Reuse existing critic
          if score >= 0.9:  # High threshold for convergence
              break
          
          # Generate pro rebuttal
          pro_arg = self.pro_debate_agent(query=query, current_position=current_position, opposing_arguments=con_arg)
          current_position = pro_arg  # Update position
          debate_history.append(f\"Pro {round+1}: {pro_arg}\")
      
      # Final synthesis
      final_synthesis = self.synthesis_agent(query=query, thesis=initial_thesis, antithesis=\"\".join(debate_history))
      return dspy.Prediction(debate_history=debate_history, synthesis=final_synthesis)
  ```
  - **Integration**: In `forward`, add a mode param (e.g., if mode==\"debate\", call `run_debate` after generating thesis).
  - **Compilation**: Add debate-specific examples to `trainset` (e.g., multi-round arguments), then compile Pro/Con agents like the others.

#### 2. Group of Experts Extension (For Parallel Specialized Inputs)
This adds an \"experts mode\" where multiple domain-specific agents provide inputs, then synthesize for a collective response.

- **New Signature**:
  - `ExpertOpinionSignature`: Provides specialized insight.
    - Inputs: `query` (str), `expertise_domain` (str, e.g., \"science\", \"philosophy\", \"humor\"), `context` (str, optional prior synthesis).
    - Output: `opinion` (str, witty/truthful take from that domain).

- **New Module**:
  - `ExpertAgent(dspy.Module)`: Uses ChainOfThought on ExpertOpinionSignature. Instantiate multiple times with different domains.

- **Pipeline Function** (Add to DialecticResponder):
  ```python
  def run_experts(self, query, domains=[\"science\", \"philosophy\", \"humor\"], max_iterations=2):
      # Parallel expert opinions
      expert_opinions = {}
      for domain in domains:
          opinion = self.expert_agent(query=query, expertise_domain=domain, context=\"\")  # One instance per domain
          expert_opinions[domain] = opinion
      
      # Initial synthesis of opinions
      combined_context = \"\\n\".join([f\"{domain}: {op}\" for domain, op in expert_opinions.items()])
      synthesis = self.synthesis_agent(query=query, thesis=combined_context, antithesis=\"\")  # Reuse synthesis
      
      # Feedback loop
      for _ in range(max_iterations):
          critique, score = self.critic_agent(query=query, thesis=combined_context, antithesis=\"\", synthesis=synthesis)
          if score >= 0.8:
              break
          # Refine by re-querying experts with critique
          for domain in domains:
              refined_op = self.expert_agent(query=query, expertise_domain=domain, context=critique)
              expert_opinions[domain] = refined_op
          combined_context = \"\\n\".join([f\"{domain}: {op}\" for domain, op in expert_opinions.items()])
          synthesis = self.synthesis_agent(query=query, thesis=combined_context, antithesis=\"\")
      
      return dspy.Prediction(expert_opinions=expert_opinions, synthesis=synthesis)
  ```
  - **Integration**: In `forward`, add a mode param (e.g., if mode==\"experts\", call `run_experts`). Allow user to specify domains via CLI.
  - **Compilation**: Add expert-specific examples to `trainset` (e.g., domain-tagged responses), compile a shared ExpertAgent.

#### Extended Metric for Optimization
Update `philosophical_metric` to handle new modes:
- Add checks for \"debate resolution\" (e.g., if arguments converge) or \"expert diversity\" (e.g., varied opinions).
- Example: `debate_score = len(debate_history) > 2 and 'resolved' in synthesis.lower()`

This extension builds directly on the existing prototype and can be integrated into the package's responders module. 

## Introducing 'diaspy': A Package for Multi-LLM Dialectical Workflows

This repository is evolving into '**diaspy**', a robust Python package for building multi-agent and dialectical LLM workflows using DSPy. Inspired by philosophical traditions (e.g., Hegelian dialectics, Mill's emphasis on debate, Popper's falsifiability, and cybernetic feedback), 'diaspy' enables many LLM instances—potentially of the same or different models—to engage in structured argumentative interactions and achieve epistemologically grounded functions like thesis-antithesis-synthesis, multi-round debates, and group-of-experts consensus.

At its core, 'diaspy' solves the **AI management problem**: In a world of diverse, specialized models (where no single model excels at everything due to the 'no free lunch' theorem), we need orchestration tools to manage and integrate them effectively. By providing modular signatures, agents, responders, and pipelines, 'diaspy' replaces archaic direct interactions with single LLMs, evolving DSPy's paradigm into scalable, philosophically inspired multi-LLM systems. This allows for emergent intelligence through deliberation, feedback loops, and optimization—ultimately producing more nuanced, truthful, and balanced outputs.

The original 'masterpiece.py' script serves as a prototype, which will be modularized into the package. Future expansions will support heterogeneous LLMs, custom interaction patterns, and advanced metrics for epistemological rigor. 

# diaspy: Dialectical LLM Workflows with DSPy

![diaspy Logo](path/to/logo.png)  <!-- Add if you have one -->

**diaspy** is a Python package that leverages DSPy to create multi-agent dialectical workflows for LLMs. It addresses common LLM shortcomings (e.g., hallucinations, prompt sensitivity) through philosophically inspired processes like thesis-antithesis-synthesis, adversarial debates, and expert consultations.

Inspired by Hegelian dialectics, Popper's falsifiability, and cybernetic feedback, diaspy promotes truth-oriented AI interactions.

## Features

- **Modular Agents**: Thesis, Antithesis, Synthesis, Critic, Pro/Con Debate, Expert.
- **Responder Modes**: Binary (classic dialectic), Debate (adversarial), Experts (multi-domain).
- **DSPy Optimization**: Bootstrap few-shot learning with philosophical metric.
- **CLI Interface**: Interactive command-line tool.
- **Extensible**: Customizable for specific domains or LLMs.

## Installation

```bash
pip install -e .
```

Or from PyPI (once published):

```bash
pip install diaspy
```

### Dependencies
- dspy-ai
- Python >=3.8

Set your xAI API key:

```bash
export XAI_API_KEY='your-key'
```

## Quickstart

Run the CLI:

```bash
diaspy
```

Enter a query and select a mode (binary, debate, experts).

### Programmatic Usage

```python
import dspy
from diaspy.utils import compile_agents, trainset
from diaspy.responders import DialecticResponder

# Configure DSPy
grok = dspy.LM(model="xai/grok-3-mini", api_key=os.environ['XAI_API_KEY'])
dspy.settings.configure(lm=grok)

# Compile and create responder
compiled = compile_agents(trainset)
responder = DialecticResponder(**compiled)

# Example: Binary mode
prediction = responder(query="What is consciousness?", mode='binary')
print(prediction.synthesis)
```

See `examples/diaspy_demo.py` for more (copy sections into a Jupyter notebook).

## Documentation

- **Agents** (`src/diaspy/agents.py`): Core modules for dialectical components.
- **Signatures** (`src/diaspy/signatures.py`): DSPy input/output definitions.
- **Responder** (`src/diaspy/responders.py`): Orchestrates modes with refinement loops.
- **Utils** (`src/diaspy/utils.py`): Training data, metrics, compilation.

Run tests:

```bash
pytest
```

## Contributing

1. Fork the repo.
2. Create a branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m 'Add feature'`.
4. Push: `git push origin feature-name`.
5. Pull request.

## Copyright

Copyright (c) 2025 The SciPhi Initiative, LLC. All rights reserved.

This package is proprietary and not licensed for external use without permission. 