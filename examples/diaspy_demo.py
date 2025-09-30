# =============================================================================
# CELL 1: MARKDOWN - Introduction
# =============================================================================
"""
# Diaspy Demo: Dialectical LLM Workflows

This notebook demonstrates the core functionality of the `diaspy` package, which uses DSPy and multi-agent dialectical processes to improve LLM reliability by reducing hallucinations and prompt sensitivity.

## Core Concepts

- **Thesis-Antithesis-Synthesis**: Hegelian dialectics for balanced reasoning
- **Multi-Agent Debate**: Adversarial truth-oriented processes  
- **Expert Consultation**: Domain-specific perspectives
- **Iterative Refinement**: Critic-driven feedback loops

Let's explore the three main modes: `binary`, `debate`, and `experts`.
"""

# =============================================================================
# CELL 2: PYTHON - Setup and Configuration
# =============================================================================
# Import necessary modules
import os
import dspy
from diaspy.responders import DialecticResponder
from diaspy.utils import compile_agents, trainset
from diaspy.agents import ThesisAgent, AntithesisAgent, SynthesisAgent, CriticAgent

# Setup environment (you need to set your XAI_API_KEY)
api_key = os.environ.get('XAI_API_KEY')
if not api_key:
    print("Warning: XAI_API_KEY not set. Please set it before running the cells below.")
    print("export XAI_API_KEY='your-api-key-here'")
else:
    print("API key found, setting up DSPy with Grok-3-mini...")
    grok = dspy.LM(model="xai/grok-3-mini", api_key=api_key, cache=False)
    dspy.settings.configure(lm=grok)
    print("✓ DSPy configured successfully")

# =============================================================================
# CELL 3: MARKDOWN - Individual Agents Section
# =============================================================================
"""
## 1. Individual Agents

First, let's explore the individual agents that form the foundation of our dialectical process.
"""

# =============================================================================
# CELL 4: PYTHON - Test Individual Agents
# =============================================================================
# Test individual agents before compilation
if api_key:
    # Create uncompiled agents for initial testing
    thesis_agent = ThesisAgent()
    antithesis_agent = AntithesisAgent()
    synthesis_agent = SynthesisAgent()
    critic_agent = CriticAgent()
    
    # Test query
    test_query = "What is the nature of consciousness?"
    
    print("=== Testing Individual Agents ===")
    print(f"Query: {test_query}\n")
    
    # Generate thesis
    thesis = thesis_agent(test_query)
    print(f"Thesis: {thesis}\n")
    
    # Generate antithesis
    antithesis = antithesis_agent(test_query, thesis)
    print(f"Antithesis: {antithesis}\n")
    
    # Generate synthesis
    synthesis = synthesis_agent(test_query, thesis, antithesis)
    print(f"Synthesis: {synthesis}\n")
    
    # Critique the synthesis
    critique, score = critic_agent(test_query, thesis, antithesis, synthesis)
    print(f"Critique: {critique}")
    print(f"Score: {score}\n")
else:
    print("Skipping agent tests - API key not configured")

# =============================================================================
# CELL 5: MARKDOWN - Compiled Agents Section
# =============================================================================
"""
## 2. Compiled Agents and Full Responder

Now let's compile the agents using DSPy's optimization and create the full dialectical responder.
"""

# =============================================================================
# CELL 6: PYTHON - Compile Agents and Create Responder
# =============================================================================
# Compile agents using training data and create responder
if api_key:
    print("Compiling agents with DSPy optimization...")
    print("This may take a moment as agents are optimized with training examples...\n")
    
    # Compile all agents
    compiled_agents = compile_agents(trainset)
    print("✓ Agents compiled successfully")
    
    # Create the dialectical responder
    responder = DialecticResponder(**compiled_agents)
    print("✓ DialecticResponder created")
    
    # Show what training data looks like
    print(f"\nTraining set contains {len(trainset)} examples")
    print("Sample training example:")
    sample = trainset[0]
    print(f"  Query: {sample.query}")
    print(f"  Thesis: {sample.thesis}")
else:
    print("Skipping compilation - API key not configured")

# =============================================================================
# CELL 7: MARKDOWN - Binary Mode Section
# =============================================================================
"""
## 3. Mode 1: Binary Dialectic

The binary mode implements classic Hegelian dialectics: thesis → antithesis → synthesis, with iterative refinement based on critic feedback.
"""

# =============================================================================
# CELL 8: PYTHON - Test Binary Mode
# =============================================================================
# Test Binary Mode
if api_key:
    print("=== Binary Dialectic Mode ===")
    
    query = "Is artificial intelligence a threat to humanity?"
    print(f"Query: {query}\n")
    
    # Run binary dialectic
    prediction = responder(query=query, mode='binary', max_iterations=2)
    
    print("Results:")
    print(f"Thesis: {prediction.thesis}\n")
    print(f"Antithesis: {prediction.antithesis}\n") 
    print(f"Final Synthesis: {prediction.synthesis}\n")
    
    print("Critique History:")
    for i, critique in enumerate(prediction.critiques):
        print(f"  Iteration {i+1}: {critique}\n")
    
    print(f"Total iterations: {len(prediction.critiques)}")
else:
    print("Skipping binary mode test - API key not configured")

# =============================================================================
# CELL 9: MARKDOWN - Debate Mode Section
# =============================================================================
"""
## 4. Mode 2: Debate Mode

The debate mode simulates adversarial discussion with alternating pro/con arguments, ending in synthesis.
"""

# =============================================================================
# CELL 10: PYTHON - Test Debate Mode
# =============================================================================
# Test Debate Mode
if api_key:
    print("=== Debate Mode ===")
    
    query = "Should we pursue genetic engineering of humans?"
    print(f"Query: {query}\n")
    
    # Run debate mode
    prediction = responder(query=query, mode='debate', max_rounds=3, max_iterations=2)
    
    print("Debate History:")
    for i, argument in enumerate(prediction.debate_history):
        print(f"  {i+1}. {argument}\n")
    
    print(f"Final Synthesis: {prediction.synthesis}\n")
    print(f"Total debate rounds: {len(prediction.debate_history)}")
else:
    print("Skipping debate mode test - API key not configured")

# =============================================================================
# CELL 11: MARKDOWN - Expert Mode Section
# =============================================================================
"""
## 5. Mode 3: Expert Consultation

The experts mode gathers opinions from different domain specialists and synthesizes them into a comprehensive response.
"""

# =============================================================================
# CELL 12: PYTHON - Test Expert Mode
# =============================================================================
# Test Expert Mode
if api_key:
    print("=== Expert Consultation Mode ===")
    
    query = "How can we address climate change effectively?"
    domains = ['science', 'economics', 'policy', 'technology']
    print(f"Query: {query}")
    print(f"Expert domains: {domains}\n")
    
    # Run expert consultation mode
    prediction = responder(query=query, mode='experts', domains=domains, max_iterations=2)
    
    print("Expert Opinions:")
    for domain, opinion in prediction.expert_opinions.items():
        print(f"  {domain.capitalize()}: {opinion}\n")
    
    print(f"Synthesized Response: {prediction.synthesis}\n")
    print(f"Domains consulted: {len(prediction.expert_opinions)}")
else:
    print("Skipping expert mode test - API key not configured")

# =============================================================================
# CELL 13: MARKDOWN - Comparative Analysis Section
# =============================================================================
"""
## 6. Comparative Analysis

Let's compare how the same query is handled across different modes to see the value of dialectical processes.
"""

# =============================================================================
# CELL 14: PYTHON - Comparative Analysis
# =============================================================================
# Comparative analysis across modes
if api_key:
    print("=== Comparative Analysis ===")
    
    test_query = "What are the ethical implications of artificial intelligence?"
    print(f"Query: {test_query}\n")
    
    modes_to_test = ['binary', 'debate', 'experts']
    results = {}
    
    for mode in modes_to_test:
        print(f"Running {mode} mode...")
        try:
            if mode == 'experts':
                prediction = responder(query=test_query, mode=mode, domains=['ethics', 'technology', 'philosophy'])
            else:
                prediction = responder(query=test_query, mode=mode, max_iterations=1)
            
            results[mode] = prediction
            print(f"✓ {mode} completed")
        except Exception as e:
            print(f"✗ {mode} failed: {e}")
            results[mode] = None
    
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    
    for mode, prediction in results.items():
        if prediction:
            print(f"\n{mode.upper()} MODE:")
            print("-" * 30)
            
            if hasattr(prediction, 'synthesis'):
                print(f"Final Output: {prediction.synthesis}")
            elif hasattr(prediction, 'thesis'):
                print(f"Thesis: {prediction.thesis}")
            
            # Show unique characteristics of each mode
            if mode == 'binary' and hasattr(prediction, 'critiques'):
                print(f"Critiques: {len(prediction.critiques)}")
            elif mode == 'debate' and hasattr(prediction, 'debate_history'):
                print(f"Debate rounds: {len(prediction.debate_history)}")
            elif mode == 'experts' and hasattr(prediction, 'expert_opinions'):
                print(f"Expert domains: {list(prediction.expert_opinions.keys())}")
else:
    print("Skipping comparative analysis - API key not configured")

# =============================================================================
# CELL 15: MARKDOWN - Custom Usage Section
# =============================================================================
"""
## 7. Custom Usage Examples

Here are some examples of how to use diaspy components in your own applications.
"""

# =============================================================================
# CELL 16: PYTHON - Custom Usage Examples
# =============================================================================
# Custom usage examples
if api_key:
    print("=== Custom Usage Examples ===\n")
    
    # Example 1: Using specific compiled agents
    print("1. Using individual compiled agents:")
    thesis_agent_compiled = compiled_agents['thesis']
    antithesis_agent_compiled = compiled_agents['antithesis']
    
    query = "What is the purpose of education?"
    thesis = thesis_agent_compiled(query)
    antithesis = antithesis_agent_compiled(query, thesis)
    
    print(f"Query: {query}")
    print(f"Compiled Thesis: {thesis}")
    print(f"Compiled Antithesis: {antithesis}\n")
    
    # Example 2: Custom responder configuration
    print("2. Custom responder with specific parameters:")
    custom_prediction = responder(
        query="Should we colonize Mars?",
        mode='binary',
        max_iterations=3  # More iterations for higher quality
    )
    print(f"High-iteration synthesis: {custom_prediction.synthesis}\n")
    
    # Example 3: Batch processing multiple queries
    print("3. Batch processing multiple queries:")
    queries = [
        "What is the meaning of friendship?",
        "How should we approach aging?",
        "What makes art valuable?"
    ]
    
    batch_results = []
    for i, q in enumerate(queries):
        try:
            result = responder(query=q, mode='binary', max_iterations=1)
            batch_results.append(result.synthesis)
            print(f"Query {i+1}: {q}")
            print(f"Result: {result.synthesis[:100]}...\n")
        except Exception as e:
            print(f"Query {i+1} failed: {e}\n")
            batch_results.append(None)
    
    print(f"Processed {len([r for r in batch_results if r])} out of {len(queries)} queries successfully")
else:
    print("Skipping custom usage examples - API key not configured")

# =============================================================================
# CELL 17: MARKDOWN - Performance Analysis Section
# =============================================================================
"""
## 8. Performance and Quality Metrics

Let's examine how the dialectical process improves response quality over single-agent approaches.
"""

# =============================================================================
# CELL 18: PYTHON - Performance Analysis
# =============================================================================
# Performance analysis
if api_key:
    from diaspy.utils import philosophical_metric
    
    print("=== Performance and Quality Analysis ===\n")
    
    test_query = "How can technology enhance human well-being?"
    
    # Compare single agent vs dialectical process
    print("Comparing single-agent vs dialectical approaches:")
    print(f"Query: {test_query}\n")
    
    # Single agent (just thesis)
    single_response = compiled_agents['thesis'](test_query)
    single_score = philosophical_metric(None, single_response)
    
    print("SINGLE AGENT APPROACH:")
    print(f"Response: {single_response}")
    print(f"Quality Score: {single_score:.3f}\n")
    
    # Dialectical process
    dialectical_prediction = responder(query=test_query, mode='binary', max_iterations=2)
    dialectical_score = philosophical_metric(None, dialectical_prediction)
    
    print("DIALECTICAL APPROACH:")
    print(f"Final Synthesis: {dialectical_prediction.synthesis}")
    print(f"Quality Score: {dialectical_score:.3f}")
    print(f"Iterations Used: {len(dialectical_prediction.critiques)}\n")
    
    # Score improvement with zero handling
    improvement = dialectical_score - single_score
    if single_score == 0:
        improvement_pct = float('inf') if improvement > 0 else 0.0
        print(f"Quality Improvement: {improvement:.3f} (infinite % increase since base score was 0)")
    else:
        improvement_pct = (improvement / single_score) * 100
        print(f"Quality Improvement: {improvement:.3f} ({improvement_pct:.1f}% increase)")
    
    # Show the dialectical process components
    print("\nDialectical Process Breakdown:")
    print(f"Thesis: {dialectical_prediction.thesis}")
    print(f"Antithesis: {dialectical_prediction.antithesis}")
    print(f"Synthesis: {dialectical_prediction.synthesis}")
    if dialectical_prediction.critiques:
        print(f"Final Critique: {dialectical_prediction.critiques[-1]}")
else:
    print("Skipping performance analysis - API key not configured")

# =============================================================================
# CELL 19: MARKDOWN - Next Steps Section
# =============================================================================
"""
## 9. Next Steps and Extensions

This demo covers the core functionality of diaspy. Here are some areas for further exploration:

### Immediate Extensions
- **Custom Training Data**: Add domain-specific examples to improve performance
- **Custom Metrics**: Develop specialized quality metrics for your use case  
- **Model Configuration**: Experiment with different LLMs and parameters
- **Batch Processing**: Scale up to handle multiple queries efficiently

### Advanced Applications
- **Research Analysis**: Use expert mode for multi-disciplinary research questions
- **Decision Support**: Apply dialectical processes to complex decision making
- **Content Generation**: Leverage debate mode for balanced content creation
- **Quality Assurance**: Use critic agents as automated quality checkers

### Integration Patterns
- **API Wrapper**: Create REST APIs around dialectical responders
- **Streaming Responses**: Implement real-time dialectical conversations
- **Caching Layer**: Add response caching for improved performance
- **Multi-Modal**: Extend to handle images, documents, and other media

Try modifying the examples above or creating your own dialectical workflows!
"""

# =============================================================================
# END OF DEMO FILE
# ============================================================================= 