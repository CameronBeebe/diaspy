import os
import dspy
from dspy.teleprompt import BootstrapFewShot
from .agents import (
    ThesisAgent,
    AntithesisAgent,
    SynthesisAgent,
    CriticAgent,
    ProDebateAgent,
    ConDebateAgent,
    ExpertAgent,
)
from .responders import DialecticResponder

# Example training data (expanded for debate and experts)
trainset = [
    # Thesis examples
    dspy.Example(query="What is the meaning of life?", thesis="The meaning of life, according to existentialists like Sartre, is created by individual choices and actions.").with_inputs('query'),
    dspy.Example(query="Why is the sky blue?", thesis="The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere.").with_inputs('query'),
    dspy.Example(query="What is justice?", thesis="Justice, as per Plato, is the harmonious balance of the soul and society.").with_inputs('query'),

    # Antithesis examples
    dspy.Example(query="What is the meaning of life?", thesis="The meaning of life, according to existentialists like Sartre, is created by individual choices and actions.", antithesis="However, nihilists like Nietzsche argue that life has no inherent meaning, challenging us to create our own values.").with_inputs('query', 'thesis'),
    dspy.Example(query="Why is the sky blue?", thesis="The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere.", antithesis="On a deeper level, the perception of color is subjective, as explored in philosophy of mind.").with_inputs('query', 'thesis'),
    dspy.Example(query="What is justice?", thesis="Justice, as per Plato, is the harmonious balance of the soul and society.", antithesis="Contrastingly, Rawls proposes justice as fairness, emphasizing equality and the veil of ignorance.").with_inputs('query', 'thesis'),

    # Synthesis examples
    dspy.Example(query="What is the meaning of life?", thesis="The meaning of life, according to existentialists like Sartre, is created by individual choices and actions.", antithesis="However, nihilists like Nietzsche argue that life has no inherent meaning, challenging us to create our own values.", synthesis="Reconciling these, meaning emerges from personal creation amid apparent absurdity, blending existential choice with Nietzschean value creation.").with_inputs('query', 'thesis', 'antithesis'),
    dspy.Example(query="Why is the sky blue?", thesis="The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere.", antithesis="On a deeper level, the perception of color is subjective, as explored in philosophy of mind.", synthesis="The blue sky results from physical scattering, yet its perception invites philosophical inquiry into qualia and reality.").with_inputs('query', 'thesis', 'antithesis'),
    dspy.Example(query="What is justice?", thesis="Justice, as per Plato, is the harmonious balance of the soul and society.", antithesis="Contrastingly, Rawls proposes justice as fairness, emphasizing equality and the veil of ignorance.", synthesis="Justice integrates Platonic harmony with Rawlsian fairness, promoting balanced societies through equitable principles.").with_inputs('query', 'thesis', 'antithesis'),

    # Debate examples (simplified)
    dspy.Example(query="Is AI beneficial?", current_position="AI is beneficial for productivity.", opposing_arguments="But it can cause job loss.", pro_argument="While job loss is a concern, AI creates new opportunities and enhances efficiency, leading to net societal gains.").with_inputs('query', 'current_position', 'opposing_arguments'),
    dspy.Example(query="Is AI beneficial?", current_position="AI creates new opportunities.", supporting_arguments="It boosts productivity.", con_argument="However, ethical issues like bias and privacy concerns persist, requiring careful regulation.").with_inputs('query', 'current_position', 'supporting_arguments'),

    # Expert examples
    dspy.Example(query="What is gravity?", expertise_domain="science", context="", opinion="Gravity is the fundamental force described by Newton's law of universal gravitation and Einstein's general relativity.").with_inputs('query', 'expertise_domain', 'context'),
    dspy.Example(query="What is gravity?", expertise_domain="philosophy", context="", opinion="In philosophy, gravity metaphorically represents determinism and the inexorable laws governing existence.").with_inputs('query', 'expertise_domain', 'context'),
]

def philosophical_metric(example, pred, trace=None):
    if isinstance(pred, tuple):
        pred = pred[0]
    pred_str = pred if isinstance(pred, str) else getattr(pred, 'synthesis', '') or getattr(pred, 'opinion', '') or getattr(pred, 'pro_argument', '')
    
    # Improved factor detection with weights
    logical = 1.0 if any(word in pred_str.lower() for word in ['logical', 'reason', 'evidence', 'argument']) else 0.0
    truthful = 1.0 if any(word in pred_str.lower() for word in ['truth', 'fact', 'evidence', 'accurate']) else 0.0
    balanced = 1.0 if len(pred_str) > 50 and any(word in pred_str.lower() for word in ['balance', 'combine', 'reconcile', 'both', 'perspectives']) else 0.0
    
    # Mode-specific
    debate_resolution = 1.0 if hasattr(pred, 'debate_history') and len(pred.debate_history) > 2 and any(word in pred_str.lower() for word in ['resolved', 'conclusion', 'final']) else 0.0
    expert_diversity = 1.0 if hasattr(pred, 'expert_opinions') and len(pred.expert_opinions) > 1 and len(set(pred.expert_opinions.values())) == len(pred.expert_opinions) else 0.0
    
    # Add base coherence score based on length and basic quality
    coherence = min(1.0, len(pred_str) / 200)  # Encourage substantial responses
    
    score_factors = [logical, truthful, balanced, debate_resolution, expert_diversity, coherence]
    raw_score = sum(score_factors) / len(score_factors)
    
    # Ensure minimum score to avoid zero-division issues
    return max(raw_score, 0.1)

def compile_agents(trainset):
    teleprompter = BootstrapFewShot(metric=philosophical_metric)
    
    # Dictionary of agent classes and their corresponding example lists
    agent_configs = {
        'thesis': (ThesisAgent, [ex for ex in trainset if 'thesis' in ex and 'antithesis' not in ex]),
        'antithesis': (AntithesisAgent, [ex for ex in trainset if 'antithesis' in ex and 'synthesis' not in ex]),
        'synthesis': (SynthesisAgent, [ex for ex in trainset if 'synthesis' in ex]),
        'critic': (CriticAgent, [ex.with_inputs('query', 'thesis', 'antithesis', 'synthesis') for ex in [ex for ex in trainset if 'synthesis' in ex]]),
        'pro_debate': (ProDebateAgent, [ex for ex in trainset if 'pro_argument' in ex]),
        'con_debate': (ConDebateAgent, [ex for ex in trainset if 'con_argument' in ex]),
        'expert': (ExpertAgent, [ex for ex in trainset if 'opinion' in ex]),
    }
    
    # Compile all agents using a dictionary comprehension
    compiled_agents = {key: teleprompter.compile(agent_class(), trainset=examples) for key, (agent_class, examples) in agent_configs.items()}
    
    return compiled_agents
