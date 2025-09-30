import dspy

class ThesisSignature(dspy.Signature):
    """Generate an initial thesis response to the query, grounded in logical reasoning and truth-seeking."""

    query: str = dspy.InputField()
    thesis: str = dspy.OutputField()

class AntithesisSignature(dspy.Signature):
    """Generate a counterpoint or antithesis to the given thesis, providing an alternative perspective grounded in logical reasoning and truth-seeking."""

    query: str = dspy.InputField()
    thesis: str = dspy.InputField()
    antithesis: str = dspy.OutputField()

class SynthesisSignature(dspy.Signature):
    """Synthesize the thesis and antithesis into a final, balanced response grounded in logical reasoning and truth-seeking."""

    query: str = dspy.InputField()
    thesis: str = dspy.InputField()
    antithesis: str = dspy.InputField()
    synthesis: str = dspy.OutputField()

class CriticSignature(dspy.Signature):
    """Critique the synthesis for factual accuracy, logical consistency, and balance, providing a score and feedback. Inspired by Popper's falsifiability and cybernetic negative feedback. Output score as a decimal float between 0.0 and 1.0."""

    query: str = dspy.InputField()
    thesis: str = dspy.InputField()
    antithesis: str = dspy.InputField()
    synthesis: str = dspy.InputField()
    critique: str = dspy.OutputField()
    score: float = dspy.OutputField(desc="Decimal float between 0.0 and 1.0")

class ProArgumentSignature(dspy.Signature):
    """Generate supporting arguments for a position in a debate, maintaining logical reasoning and truthfulness."""

    query: str = dspy.InputField()
    current_position: str = dspy.InputField()
    opposing_arguments: str = dspy.InputField()
    pro_argument: str = dspy.OutputField()

class ConArgumentSignature(dspy.Signature):
    """Generate counterarguments against a position in a debate, providing alternative perspectives grounded in logical reasoning and truth-seeking."""

    query: str = dspy.InputField()
    current_position: str = dspy.InputField()
    supporting_arguments: str = dspy.InputField()
    con_argument: str = dspy.OutputField()

class ExpertOpinionSignature(dspy.Signature):
    """Provide specialized insight from a given expertise domain, ensuring responses grounded in logical reasoning and truth-seeking."""

    query: str = dspy.InputField()
    expertise_domain: str = dspy.InputField()
    context: str = dspy.InputField()
    opinion: str = dspy.OutputField()
