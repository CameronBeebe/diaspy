import dspy
from .signatures import (
    ThesisSignature,
    AntithesisSignature,
    SynthesisSignature,
    CriticSignature,
    ProArgumentSignature,
    ConArgumentSignature,
    ExpertOpinionSignature,
)

class ThesisAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ThesisSignature)

    def forward(self, query):
        return self.generate(query=query).thesis

class AntithesisAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(AntithesisSignature)

    def forward(self, query, thesis):
        return self.generate(query=query, thesis=thesis).antithesis

class SynthesisAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SynthesisSignature)

    def forward(self, query, thesis, antithesis):
        return self.generate(query=query, thesis=thesis, antithesis=antithesis).synthesis

class CriticAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CriticSignature)

    def forward(self, query, thesis, antithesis, synthesis):
        prediction = self.generate(query=query, thesis=thesis, antithesis=antithesis, synthesis=synthesis)
        try:
            if isinstance(prediction.score, float):
                score = prediction.score
            else:
                score_str = str(prediction.score).strip()
                if '/' in score_str:
                    num, den = map(float, score_str.split('/'))
                    score = num / den
                else:
                    score = float(score_str)
            # Clamp score to [0.0, 1.0]
            score = max(0.0, min(1.0, score))
        except (ValueError, AttributeError):
            score = 0.5
        return prediction.critique, score

class ProDebateAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ProArgumentSignature)

    def forward(self, query, current_position, opposing_arguments):
        return self.generate(query=query, current_position=current_position, opposing_arguments=opposing_arguments).pro_argument

class ConDebateAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ConArgumentSignature)

    def forward(self, query, current_position, supporting_arguments):
        return self.generate(query=query, current_position=current_position, supporting_arguments=supporting_arguments).con_argument

class ExpertAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ExpertOpinionSignature)

    def forward(self, query, expertise_domain, context=''):
        return self.generate(query=query, expertise_domain=expertise_domain, context=context).opinion
