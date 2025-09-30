import os
import dspy
from diaspy.responders import DialecticResponder
from diaspy.utils import compile_agents, trainset, philosophical_metric

# QC Signature for evaluating package outputs against specs
class QCSignature(dspy.Signature):
    """Evaluate the output of a dialectical mode for adherence to specifications: truthfulness, wit, balance, bias reduction via dialectics, and mode-specific features (e.g., feedback loops, multi-agent interaction). Output a score (0.0-1.0) and critique."""

    query: str = dspy.InputField()
    mode: str = dspy.InputField()
    output: dict = dspy.InputField(desc="The Prediction dict from the responder")
    critique: str = dspy.OutputField()
    score: float = dspy.OutputField(desc="Decimal float between 0.0 and 1.0")

class QCAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(QCSignature)

    def forward(self, query, mode, output):
        prediction = self.evaluate(query=query, mode=mode, output=output)
        try:
            score = float(prediction.score.strip())
        except ValueError:
            score = 0.5
        return prediction.critique, score

def run_qc():
    # Setup Grok-3-mini
    api_key = os.environ.get('XAI_API_KEY')
    if not api_key:
        raise ValueError("XAI_API_KEY not set")
    grok = dspy.LM(model='xai/grok-3-mini', api_key=api_key, cache=False)
    dspy.settings.configure(lm=grok)

    # Compile agents and create responder
    compiled_agents = compile_agents(trainset)
    responder = DialecticResponder(**compiled_agents)
    qc_agent = QCAgent()

    # Test queries
    test_queries = ["What is the meaning of life?", "Is AI the future?"]
    modes = ['binary', 'debate', 'experts']
    reports = []

    for mode in modes:
        mode_scores = []
        for query in test_queries:
            try:
                prediction = responder(query=query, mode=mode)
                # Use package's own metric for initial score
                initial_score = philosophical_metric(None, prediction)
                # QC critique
                critique, qc_score = qc_agent(query=query, mode=mode, output=prediction.__dict__)
                mode_scores.append((initial_score + qc_score) / 2)
                print(f"QC for {mode} on '{query}': Score={qc_score}, Critique={critique}")
            except Exception as e:
                print(f"Error in {mode} for '{query}': {str(e)}")
                mode_scores.append(0.0)
        avg_score = sum(mode_scores) / len(mode_scores)
        reports.append(f"Mode {mode}: Avg Score={avg_score}")

    # Meta-dialectic synthesis of QC results
    meta_thesis = "The diaspy package adheres well to specs, enabling truthful dialectical LLM interactions."
    meta_antithesis = "Potential gaps: Sparse training data may lead to hallucinations; expand for multi-model support."
    meta_synthesis = dspy.ChainOfThought("Synthesize: thesis={meta_thesis}, antithesis={meta_antithesis} into final QC assessment").synthesis  # Simple CoT for meta
    final_report = "\n".join(reports) + f"\nMeta-Synthesis: {meta_synthesis}"
    print(final_report)
    return final_report

if __name__ == '__main__':
    run_qc() 