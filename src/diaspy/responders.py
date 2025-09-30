import dspy
from .agents import (
    ThesisAgent,
    AntithesisAgent,
    SynthesisAgent,
    CriticAgent,
    ProDebateAgent,
    ConDebateAgent,
    ExpertAgent,
)

class DialecticResponder(dspy.Module):
    def __init__(self, thesis, antithesis, synthesis, critic, pro_debate=None, con_debate=None, expert=None):
        super().__init__()
        self.thesis_agent = thesis
        self.antithesis_agent = antithesis
        self.synthesis_agent = synthesis
        self.critic_agent = critic
        self.pro_debate_agent = pro_debate or ProDebateAgent()
        self.con_debate_agent = con_debate or ConDebateAgent()
        self.expert_agent = expert or ExpertAgent()

    def forward(self, query, mode='binary', max_iterations=2, domains=None, max_rounds=3):
        if mode == 'binary':
            return self._run_binary(query, max_iterations)
        elif mode == 'debate':
            return self._run_debate(query, max_rounds, max_iterations)
        elif mode == 'experts':
            domains = domains or ['science', 'philosophy', 'humor']
            return self._run_experts(query, domains, max_iterations)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _run_binary(self, query, max_iterations):
        thesis = self.thesis_agent(query)
        antithesis = self.antithesis_agent(query, thesis)
        synthesis = self.synthesis_agent(query, thesis, antithesis)
        critiques = []
        for _ in range(max_iterations):
            critique, score = self.critic_agent(query, thesis, antithesis, synthesis)
            critiques.append(critique)
            if score >= 0.8:
                break
            antithesis = self.antithesis_agent(query, thesis + '\nCritique: ' + critique)
            synthesis = self.synthesis_agent(query, thesis, antithesis)
        return dspy.Prediction(thesis=thesis, antithesis=antithesis, synthesis=synthesis, critiques=critiques)

    def _run_debate(self, query, max_rounds, max_iterations):
        thesis = self.thesis_agent(query)
        current_position = thesis
        debate_history = [f"Thesis: {thesis}"]
        for round_num in range(max_rounds):
            con_arg = self.con_debate_agent(query=query, current_position=current_position, supporting_arguments='\n'.join(debate_history))
            debate_history.append(f"Con {round_num+1}: {con_arg}")
            critique, score = self.critic_agent(query=query, thesis=thesis, antithesis=con_arg, synthesis=current_position)
            if score >= 0.9:
                break
            pro_arg = self.pro_debate_agent(query=query, current_position=current_position, opposing_arguments=con_arg)
            current_position = pro_arg
            debate_history.append(f"Pro {round_num+1}: {pro_arg}")
        synthesis = self.synthesis_agent(query=query, thesis=thesis, antithesis='\n'.join(debate_history))
        return dspy.Prediction(debate_history=debate_history, synthesis=synthesis)

    def _run_experts(self, query, domains, max_iterations):
        expert_opinions = {}
        for domain in domains:
            opinion = self.expert_agent(query=query, expertise_domain=domain, context='')
            expert_opinions[domain] = opinion
        combined_context = '\n'.join([f"{domain}: {op}" for domain, op in expert_opinions.items()])
        synthesis = self.synthesis_agent(query=query, thesis=combined_context, antithesis='')
        for _ in range(max_iterations):
            critique, score = self.critic_agent(query=query, thesis=combined_context, antithesis='', synthesis=synthesis)
            if score >= 0.8:
                break
            for domain in domains:
                refined_op = self.expert_agent(query=query, expertise_domain=domain, context=critique)
                expert_opinions[domain] = refined_op
            combined_context = '\n'.join([f"{domain}: {op}" for domain, op in expert_opinions.items()])
            synthesis = self.synthesis_agent(query=query, thesis=combined_context, antithesis='')
        return dspy.Prediction(expert_opinions=expert_opinions, synthesis=synthesis)
