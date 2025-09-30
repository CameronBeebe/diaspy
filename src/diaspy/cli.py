import os
import dspy
from .responders import DialecticResponder
from .utils import compile_agents, trainset

def main():
    api_key = os.environ.get('XAI_API_KEY')
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable is not set.")
    grok = dspy.LM(model="xai/grok-3-mini", api_key=api_key, cache=False)
    dspy.settings.configure(lm=grok)
    compiled_agents = compile_agents(trainset)
    responder = DialecticResponder(**compiled_agents)
    print("Welcome to diaspy: Dialectical LLM Workflows!")
    print("Modes: binary, debate, experts")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("Enter your query: ").strip()
        if query.lower() == 'exit':
            break
        mode = input("Enter mode (binary/debate/experts): ").strip().lower()
        try:
            prediction = responder(query=query, mode=mode)
            if mode == 'binary':
                print(f"Thesis: {prediction.thesis}\n")
                print(f"Antithesis: {prediction.antithesis}\n")
                print(f"Synthesis: {prediction.synthesis}\n")
                for i, crit in enumerate(prediction.critiques):
                    print(f"Critique {i+1}: {crit}\n")
            elif mode == 'debate':
                for arg in prediction.debate_history:
                    print(arg + "\n")
                print(f"Final Synthesis: {prediction.synthesis}\n")
            elif mode == 'experts':
                for domain, op in prediction.expert_opinions.items():
                    print(f"{domain.capitalize()}: {op}\n")
                print(f"Synthesis: {prediction.synthesis}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")

if __name__ == '__main__':
    main() 