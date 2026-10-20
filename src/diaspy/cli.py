"""Safe package entry point; model execution requires explicit confirmation."""
import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("experiment", help="Stage-zero experiment (use --dry-run)", add_help=False)
    legacy = commands.add_parser("legacy", help="Original interactive xAI demo (paid)")
    legacy.add_argument("--confirm", action="store_true", help="Allow paid model calls")
    if argv and argv[0] == "experiment":
        # LiteLLM can fetch pricing at import time; planning never needs it.
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        from experiments.theory_driven.run_staged import main as experiment_main
        return experiment_main(argv[1:])
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if not args.confirm:
        legacy.error("Model execution requires --confirm")
    if not os.environ.get("XAI_API_KEY"):
        legacy.error("XAI_API_KEY environment variable is not set")
    _legacy_main()
    return 0


def _legacy_main():
    import dspy
    from .responders import DialecticResponder
    from .utils import compile_agents, trainset

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
    raise SystemExit(main())