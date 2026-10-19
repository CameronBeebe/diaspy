"""Theory-driven experiment workspace for testing critique-first optimizers.

This experiment tests whether theory-driven optimizers beat GEPA's default 
reflection at LOW optimization budgets (50/100/200 total calls).

Components:
- arms: Different optimizer configurations (GEPA, Popperian, Bayesian/Popperian)
- benchmarks: HoVer, PUPA, IFBench task implementations
- analysis: Statistical analysis and plotting tools
- config: Experiment configuration management
- results: Structured results storage
"""

__version__ = "0.1.0"
