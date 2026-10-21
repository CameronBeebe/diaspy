# Diaspy

Diaspy is an **experimental** DSPy package containing dialectical responders and
an offline-verifiable stage-zero GEPA comparison. It is not a validated claim
that debate, typed criticism, or philosophical prompts improve model quality.

## Install

Use **CPython 3.12** (currently the only supported/tested minor version). DSPy
**3.4.1** and GEPA **0.1.4** are pinned; later DSPy APIs are not validated.
From this repository:

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

Alternatively, `python3.12 -m venv .venv`, activate it, and run
`python -m pip install -e '.[dev]'`. The base installation (`pip install .`)
includes the experiment runtime and dataset dependencies. `[dev]` adds the
socket-blocked tests, build tools, linting and type checking required by
preflight. `[notebooks]` adds optional notebook/plotting tools. Dependency
requirements live in `pyproject.toml`, not a second experiment environment.

## Safe first commands

No API key is needed. These do not construct model clients, download datasets,
or run optimization:

```sh
diaspy --help
diaspy experiment --help
diaspy experiment --stage 0 --dry-run
```

Running `diaspy` alone prints help. The dry-run is a **plan**, not an execution or
accuracy result. It reports the existing four variants (`A`, `B@half_no_regress`,
`M@half_no_regress`, `B@majority`), shared adaptive call ranges, canary calls,
held-out evaluation and a planning estimate. The estimate is not a provider
quote or hard billing limit. `python -m diaspy.cli` is equivalent to `diaspy`.
The previous experiment entry point remains available:

```sh
python -m experiments.theory_driven.run_staged --stage 0 --dry-run
```

## Offline verification

From the checkout after installing `[dev]`:

```sh
export LITELLM_LOCAL_MODEL_COST_MAP=True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -m pytest tests experiments/theory_driven/tests --disable-socket -q
python -m experiments.theory_driven.preflight --stage 0
```

Preflight runs Ruff, Pyright and the shipped experiment tests with sockets
blocked, then prints the plan. It strips provider API keys/tokens from its
child processes. Experiment tests use DSPy's `DummyLM` and attributed HotpotQA
fixtures, never live model calls. Existing dependency deprecation warnings are
not hidden. Passing offline tests establishes software behavior, not provider
availability, model quality, or scientific validity.

## Build and use away from the checkout

```sh
python -m build --installer uv
# Use an absolute path to the generated wheel when installing elsewhere:
uv venv --python 3.12 /path/to/clean-env
uv pip install --python /path/to/clean-env/bin/python \
  '/absolute/path/to/diaspy/dist/diaspy-0.1.0-py3-none-any.whl[dev]'
cd /path/to/another-directory
/path/to/clean-env/bin/diaspy --help
/path/to/clean-env/bin/diaspy experiment --stage 0 --dry-run
/path/to/clean-env/bin/python -m experiments.theory_driven.preflight --stage 0
```

**One distribution ships both `diaspy` and `experiments.theory_driven`.** The
existing experiment import namespace remains canonical to preserve module,
class, monkeypatch and serialized-type identity; there is no duplicate runtime
under a new namespace. Its acceptance tests, fixture attribution and explicit
verification configuration are packaged too. Notebooks and generated results
are not shipped in the wheel. Installed preflight checks the installed files,
not a presumed repository or an unrelated working directory.

## Explicit live execution (not part of offline verification)

Only stage 0 is supported. Live runs use HotpotQA distractor validation data
with deterministic train/validation/test splits, local BM25 retrieval and
answer F1. The configured task and canary model is
`openrouter/qwen/qwen3.5-9b`; **it is paid**. With an `OPENROUTER_API_KEY` already
provided securely in your environment, the opt-in is:

```sh
diaspy experiment --stage 0 --confirm
# Resume only completed results with matching scientific configuration:
diaspy experiment --stage 0 --confirm --resume experiments/theory_driven/results
```

This first runs the offline preflight (requires `[dev]`), then downloads data
if needed and runs the canary before the arms. Outputs are relative to your
working directory by default. A failed arm returns a nonzero exit status.
The $1 stage guard depends on costs reported by the provider; missing costs
count as zero. GEPA may overshoot the optimization cap by one iteration;
held-out evaluation and retry attempts add calls. These are not guaranteed
billing bounds. Read [the experiment notes](experiments/theory_driven/README.md)
for budgets, routing, traces, resume semantics and methodological limits.

The original interactive responder is retained behind a separate paid opt-in:

```sh
diaspy legacy --help
diaspy legacy --confirm  # requires XAI_API_KEY; compiles agents before prompting
```

The Python modules `diaspy.agents`, `diaspy.responders`, `diaspy.signatures` and
`diaspy.utils` remain available. Their model execution requires an explicitly
configured DSPy LM. Legacy demos are not the stage-zero experiment, and pilot
stages/other benchmark adapters are not validated for real experiments.

## License

Copyright (c) 2023-2026 The SciPhi Initiative, LLC. Released under the
[MIT License](LICENSE). The shipped HotpotQA fixture is not covered by the MIT
License; it retains its separate source attribution and CC BY-SA 4.0 license
metadata.
