# Repository Guidelines

## Project Structure & Module Organization

`telos_interp/` contains the Python package for trajectory loading, grid utilities, probe models, shared training code, and the Tyro-based CLI. CLI commands live under `telos_interp/commands/<command>/` with `*_fn.py` implementations, optional `*_utils.py`, and command-specific README files. Tests are in `tests/` and follow `test_*.py`. Experiment configuration templates are in `configs/`, standalone analysis scripts are at the repository root, and generated or published artifacts live in `results/`, `reports/`, `flag_swap/`, and `telos_interp/trace_viewer/`.

## Build, Test, and Development Commands

Use Python 3.10+ and `uv` for dependency management.

```bash
uv sync                 # create/update .venv from pyproject.toml and uv.lock
make install            # install runtime dependencies and editable package
make install-dev        # install all extras, pre-commit hooks, editable package
make test               # run pytest with project config and verbose output
make check-style        # check Ruff formatting and linting without fixes
make fix-style          # apply Ruff formatting and auto-fixes
uv run interp-cli --help # inspect available CLI commands
```

For focused testing, run `uv run pytest tests/test_grid_utils.py -vv`. To skip expensive cases, use `uv run pytest -m "not slow and not require_cuda_gpu"`.

## Coding Style & Naming Conventions

Ruff is the formatter and linter; configuration is in `pyproject.toml`. Keep Python code compatible with Python 3.10+, use a 119-character line length, sorted imports, and Google-style docstrings for public functions. Prefer existing package helpers over duplicating parsing, loading, or training logic. Name tests `test_<behavior>.py`; name command implementation files `<command>_fn.py`; keep CLI subcommand names in snake_case.

## Testing Guidelines

Pytest discovers tests from `tests/` using `test_*.py`. Add or update focused tests when changing shared utilities, activation loading, probe models, training behavior, or command contracts. Mark long-running tests with `slow` and GPU-dependent tests with `require_cuda_gpu`. Run `make test` before opening a PR, or document any skipped tests and why.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, often with a scope prefix such as `flag-swap:`. Use concise messages like `flag-swap: add MLX intervention runner` or `fix probe normalization`. PRs should describe the motivation, summarize behavioral changes, link related issues or experiments, list validation commands, and include screenshots or report snippets when visualizations or generated outputs change.

## Agent-Specific Instructions

Do not overwrite generated datasets, probe checkpoints, or result files unless the task explicitly requires it. Treat `configs/*.conf` as shell-style experiment templates, not TOML. When changing CLI behavior, update the relevant command README and verify `uv run interp-cli <command> --help`.
