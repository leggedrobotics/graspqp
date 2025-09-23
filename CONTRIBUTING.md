# Contributing

Thanks for your interest in contributing to GraspQP! This guide will help you get set up.

## Development setup

1. Create and activate an environment
2. Install the packages in editable mode

```bash
pip install -e graspqp[full]
pip install -e graspqp_isaaclab/src
```

3. Install dev tools and pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

## Coding style

- Python formatting with Black and isort (run `make format`)
- Lint with flake8 (run `make lint`)
- Keep functions small and documented; prefer docstrings for public APIs

## Tests

- Run tests: `make test`
- Add minimal unit tests for new public behaviors

## Pull requests

- Describe the change and motivation
- Include before/after behavior if applicable
- Link related issues
