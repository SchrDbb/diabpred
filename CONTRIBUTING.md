# Contributing to DiabPred

Thank you for your interest in contributing! This document outlines how to
get started.

## Code of Conduct

This project adheres to a [Contributor Covenant](https://www.contributor-covenant.org/)
code of conduct. Be respectful and constructive.

## Ways to Contribute

- **Bug reports** – Open a GitHub Issue with a minimal reproducer.
- **Feature requests** – Open a GitHub Issue describing the use-case.
- **Pull requests** – Fix a bug, add a new classifier, improve documentation.
- **New datasets** – Add support for alternative diabetes datasets.

## Development Setup

```bash
git clone https://github.com/yourusername/diabpred.git
cd diabpred
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -v
```

## Adding a New Classifier

1. Import it in `diabpred/models.py`
2. Add an entry to `MODEL_REGISTRY` with a unique name and default hyperparameters
3. Run `pytest tests/ -v` to confirm all tests pass
4. Add a sentence to `paper/paper.md` if the new model is noteworthy

## Pull Request Guidelines

- Keep PRs focused and small.
- Include tests for any new functionality.
- Ensure `pytest tests/ -v` passes locally before submitting.
- Update `CHANGELOG.md` with a summary of your change.
- Follow the existing code style (PEP 8, 100-char line limit).

## Running Tests

```bash
pytest tests/ -v                              # all tests
pytest tests/ -v --cov=diabpred               # with coverage
pytest tests/test_diabpred.py::TestPredict    # specific class
```

## Reporting Security Issues

Please do **not** open a public issue for security vulnerabilities. Email
`your@email.com` directly.
