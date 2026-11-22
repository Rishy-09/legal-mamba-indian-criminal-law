# Contributing

Contributions to Legal-Mamba are welcome, particularly improvements to correctness, reproducibility, evaluation, documentation, and experimentation.

## Development Setup

```bash
git clone https://github.com/Rishy-09/legal-mamba-indian-criminal-law.git
cd legal-mamba-indian-criminal-law

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
```

## Before Submitting Changes

Compile the source:

```bash
python -m compileall -q src
```

Run the automated tests:

```bash
python -m unittest discover -s tests -v
```

All existing tests should pass.

## Model and Training Changes

Changes to model architecture, tokenizer configuration, vocabulary size, context length, or training configuration may make existing checkpoints incompatible.

When changing these components:

- document the new configuration
- add or update relevant tests
- report new training/evaluation results where applicable
- clearly state checkpoint compatibility
- update the README when documented behaviour changes

## Data and Credentials

Do not commit:

- API keys or credentials
- virtual environments
- Python caches
- temporary generated files
- large trained checkpoints

The corpus-generation utility expects `GOOGLE_API_KEY` through the environment.

## Pull Requests

Keep changes focused and include:

- a concise summary
- motivation
- affected components
- validation performed
- model/checkpoint impact, if any
- updated results when training behaviour changes

## Research Scope

Legal-Mamba is an educational research project.

Generated model output must not be presented as authoritative legal information or professional legal advice.