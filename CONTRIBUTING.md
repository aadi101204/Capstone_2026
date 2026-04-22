# Contributing to Capstone_2026

Thank you for your interest in contributing. This is a research capstone project;
contributions that improve reproducibility, code clarity, or extend the encryption
analysis pipeline are welcome.

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. Create a **feature branch** off `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. Install dependencies and set up a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # Windows
   pip install -r requirements.txt
   ```

## Making Changes

- Keep changes focused — one logical change per PR.
- Follow the existing module layout:
  - `src/` — core model and data components (`models.py`, `dataset.py`, `chaos.py`, `utils.py`)
  - Root — pipeline scripts (`main.py`, `encryption.py`, `decryption.py`, `inference.py`, `correlation_analysis.py`, `outputParameters.py`)
  - `docs/` — architecture and change documentation
- Add or update docstrings for any modified public function.
- Do **not** commit:
  - Model weights (`*.pth`, `*.pt`, `*.ckpt`)
  - Dataset images (`Dataset/`, `KIIT-MiTA/`, `TrainSet/`)
  - Generated output (`FinalResults/`, `sequence_file.csv`)

## Submitting a Pull Request

1. Push your branch to your fork.
2. Open a PR against `main` with a clear title and description of the change.
3. Reference any relevant issue numbers.
4. PRs will be reviewed before merging.

## Reporting Issues

Open a GitHub Issue describing:
- What you expected to happen
- What actually happened
- Steps to reproduce (including OS, Python version, and PyTorch version)

## Code Style

- Standard Python (PEP 8). No linter is enforced, but keep line lengths reasonable (~100 chars).
- Prefer explicit over implicit — document non-obvious numerical choices (e.g. SIREN factor of 30).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
