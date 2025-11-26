# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `generate.py` (text sampling), `get_log_likelihood.py` (conditional likelihood), `chat.py` (CLI chat), `app.py` (Gradio demo), `eval_llada.py` (lm-eval adapter).
- Data and assets: `data/` (e.g., `poem_data.json`), `imgs/` (figures), `visualization/` (step-by-step sampling visualizers).
- Benchmarks: `eval_llada_lm_eval.sh` (lm-eval), `eval_llada_opencompass.sh`, `opencompass/` (vendored configs and runner).

## Build, Test, and Development Commands
- Create env (recommended): `python -m venv .venv && source .venv/bin/activate`.
- Install (minimal): `pip install transformers==4.38.2 gradio torch` (see scripts for extras).
- Run chat: `python chat.py`.
- Run demo UI: `python app.py`.
- Sample generation (batch supported): `python generate.py`.
- Log-likelihood example: `python get_log_likelihood.py`.
- lm-eval: `bash ./eval_llada_lm_eval.sh` (sets `HF_ALLOW_CODE_EVAL=1`, `HF_DATASETS_TRUST_REMOTE_CODE=true`).
- OpenCompass: `bash ./eval_llada_opencompass.sh` (installs `opencompass/` in editable mode).

## Coding Style & Naming Conventions
- Python 3.8+; follow PEP 8, 4-space indent; UTF-8; ASCII in source unless needed.
- Names: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants (e.g., `MASK_ID = 126336`).
- Docstrings for public functions; short inline comments for non-obvious logic (e.g., remasking, CFG).
- File names: lowercase with underscores; scripts runnable via `python <file>.py`.

## Testing Guidelines
- Smoke tests: run `python chat.py` and `python generate.py` to verify inference; for ppl/gen tasks, run a single lm-eval/OpenCompass job before PR (see the two `*.sh` scripts).
- If adding unit tests, prefer `pytest` with `tests/test_*.py`; keep tests GPU-conditional when possible and skip network by default.
- For changes to `generate.py` or `get_log_likelihood.py`, include before/after metrics on a small subset (e.g., GSM8K 20 samples).

## Commit & Pull Request Guidelines
- Commits: imperative mood and scoped when helpful, e.g., `Fix: clamp EOS logits in generate.py`, `Add: OpenCompass GSM8K config` (repo history uses Add/Update/Delete/Fix).
- PRs must include: brief summary, motivation, changed files list, how to run (commands), and sample outputs/metrics. Link related issues.
- For API/behavioral changes, update `README.md`, `EVAL.md`, or scripts accordingly and attach logs or screenshots (Gradio/CLI).

## Security & Configuration Tips
- GPU strongly recommended; set left-padding if needed (see note in `generate.py`).
- For HumanEval/MBPP, the scripts enable code execution—run in isolated env and review outputs.
