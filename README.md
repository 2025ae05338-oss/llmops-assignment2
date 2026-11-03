# LLMOps Assignment 2: API Cost, Latency, and Prompt Version Management

EduAssist AI academic Q&A assistant. The notebook measures LLM latency, works out token-based cost, and versions, evaluates, and promotes prompts with the MLflow Prompt Registry.

| | |
|---|---|
| Python | 3.12.3 |
| LLM | Google Gemini Developer API (`v1beta`), model `gemini-3.5-flash-lite`, called through the `google-genai` SDK |
| Pricing used | USD 0.30 per 1M input tokens / USD 2.50 per 1M output tokens (thinking tokens count as output). Paid-tier Standard rate, taken from https://ai.google.dev/gemini-api/docs/pricing on 2026-11-02. |
| Billing | The experiment ran on the Gemini **free tier**, so it cost USD 0. The reported costs are what the same traffic would cost at the paid-tier rates above. |
| Tracking | MLflow 3.17.2 with a SQLite backend (`mlflow.db`). The Prompt Registry needs a database-backed store. |

## Repository layout

```
llmops_assignment2.ipynb          main notebook (Tasks 1-3), executed with outputs
llmops_assignment2.html           HTML export of the executed notebook
latency_tokens_cost.csv           Task 1 + 2: per-request latency, tokens, cost (prompt v1)
prompt_evaluation_results.csv     Task 3: per-version comparison table + recommendation
outputs/                          prompt_responses_detail.csv (all 15 responses), prompt_diff.txt, charts
screenshots/                      MLflow UI: experiment, metrics, Prompt Registry, diff, production alias
report/report.pdf                 final report (500-700 words)
scripts/build_notebook.py         generates the notebook from source (diff-friendly)
scripts/capture_screenshots.py    captures the MLflow UI screenshots with Playwright
scripts/build_report.py           renders report/report.md to PDF
requirements.txt
```

## Required libraries

`mlflow`, `google-genai`, `pandas`, `matplotlib`, `python-dotenv`, `jupyter`/`nbconvert`/`ipykernel`. The exact versions are pinned in `requirements.txt`. `playwright` and `markdown` are optional; they are only used to capture screenshots and build the PDF.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | yes | Gemini API key from https://aistudio.google.com/apikey. **Never hard-code it or commit it.** |
| `GEMINI_MODEL` | no | Model override (default `gemini-3.5-flash-lite`) |
| `PRICE_INPUT_PER_M`, `PRICE_OUTPUT_PER_M` | no | Price overrides in USD per 1M tokens (defaults 0.30 / 2.50) |
| `PACING_SECONDS` | no | Pause between calls to stay inside free-tier rate limits (default 4). The pause is not counted in latency. |
| `MLFLOW_TRACKING_URI` | no | Defaults to `sqlite:///mlflow.db` |

You can put these in a `.env` file. Copy `.env.example` to `.env`; `.env` is git-ignored.

## Setup

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit .env and set GEMINI_API_KEY
```

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then set GEMINI_API_KEY
```

## Quick run (Linux / lab VM)

```bash
bash scripts/lab_run.sh all     # setup, run notebook, MLflow UI, screenshots, report PDF, submission zip
```
See `docs/LAB_STEPS.md` for the lab walkthrough.

## Reproduce the experiment

1. Optional: start from a clean registry so the prompt versions are numbered 1/2/3. Delete `mlflow.db` and `mlruns/`.
2. Run the notebook, either interactively or headless:
   ```bash
   jupyter notebook llmops_assignment2.ipynb          # interactive: Run All
   # or
   jupyter nbconvert --to notebook --execute --inplace llmops_assignment2.ipynb
   jupyter nbconvert --to html llmops_assignment2.ipynb
   ```
   The notebook makes 20 API calls in total: 5 for Task 1 and 15 for Task 3. It writes `latency_tokens_cost.csv`, `prompt_evaluation_results.csv` and `outputs/*`. It creates the experiment `eduassist_llmops_assignment`, registers `eduassist_qa` versions 1-3 with commit messages, logs one run per version, and sets the `production` alias on the winning version.
3. Open the MLflow UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
   ```
   Then go to http://127.0.0.1:5000. Use **Experiments → eduassist_llmops_assignment** for runs and metrics, and **Prompts → eduassist_qa** for versions, the *Compare* tab for the diff, and the `@production` alias.
4. Optional: recapture the screenshots and rebuild the report PDF:
   ```bash
   python -m playwright install chromium
   python scripts/capture_screenshots.py --url http://127.0.0.1:5000
   python scripts/build_report.py --where "BITS lab VM"
   ```
   The report is generated from `outputs/run_summary.json`, so its numbers always match the run that produced the CSVs.

## Using the production prompt

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
prompt = mlflow.genai.load_prompt("prompts:/eduassist_qa@production")
text = prompt.format(question="What is a hash table?")
```

## Security

No API keys, billing details, or personal data are stored in this repository or in the notebook outputs. The key is read only from the environment.
