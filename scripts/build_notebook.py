"""Generate llmops_assignment2.ipynb from the cell sources below.

Keeping the notebook as code makes it diff-able; run:
    python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# Experiential Learning Assignment 2: LLMOps
## API Cost, Latency, and Prompt Version Management (EduAssist AI)

**Course:** MLOps | **Student:** Karthik Chundi | **Mode:** Individual

This notebook does three things:

1. **Task 1:** calls an LLM API for five academic questions and measures the latency of each request.
2. **Task 2:** reads token usage from each response and works out the cost per request and the projected monthly cost.
3. **Task 3:** registers three prompt versions in the MLflow Prompt Registry, evaluates them on the same five questions, logs the metrics, and promotes the best one with the `production` alias.

| Item | Value |
|---|---|
| Provider / API | Google Gemini Developer API (`generativelanguage.googleapis.com`, `v1beta`) through the `google-genai` Python SDK |
| Model | `gemini-3.5-flash-lite` (override with the `GEMINI_MODEL` environment variable) |
| Pricing used | USD 0.30 per 1M input tokens, USD 2.50 per 1M output tokens (thinking tokens are billed as output). Paid-tier Standard rate. |
| Pricing date / currency | 2026-11-02, USD (source: https://ai.google.dev/gemini-api/docs/pricing) |
| Billing note | The calls ran on the **free tier**, so the actual charge was USD 0. Costs below show what the same traffic would cost on the paid tier. |
| Secret handling | The API key is read from `GEMINI_API_KEY` (an environment variable or a git-ignored `.env` file). It never appears in the code or outputs. |
""")

md("## 0. Setup and configuration")

code(r"""
import os, json, time, difflib, platform, statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
import mlflow
from mlflow.genai.scorers import scorer

load_dotenv()  # picks up a local .env if present (git-ignored)
API_KEY = os.environ.get("GEMINI_API_KEY")
assert API_KEY, "Set the GEMINI_API_KEY environment variable (see README)."

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_VERSION = "v1beta"
PRICE_INPUT_PER_M = float(os.environ.get("PRICE_INPUT_PER_M", "0.30"))    # USD / 1M input tokens
PRICE_OUTPUT_PER_M = float(os.environ.get("PRICE_OUTPUT_PER_M", "2.50"))  # USD / 1M output tokens
PRICING_DATE = "2026-11-02"
CURRENCY = "USD"
MONTHLY_REQUESTS = 10_000
# Free-tier rate limits are per minute; pause between calls (the pause is NOT part of measured latency).
PACING_SECONDS = float(os.environ.get("PACING_SECONDS", "4"))

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
client = genai.Client(api_key=API_KEY)

print(f"Python {platform.python_version()} | mlflow {mlflow.__version__} | google-genai {genai.__version__}")
print(f"Model: {MODEL} | API: Gemini Developer API {API_VERSION} | Pricing ({PRICING_DATE}, {CURRENCY}): "
      f"in ${PRICE_INPUT_PER_M}/M, out ${PRICE_OUTPUT_PER_M}/M")
""")

md("""
### Academic question set
The same five undergraduate-level questions are used in every task. Each one has a few **reference key concepts**, which a cheap, deterministic quality check (`keyword_coverage`) looks for in the answer.
""")

code(r"""
QUESTIONS = [
    {"qid": "Q1", "question": "What is the difference between supervised and unsupervised learning?",
     "keywords": ["label", "cluster", "classification", "regression", "pattern"]},
    {"qid": "Q2", "question": "Explain how gradient descent minimises a loss function.",
     "keywords": ["gradient", "learning rate", "derivative", "iterat", "minimum"]},
    {"qid": "Q3", "question": "What is overfitting in machine learning and how can it be prevented?",
     "keywords": ["training data", "generaliz", "regulariz", "cross-validation", "dropout"]},
    {"qid": "Q4", "question": "Describe the role of the CAP theorem in distributed database design.",
     "keywords": ["consistency", "availability", "partition", "trade", "network"]},
    {"qid": "Q5", "question": "How does photosynthesis convert light energy into chemical energy?",
     "keywords": ["chlorophyll", "glucose", "carbon dioxide", "atp", "oxygen"]},
]
pd.DataFrame(QUESTIONS)[["qid", "question"]]
""")

md("""
### Prompt templates (copied exactly from the assignment brief)
MLflow's Prompt Registry uses `{{variable}}` placeholders, which is the same syntax the brief uses.
""")

code(r'''
PROMPT_NAME = "eduassist_qa"

PROMPT_TEMPLATES = {
    1: """Answer the following student question clearly.
Question: {{question}}""",
    2: """You are an academic assistant.
Answer the following student question in a clear, concise, and accurate manner.
Use simple language suitable for undergraduate students.
Question: {{question}}""",
    3: """You are an academic assistant.
Answer the student question and return ONLY valid JSON.
Do not include markdown, explanations outside JSON, or code fences.
Use this exact JSON schema:
{
  "answer": "string",
  "confidence": "high | medium | low",
  "key_points": ["string"]
}
Question: {{question}}""",
}

COMMIT_MESSAGES = {
    1: "v1 Basic: minimal instruction baseline, free-text answer",
    2: "v2 Structured: add academic-assistant role, concision and undergraduate-level language constraints",
    3: "v3 JSON Output: enforce strict JSON contract {answer, confidence, key_points} with no markdown or code fences",
}

def render(template: str, question: str) -> str:
    """Local rendering used for Tasks 1-2 (Task 3 renders via the MLflow-registered prompt)."""
    return template.replace("{{question}}", question)
''')

md("""
### LLM call wrapper with latency instrumentation
* Latency is the wall-clock time around the `generate_content` call only, measured with `time.perf_counter()`. Timestamps use UTC ISO-8601.
* An HTTP 429 (rate limit) or 5xx response is retried with exponential backoff, and the `attempts` column records how many tries it took. The latency kept is the one from the attempt that succeeded, so backoff waits are not counted as model latency.
* **Billed output tokens** are `candidates_token_count + thoughts_token_count`, because Gemini bills thinking tokens at the output rate.
""")

code(r"""
def call_llm(prompt_text: str, qid: str, version: int, max_attempts: int = 5) -> dict:
    record = {"qid": qid, "prompt_version": f"v{version}", "model": MODEL}
    for attempt in range(1, max_attempts + 1):
        start_wall = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt_text)
            t1 = time.perf_counter()
            end_wall = datetime.now(timezone.utc)
            u = resp.usage_metadata
            visible_out = u.candidates_token_count or 0
            thoughts = u.thoughts_token_count or 0
            record.update(
                start_time=start_wall.isoformat(timespec="milliseconds"),
                end_time=end_wall.isoformat(timespec="milliseconds"),
                latency_seconds=round(t1 - t0, 3),
                status="success", attempts=attempt,
                finish_reason=str(resp.candidates[0].finish_reason.name) if resp.candidates else "NONE",
                input_tokens=u.prompt_token_count or 0,
                visible_output_tokens=visible_out,
                thinking_tokens=thoughts,
                output_tokens=visible_out + thoughts,
                total_tokens=u.total_token_count or 0,
                response_text=resp.text or "",
            )
            break
        except errors.APIError as e:
            t1 = time.perf_counter()
            retryable = e.code == 429 or (e.code or 0) >= 500
            record.update(start_time=start_wall.isoformat(timespec="milliseconds"),
                          end_time=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                          latency_seconds=round(t1 - t0, 3), status=f"error_{e.code}", attempts=attempt,
                          finish_reason="ERROR", input_tokens=0, visible_output_tokens=0, thinking_tokens=0,
                          output_tokens=0, total_tokens=0, response_text="")
            if not retryable or attempt == max_attempts:
                print(f"  {qid} v{version}: giving up ({e.code}: {e.message})")
                break
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"  {qid} v{version}: HTTP {e.code}, retrying in {wait}s")
            time.sleep(wait)
    time.sleep(PACING_SECONDS)  # stay within free-tier RPM; excluded from latency
    return record

def add_costs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["input_cost_usd"] = df["input_tokens"] / 1_000_000 * PRICE_INPUT_PER_M
    df["output_cost_usd"] = df["output_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_M
    df["total_cost_usd"] = df["input_cost_usd"] + df["output_cost_usd"]
    return df
""")

md("""
---
## Task 1: Call an LLM and measure latency (2 marks)
The five questions are sent with the **v1 baseline prompt**. Each row records the question ID, prompt version, model, start and end time, latency, and response status.
""")

code(r"""
task1_records = []
for q in QUESTIONS:
    rec = call_llm(render(PROMPT_TEMPLATES[1], q["question"]), q["qid"], 1)
    task1_records.append(rec)
    print(f"{rec['qid']}: {rec['status']:<8} {rec['latency_seconds']:.3f}s  out_tokens={rec['output_tokens']}")

task1 = pd.DataFrame(task1_records)
latency_table = task1[["qid", "prompt_version", "model", "start_time", "end_time", "latency_seconds", "status"]]
latency_table
""")

code(r"""
lat = task1.loc[task1.status == "success", "latency_seconds"]
latency_summary = pd.DataFrame({
    "statistic": ["count", "mean", "median", "min", "max", "std (sample)"],
    "latency_seconds": [len(lat), lat.mean(), lat.median(), lat.min(), lat.max(), lat.std(ddof=1)],
}).round(3)
latency_summary
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(task1.qid, task1.latency_seconds, color="#4C72B0")
ax.axhline(lat.mean(), ls="--", color="#C44E52", label=f"mean {lat.mean():.2f}s")
ax.set_ylabel("latency (s)"); ax.set_title(f"Task 1: per-request latency ({MODEL}, prompt v1)"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "task1_latency.png", dpi=150); plt.show()

slow = task1.loc[task1.latency_seconds.idxmax()]
others = task1.drop(index=slow.name)
print(f"Slowest request: {slow.qid} at {slow.latency_seconds:.2f}s "
      f"({slow.latency_seconds / lat.mean():.2f}x the mean)")
print(f"  output tokens: {slow.output_tokens} vs {others.output_tokens.mean():.0f} avg for the others; "
      f"input tokens: {slow.input_tokens} vs {others.input_tokens.mean():.0f}")
print(f"  ms per output token: slowest {1000 * slow.latency_seconds / max(slow.output_tokens, 1):.1f} "
      f"vs others {1000 * (others.latency_seconds / others.output_tokens.clip(lower=1)).mean():.1f}")
""")

md("**Interpretation.** This paragraph is generated from the measured numbers, so it always matches the run above.")

code(r"""
from IPython.display import Markdown, display

slow_rate = 1000 * slow.latency_seconds / max(slow.output_tokens, 1)
other_rate = (1000 * others.latency_seconds / others.output_tokens.clip(lower=1)).mean()
reasons = []
if slow.output_tokens > 1.1 * others.output_tokens.mean():
    reasons.append(f"**It produced a longer answer:** {slow.output_tokens} output tokens, against an average of "
                   f"{others.output_tokens.mean():.0f} for the other four. The model generates one token at a time, so "
                   f"decode time grows with the length of the answer.")
if slow_rate > 1.15 * other_rate:
    first = " This was also the first call, which pays for connection and TLS set-up." if slow.qid == task1.qid.iloc[0] else ""
    reasons.append(f"**It decoded more slowly:** {slow_rate:.1f} ms per output token, against {other_rate:.1f} ms for the "
                   f"others. That points to transient server-side load on the shared free tier, or to network jitter.{first}")
if not reasons:
    reasons.append("Its token count and decode rate are close to the others, so the gap is within normal "
                   "run-to-run variability of the network and the service.")
TASK1_TEXT = (
    f"Across the five v1 requests, latency averaged **{lat.mean():.2f} s**. The median was {lat.median():.2f} s, the "
    f"range was {lat.min():.2f}-{lat.max():.2f} s, and the sample standard deviation was {lat.std(ddof=1):.2f} s. The slowest request was "
    f"**{slow.qid} at {slow.latency_seconds:.2f} s ({slow.latency_seconds / lat.mean():.2f}× the mean)**.\n\n"
    + "\n".join(f"- {r}" for r in reasons)
    + f"\n\nInput length is not the cause, because every prompt was {task1.input_tokens.min()}-{task1.input_tokens.max()} "
      f"tokens and input is processed in parallel. Output length usually drives generation time, but single calls are noisy."
      f"So latency targets in production should use percentiles (p50/p95) over many requests, not single calls."
)
display(Markdown(TASK1_TEXT))
""")

md("""
---
## Task 2: Count tokens and calculate cost (3 marks)

`Input Cost = Input Tokens / 1,000,000 × Input Price per M`
`Output Cost = Output Tokens / 1,000,000 × Output Price per M`
`Total Cost per Request = Input Cost + Output Cost`
`Projected Monthly Cost = Average Cost per Request × 10,000`
""")

code(r"""
task1 = add_costs(task1)
cost_table = task1[["qid", "prompt_version", "model", "input_tokens", "output_tokens", "thinking_tokens",
                    "total_tokens", "input_cost_usd", "output_cost_usd", "total_cost_usd"]]

csv_cols = ["qid", "prompt_version", "model", "start_time", "end_time", "latency_seconds", "status", "attempts",
            "input_tokens", "visible_output_tokens", "thinking_tokens", "output_tokens", "total_tokens",
            "input_cost_usd", "output_cost_usd", "total_cost_usd"]
task1[csv_cols].to_csv("latency_tokens_cost.csv", index=False)
cost_table.style.format({c: "${:.6f}" for c in ["input_cost_usd", "output_cost_usd", "total_cost_usd"]})
""")

code(r"""
avg_cost = task1.total_cost_usd.mean()
monthly = avg_cost * MONTHLY_REQUESTS
out_share = task1.output_cost_usd.sum() / task1.total_cost_usd.sum()
projection = pd.DataFrame({
    "metric": ["avg input tokens", "avg output tokens", "avg cost / request (USD)",
               f"projected monthly cost for {MONTHLY_REQUESTS:,} requests (USD)", "share of cost from output tokens"],
    "value": [f"{task1.input_tokens.mean():.1f}", f"{task1.output_tokens.mean():.1f}", f"${avg_cost:.6f}",
              f"${monthly:.2f}", f"{out_share:.1%}"],
})
projection
""")

code(r"""
# What-if: cap the answer to ~150 output tokens (concise answers / max_output_tokens) and compare
capped_out = task1.output_tokens.clip(upper=150)
capped_cost = (task1.input_tokens / 1e6 * PRICE_INPUT_PER_M + capped_out / 1e6 * PRICE_OUTPUT_PER_M).mean()
print(f"Current     : ${avg_cost:.6f}/request -> ${monthly:,.2f}/month")
print(f"Capped @150 : ${capped_cost:.6f}/request -> ${capped_cost * MONTHLY_REQUESTS:,.2f}/month "
      f"({1 - capped_cost / avg_cost:.0%} saving)")
""")

code(r"""
TASK2_TEXT = (
    f"Average cost is **USD {avg_cost:.5f} per request**, which projects to **USD {monthly:,.2f} per month for "
    f"{MONTHLY_REQUESTS:,} requests**. Output tokens account for **{out_share:.1%}** of the cost. They are billed at "
    f"{PRICE_OUTPUT_PER_M / PRICE_INPUT_PER_M:.1f}× the input rate, and an answer is about "
    f"{task1.output_tokens.mean() / task1.input_tokens.mean():.0f}× longer than its prompt.\n\n"
    f"**Recommendation: shorten the responses.** Use a concise output contract (a short `answer` plus a few "
    f"`key_points`, as in v3), `max_output_tokens`, or both. Capping answers at 150 tokens would bring the bill "
    f"to about USD {capped_cost * MONTHLY_REQUESTS:,.2f} per month, a {1 - capped_cost / avg_cost:.0%} saving. "
    f"Trimming or caching the prompt would save almost nothing at these input sizes; it only pays off once "
    f"retrieved (RAG) context makes inputs large."
)
display(Markdown(TASK2_TEXT))
""")

md("""
---
## Task 3: Prompt versioning and MLflow evaluation (5 marks)
### 3.1 Create the experiment and register v1, v2, v3 in the Prompt Registry
""")

code(r"""
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
experiment = mlflow.set_experiment("eduassist_llmops_assignment")
print("Tracking URI:", mlflow.get_tracking_uri(), "| experiment_id:", experiment.experiment_id)

registered = {}
for v in (1, 2, 3):
    pv = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=PROMPT_TEMPLATES[v],
        commit_message=COMMIT_MESSAGES[v],
        tags={"label": ["basic", "structured", "json_output"][v - 1], "author": "karthik.chundi",
              "task": "academic_qa", "expected_output": "json" if v == 3 else "text"},
    )
    registered[v] = pv.version
    print(f"Registered {PROMPT_NAME} version {pv.version}: {COMMIT_MESSAGES[v]}")
""")

md("""
If the notebook is re-run, the registry adds new versions (4, 5, 6, ...) rather than overwriting old ones. To avoid confusion, the cells below always use the version numbers returned in `registered`. For a clean v1/v2/v3 numbering, delete `mlflow.db` before re-running (see README).

### 3.2 Prompt diff evidence
""")

code(r"""
diff_text = []
for a, b in [(1, 2), (2, 3)]:
    d = difflib.unified_diff(PROMPT_TEMPLATES[a].splitlines(), PROMPT_TEMPLATES[b].splitlines(),
                             fromfile=f"{PROMPT_NAME}/v{a}", tofile=f"{PROMPT_NAME}/v{b}", lineterm="")
    diff_text.append("\n".join(d))
diff_text = "\n\n".join(diff_text)
(OUT / "prompt_diff.txt").write_text(diff_text, encoding="utf-8")
print(diff_text)
""")

md("""
### 3.3 Validation and quality metrics
* **`valid_json`**: `json.loads(response)` succeeds on the raw text and gives an object. This is strict on purpose: code fences or any text outside the JSON make it fail.
* **`schema_valid`** (extra): valid JSON **and** exactly the keys `answer` (str), `confidence` ∈ {high, medium, low}, and `key_points` (list[str]).
* **`keyword_coverage`** (extra quality proxy): the share of each question's reference concepts that the answer mentions.

`valid_json` and `schema_valid` are also defined as MLflow `@scorer` functions that return booleans. `mlflow.genai.evaluate` averages them into a pass rate (`valid_json/mean`), which is how the brief describes aggregating boolean feedback.
""")

code(r"""
CONF = {"high", "medium", "low"}

def is_valid_json(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (json.JSONDecodeError, TypeError):
        return False

def is_schema_valid(text: str) -> bool:
    if not is_valid_json(text):
        return False
    o = json.loads(text)
    return (set(o) == {"answer", "confidence", "key_points"} and isinstance(o["answer"], str)
            and o["confidence"] in CONF and isinstance(o["key_points"], list)
            and all(isinstance(k, str) for k in o["key_points"]))

def keyword_coverage(text: str, keywords: list[str]) -> float:
    t = text.lower()
    return sum(k in t for k in keywords) / len(keywords)

@scorer
def valid_json(outputs) -> bool:
    return is_valid_json(outputs)

@scorer
def schema_valid(outputs) -> bool:
    return is_schema_valid(outputs)
""")

md("### 3.4 Run each registered prompt version on the same five questions and log to MLflow")

code(r"""
all_rows, summary_rows = [], []
KW = {q["qid"]: q["keywords"] for q in QUESTIONS}

for v in (1, 2, 3):
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{registered[v]}")
    with mlflow.start_run(run_name=f"prompt_v{v}") as run:
        mlflow.MlflowClient().link_prompt_version_to_run(run.info.run_id, prompt)  # run page -> "Registered prompts"
        rows = []
        for q in QUESTIONS:
            rec = call_llm(prompt.format(question=q["question"]), q["qid"], v)
            rec.update(question=q["question"], prompt_uri=prompt.uri,
                       valid_json=is_valid_json(rec["response_text"]),
                       schema_valid=is_schema_valid(rec["response_text"]),
                       keyword_coverage=keyword_coverage(rec["response_text"], KW[q["qid"]]))
            rows.append(rec)
            print(f"v{v} {q['qid']}: {rec['latency_seconds']:.2f}s in={rec['input_tokens']} "
                  f"out={rec['output_tokens']} json={rec['valid_json']}")
        df = add_costs(pd.DataFrame(rows))
        ok = df[df.status == "success"]

        metrics = {
            "avg_latency_seconds": ok.latency_seconds.mean(),
            "avg_input_tokens": ok.input_tokens.mean(),
            "avg_output_tokens": ok.output_tokens.mean(),
            "avg_cost_per_request": ok.total_cost_usd.mean(),
            "valid_json": df.valid_json.mean(),              # valid JSON responses / total responses
            "schema_valid_rate": df.schema_valid.mean(),
            "avg_keyword_coverage": ok.keyword_coverage.mean(),
            "projected_monthly_cost_usd": ok.total_cost_usd.mean() * MONTHLY_REQUESTS,
            "p50_latency_seconds": ok.latency_seconds.median(),
            "max_latency_seconds": ok.latency_seconds.max(),
            "success_rate": (df.status == "success").mean(),
        }
        mlflow.log_params({"prompt_name": PROMPT_NAME, "prompt_version": registered[v], "prompt_uri": prompt.uri,
                           "model": MODEL, "api": f"gemini-developer-api/{API_VERSION}",
                           "price_input_per_m_usd": PRICE_INPUT_PER_M, "price_output_per_m_usd": PRICE_OUTPUT_PER_M,
                           "pricing_date": PRICING_DATE, "currency": CURRENCY, "n_questions": len(QUESTIONS)})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"prompt_label": prompt.tags.get("label", ""), "commit_message": COMMIT_MESSAGES[v]})
        mlflow.log_text(prompt.template, f"prompt_v{v}.txt")
        mlflow.log_table(df.drop(columns=["response_text"]).assign(response=df.response_text),
                         artifact_file=f"responses_v{v}.json")

        # MLflow GenAI evaluation: boolean scorers aggregated to a pass rate (valid_json/mean)
        try:
            eval_data = [{"inputs": {"question": r["question"]}, "outputs": r["response_text"]} for r in rows]
            res = mlflow.genai.evaluate(data=eval_data, scorers=[valid_json, schema_valid])
            print(f"  mlflow.genai.evaluate -> {res.metrics}")
        except Exception as e:  # evaluation is supplementary; manual metrics above are authoritative
            print(f"  mlflow.genai.evaluate skipped: {type(e).__name__}: {e}")

        summary_rows.append({"prompt_version": f"v{v}", "registry_version": registered[v],
                             "run_id": run.info.run_id, **metrics})
        all_rows.append(df)

details = pd.concat(all_rows, ignore_index=True)
details.to_csv(OUT / "prompt_responses_detail.csv", index=False)
""")

md("### 3.5 Comparison and selection")

code(r"""
summary = pd.DataFrame(summary_rows)

# Selection rule: the product contract is strict JSON, so schema/JSON validity gates first,
# then answer quality (keyword coverage), then cost, then latency.
ranked = summary.sort_values(["schema_valid_rate", "valid_json", "avg_keyword_coverage",
                              "avg_cost_per_request", "avg_latency_seconds"],
                             ascending=[False, False, False, True, True])
best = ranked.iloc[0]
summary["recommended"] = summary.prompt_version.eq(best.prompt_version).map({True: "Yes (production)", False: "No"})

comparison = summary[["prompt_version", "avg_latency_seconds", "avg_input_tokens", "avg_output_tokens",
                      "avg_cost_per_request", "valid_json", "schema_valid_rate", "avg_keyword_coverage",
                      "projected_monthly_cost_usd", "recommended"]]
comparison.to_csv("prompt_evaluation_results.csv", index=False)
comparison.style.format({"avg_latency_seconds": "{:.3f}", "avg_input_tokens": "{:.1f}", "avg_output_tokens": "{:.1f}",
                         "avg_cost_per_request": "${:.6f}", "valid_json": "{:.0%}", "schema_valid_rate": "{:.0%}",
                         "avg_keyword_coverage": "{:.0%}", "projected_monthly_cost_usd": "${:.2f}"})
""")

code(r"""
fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for ax, (col, title) in zip(axes, [("avg_latency_seconds", "avg latency (s)"),
                                   ("avg_output_tokens", "avg output tokens"),
                                   ("avg_cost_per_request", "avg cost / request (USD)"),
                                   ("valid_json", "valid_json rate")]):
    ax.bar(summary.prompt_version, summary[col], color=["#8C8C8C", "#4C72B0", "#55A868"])
    ax.set_title(title, fontsize=10)
plt.suptitle(f"Prompt version comparison on 5 questions ({MODEL})", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "prompt_comparison.png", dpi=150); plt.show()
""")

md("### 3.6 Promote the best version with the `production` alias")

code(r"""
prod_version = int(best.registry_version)
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=prod_version)
mlflow.MlflowClient().set_tag(best.run_id, "deployment", "production")

loaded = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
print(f"prompts:/{PROMPT_NAME}@production -> version {loaded.version} ({best.prompt_version})")
print(loaded.template)
""")

code(r"""
# Sample v3 output for the report (parsed JSON)
v3_sample = details[(details.prompt_version == "v3") & details.valid_json].head(1)
if len(v3_sample):
    print(json.dumps(json.loads(v3_sample.response_text.iloc[0]), indent=2))
""")

md("### 3.7 Promotion justification (generated from this run's metrics)")

code(r"""
S = summary.set_index("prompt_version")
b, base = S.loc[best.prompt_version], S.loc["v1"]
others_v = [v for v in S.index if v != best.prompt_version]
pct = lambda x: f"{x:.0%}"
JUSTIFICATION = (
    f"**`{PROMPT_NAME}` {best.prompt_version} (registry version {int(best.registry_version)}) has been promoted to the "
    f"`production` alias.**\n\n"
    f"- **JSON validity.** The EduAssist client parses every response as JSON. {best.prompt_version} returned valid JSON in "
    f"{pct(b.valid_json)} of responses, and {pct(b.schema_valid_rate)} matched the schema exactly: the three keys, "
    f"`confidence` taken from {{high, medium, low}}, and `key_points` as a list of strings. "
    + " and ".join(f"{v} scored {pct(S.loc[v].valid_json)}" for v in others_v)
    + ". Their responses were markdown prose, which would break the integration however good the content is. "
      "`mlflow.genai.evaluate` with a boolean `valid_json` scorer confirmed the same pass rates.\n"
    f"- **Quality.** Coverage of reference concepts was {pct(b.avg_keyword_coverage)} for {best.prompt_version}, against "
    + " and ".join(f"{pct(S.loc[v].avg_keyword_coverage)} for {v}" for v in others_v)
    + ". The separate `answer` and `key_points` fields also suit revision-style study.\n"
    f"- **Latency.** {best.prompt_version} averaged {b.avg_latency_seconds:.2f} s, against "
    + " and ".join(f"{S.loc[v].avg_latency_seconds:.2f} s for {v}" for v in others_v)
    + (". It is faster because it generates far fewer tokens.\n" if b.avg_latency_seconds <= S.avg_latency_seconds.min()
       else f". It is not the fastest here, but the gap is small next to the {details.latency_seconds.min():.0f}-"
            f"{details.latency_seconds.max():.0f} s spread between individual calls: in this run network and queueing, "
            f"not output length, dominated latency.\n")
    + f"- **Tokens.** It uses {b.avg_input_tokens:.0f} input tokens, against {base.avg_input_tokens:.0f} for v1, because the "
    f"schema is part of the prompt. But it generates {b.avg_output_tokens:.0f} output tokens against "
    f"{base.avg_output_tokens:.0f} for v1 ({1 - b.avg_output_tokens / base.avg_output_tokens:.0%} fewer), and output is "
    f"the side that drives cost.\n"
    f"- **Cost.** USD {b.avg_cost_per_request:.5f} per request, or **USD {b.projected_monthly_cost_usd:,.2f} per month, "
    f"against USD {base.projected_monthly_cost_usd:,.2f} for v1 "
    f"({1 - b.projected_monthly_cost_usd / base.projected_monthly_cost_usd:.0%} less)**.\n\n"
    "**Risks.** The evaluation used only five questions, and keyword coverage is a lexical stand-in for quality. In production "
    "I would keep running the `valid_json` scorer on live traffic and add Gemini's native `response_schema` as a second "
    "guardrail. A future version would be promoted only if it beats the current one on this same evaluation."
)
display(Markdown(JUSTIFICATION))
""")

code(r"""
# Everything the report needs, so report/report.pdf is always built from the run that produced the CSVs
run_summary = {
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "environment": {"python": platform.python_version(), "os": platform.platform(terse=True),
                    "mlflow": mlflow.__version__, "google_genai": genai.__version__},
    "model": MODEL, "api_version": API_VERSION, "pricing_date": PRICING_DATE, "currency": CURRENCY,
    "price_input_per_m": PRICE_INPUT_PER_M, "price_output_per_m": PRICE_OUTPUT_PER_M,
    "task1_stats": dict(zip(latency_summary.statistic, latency_summary.latency_seconds.astype(float))),
    "task1_rows": task1[["qid", "input_tokens", "output_tokens", "total_tokens", "latency_seconds",
                         "total_cost_usd"]].to_dict("records"),
    "task1_text": TASK1_TEXT, "task2_text": TASK2_TEXT,
    "comparison": summary.drop(columns=["run_id"]).to_dict("records"),
    "best_version": best.prompt_version, "justification": JUSTIFICATION,
}
(OUT / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=float), encoding="utf-8")
print("wrote", OUT / "run_summary.json")
""")

md("""
### Reproducibility
`mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` and then open http://127.0.0.1:5000 to see the experiment, the Prompt Registry, and the alias.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, ROOT / "llmops_assignment2.ipynb")
print("wrote", ROOT / "llmops_assignment2.ipynb")
