"""Build report/report.md and report/report.pdf from outputs/run_summary.json (written by the notebook).

    python scripts/build_report.py [--student "Karthik Chundi"] [--where "BITS lab VM"]
"""
import argparse
import json
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--student", default="Karthik Chundi")
ap.add_argument("--where", default="", help="where the run was executed, e.g. 'BITS Prayogshala lab VM'")
ap.add_argument("--no-pdf", action="store_true")
args = ap.parse_args()

s = json.loads((ROOT / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
t1, env = s["task1_stats"], s["environment"]
where = f" on {args.where}" if args.where else ""

task1_rows = "\n".join(f"| {r['qid']} | {r['input_tokens']} | {r['output_tokens']} | {r['total_tokens']} | "
                       f"{r['latency_seconds']:.2f} | {r['total_cost_usd']:.6f} |" for r in s["task1_rows"])
names = {"v1": "v1 Basic", "v2": "v2 Structured", "v3": "v3 JSON"}
cmp_rows = []
for r in s["comparison"]:
    win = r["prompt_version"] == s["best_version"]
    cells = [names.get(r["prompt_version"], r["prompt_version"]), f"{r['avg_latency_seconds']:.2f} s",
             f"{r['avg_input_tokens']:.1f}", f"{r['avg_output_tokens']:.1f}", f"${r['avg_cost_per_request']:.6f}",
             f"{r['valid_json']:.0%}", "Yes → `production`" if win else "No"]
    cmp_rows.append("| " + " | ".join(f"**{c}**" if win and i else c for i, c in enumerate(cells)) + " |")

shots = [("01_experiment_runs.png", "Experiment `eduassist_llmops_assignment`: runs"),
         ("02b_runs_metrics_table.png", "Run comparison: parameters and metrics"),
         ("03_run_metrics_prompt_v1.png", "Run metrics: prompt v1"),
         ("03_run_metrics_prompt_v2.png", "Run metrics: prompt v2"),
         ("03_run_metrics_prompt_v3.png", "Run metrics: prompt v3"),
         ("05_prompt_registry_versions.png", "Prompt Registry: `eduassist_qa` versions with commit messages"),
         ("06_prompt_diff_compare.png", "Prompt diff (Compare view)"),
         ("07_production_alias.png", "`production` alias on the promoted version")]
# also pick up any manually captured lab screenshots (screenshots/lab_*.png)
shots += [(p.name, p.stem.replace("lab_", "Lab: ").replace("_", " "))
          for p in sorted((ROOT / "screenshots").glob("lab_*.png"))]
appendix = "\n\n".join(f'<div class="fig"><p><b>A{i}. {markdown.markdown(cap)[3:-4]}</b></p>'
                       f'<img src="../screenshots/{f}" alt="{f}"></div>'
                       for i, (f, cap) in enumerate([x for x in shots if (ROOT / "screenshots" / x[0]).exists()], 1))

md = f"""# LLMOps: API Cost, Latency, and Prompt Version Management

**Course:** MLOps, Experiential Learning Assignment 2 | **Student:** {args.student} | **Run:** {s['run_date']}{where}

**Setup:** Google Gemini Developer API (`{s['api_version']}`, `google-genai` {env['google_genai']}), model **`{s['model']}`**. Pricing: **USD {s['price_input_per_m']:.2f} per 1M input tokens and USD {s['price_output_per_m']:.2f} per 1M output tokens** (paid-tier Standard rate, published price as of {s['pricing_date']}). Free tier was used (actual spend USD 0); costs are at paid-tier rates. Python {env['python']} on {env['os']}, MLflow {env['mlflow']} (SQLite). The key came from the `GEMINI_API_KEY` environment variable.

## 1. LLMOps vs MLOps

MLOps versions models trained from data. LLMOps calls hosted models whose prompt, model version, context and guardrails change independently, and adds token billing, output-length-driven latency and non-determinism, so prompts are versioned and evaluated like model artefacts.

## 2. Task 1: Latency (prompt v1, five questions)

| Statistic | Mean | Median | Min | Max | Std (sample) |
|---|---|---|---|---|---|
| Latency (s) | {t1['mean']:.2f} | {t1['median']:.2f} | {t1['min']:.2f} | {t1['max']:.2f} | {t1['std (sample)']:.2f} |

![Task 1 latency](../outputs/task1_latency.png)

{s['task1_text']}

## 3. Task 2: Tokens and cost

| Q | Input tok | Output tok | Total tok | Latency (s) | Cost (USD) |
|---|---|---|---|---|---|
{task1_rows}

{s['task2_text']}

## 4. Task 3: Prompt versioning and evaluation in MLflow

The three prompts were registered as `eduassist_qa` versions with commit messages and run on the same five questions in experiment `eduassist_llmops_assignment`. Each run logged the five required metrics and was linked to its prompt version.

| Prompt Version | Avg Latency | Avg Input Tokens | Avg Output Tokens | Avg Cost | Valid JSON Rate | Recommended? |
|---|---|---|---|---|---|---|
{chr(10).join(cmp_rows)}

![Prompt comparison](../outputs/prompt_comparison.png)

### Promotion justification

{s['justification']}

<div style="page-break-before: always"></div>

## Appendix: MLflow screenshots

{appendix}
"""
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report" / "report.md").write_text(md, encoding="utf-8")
body_words = md.split("## Appendix")[0]
body_words = "\n".join(l for l in body_words.splitlines() if not l.startswith(("|", "!", "<")))
print(f"wrote report/report.md (~{len(body_words.split())} words excl. tables/appendix; "
      f"justification ~{len(s['justification'].split())} words)")


def launch_browser(p):
    """Bundled Playwright Chromium if installed, else the system Chrome/Edge (lab VMs often block the download)."""
    for kw in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
        try:
            return p.chromium.launch(**kw)
        except Exception as e:
            last = e
    raise SystemExit(f"No usable Chromium/Chrome found ({last}). Run: python -m playwright install chromium")


if not args.no_pdf:
    css = """
    body { font-family: 'Segoe UI', 'Liberation Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; margin: 0; }
    h1 { font-size: 17pt; margin: 0 0 6px; color: #12355b; }
    h2 { font-size: 12.5pt; margin: 16px 0 6px; color: #12355b; border-bottom: 1px solid #d0d7de; padding-bottom: 2px; }
    h3 { font-size: 11pt; margin: 12px 0 4px; }
    table { border-collapse: collapse; margin: 6px 0 10px; font-size: 9.5pt; }
    th, td { border: 1px solid #c8ced6; padding: 3px 8px; text-align: left; }
    th { background: #eef2f7; }
    img { max-width: 100%; display: block; margin: 6px auto; }
    .fig { page-break-inside: avoid; margin-bottom: 14px; }
    .fig img { max-height: 225mm; object-fit: contain; }
    code { background: #f3f4f6; padding: 0 3px; border-radius: 3px; font-size: 9pt; }
    """
    html = markdown.markdown(md, extensions=["tables"])
    html_path = ROOT / "report" / "_report.html"
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head>"
                         f"<body>{html}</body></html>", encoding="utf-8")
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page()
        page.goto(html_path.as_uri())
        page.pdf(path=str(ROOT / "report" / "report.pdf"), format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "16mm", "left": "16mm", "right": "16mm"})
        browser.close()
    html_path.unlink()
    print("wrote report/report.pdf")
