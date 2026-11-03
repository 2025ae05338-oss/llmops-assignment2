"""Capture MLflow UI screenshots for the submission (needs a running `mlflow ui`).

    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    python scripts/capture_screenshots.py --url http://127.0.0.1:5000
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote

import mlflow
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:5000")
ap.add_argument("--db", default="sqlite:///mlflow.db")
ap.add_argument("--out", default="screenshots")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(exist_ok=True)
mlflow.set_tracking_uri(args.db)
client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("eduassist_llmops_assignment")
runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
run_ids = [r.info.run_id for r in runs if r.info.run_name.startswith("prompt_v")]
prod = mlflow.genai.load_prompt("prompts:/eduassist_qa@production")
latest = max(int(v.version) for v in client.search_prompt_versions("eduassist_qa"))
base, eid = args.url.rstrip("/"), exp.experiment_id



def launch_browser(p):
    """Bundled Playwright Chromium if installed, else the system Chrome/Edge (lab VMs often block the download)."""
    for kw in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
        try:
            return p.chromium.launch(**kw)
        except Exception as e:
            last = e
    raise SystemExit(f"No usable Chromium/Chrome found ({last}). Run: python -m playwright install chromium")


def close_assistant(page):
    # The MLflow Assistant side panel eats a third of the viewport; close it if present.
    for btn in page.locator('button[aria-label="Close"]').all():
        box = btn.bounding_box()
        if box and box["x"] > 1400 and box["y"] < 80:
            btn.click()
            time.sleep(0.8)
            return


def shot(page, route, name, wait=4, full_page=False, before=None):
    page.goto(f"{base}/#{route}")
    time.sleep(wait)
    close_assistant(page)
    if before:
        before(page)
        time.sleep(2)
    page.screenshot(path=str(out / name), full_page=full_page)
    print("saved", out / name)


with sync_playwright() as p:
    browser = launch_browser(p)
    page = browser.new_page(viewport={"width": 1600, "height": 950})

    # 1. Experiment runs list
    shot(page, f"/experiments/{eid}/runs?workflowType=machine_learning", "01_experiment_runs.png")

    # 2. Metrics comparison across the three prompt-version runs
    q = f"/compare-runs?runs={quote(json.dumps(run_ids))}&experiments={quote(json.dumps([eid]))}"
    shot(page, q, "02_runs_metrics_comparison.png", wait=6, full_page=True)

    # The compare page scrolls inside a container, so full_page misses the metrics table;
    # a very tall viewport renders the whole table instead.
    page.set_viewport_size({"width": 1600, "height": 2600})
    shot(page, q, "02b_runs_metrics_table.png", wait=6)
    page.set_viewport_size({"width": 1600, "height": 950})

    # 3. Each run's metrics page (shows every logged metric incl. valid_json)
    for r in runs:
        if r.info.run_name.startswith("prompt_v"):
            shot(page, f"/experiments/{eid}/runs/{r.info.run_id}", f"03_run_metrics_{r.info.run_name}.png", wait=5)

    # 4. Prompt Registry: list and versions
    shot(page, "/prompts", "04_prompt_registry_list.png")
    shot(page, f"/prompts/eduassist_qa?promptVersion={latest}", "05_prompt_registry_versions.png")

    # 5. Prompt diff (Compare tab)
    def compare(pg):
        pg.get_by_text("Compare", exact=True).first.click()
        time.sleep(1.5)
        pg.get_by_text("Text", exact=True).first.click()  # Markdown view disables diff highlighting
    shot(page, f"/prompts/eduassist_qa?promptVersion={latest}", "06_prompt_diff_compare.png", before=compare)

    # 6. Production alias on the promoted version
    def preview(pg):
        pg.get_by_text("Preview", exact=True).first.click()  # the Compare tab is sticky from the previous shot
    shot(page, f"/prompts/eduassist_qa?promptVersion={prod.version}", "07_production_alias.png", before=preview)

    browser.close()
