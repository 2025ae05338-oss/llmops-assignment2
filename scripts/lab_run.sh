#!/usr/bin/env bash
# One-command lab runner for LLMOps Assignment 2 (Linux lab VM).
#
#   bash scripts/lab_run.sh all       # setup -> run notebook -> MLflow UI -> screenshots -> report -> zip
#   bash scripts/lab_run.sh setup     # venv + dependencies + API key
#   bash scripts/lab_run.sh run       # fresh MLflow DB, execute notebook, export HTML
#   bash scripts/lab_run.sh ui        # start MLflow UI on :5000 (background)
#   bash scripts/lab_run.sh shots     # automated MLflow screenshots (needs ui)
#   bash scripts/lab_run.sh report    # build report/report.pdf from this run
#   bash scripts/lab_run.sh package   # build the submission ZIP
#   bash scripts/lab_run.sh stop      # stop the MLflow UI
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
VENV=.venv-lab
PORT=${MLFLOW_PORT:-5000}
WHERE=${LAB_WHERE:-BITS lab VM}
export MLFLOW_DISABLE_AGENT_HINT=1

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

pick_python() {
  for c in python3.12 python3.11 python3.10 "$HOME/bin/python3" python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      echo "$c"; return
    fi
  done
  echo "ERROR: need Python >= 3.11 (try: ls /usr/bin/python3*)" >&2; exit 1
}

setup() {
  local PY; PY=$(pick_python)
  say "Using $($PY --version) ($PY)"
  [ -d "$VENV" ] || "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r requirements.txt
  "$VENV/bin/python" -m ipykernel install --sys-prefix --name llmops-a2 --display-name "Python (llmops-a2)" >/dev/null
  say "Installing Playwright Chromium (falls back to system Chrome if this fails)"
  "$VENV/bin/python" -m playwright install chromium >/dev/null 2>&1 || echo "  (download failed; system Chrome will be used)"

  if [ -z "${GEMINI_API_KEY:-}" ] && ! grep -qE '^GEMINI_API_KEY=.{20,}' .env 2>/dev/null; then
    say "Paste your Gemini API key (input hidden; saved to .env with mode 600)"
    read -rsp "GEMINI_API_KEY: " KEY; echo
    (umask 077; printf 'GEMINI_API_KEY=%s\n' "$KEY" > .env)
  fi
  "$VENV/bin/python" - <<'EOF'
import os
from dotenv import load_dotenv
from google import genai
load_dotenv(".env")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
r = client.models.generate_content(
    model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"), contents="Reply with OK")
print("  API check:", r.text.strip()[:20], "| tokens in/out:", r.usage_metadata.prompt_token_count, r.usage_metadata.candidates_token_count)
EOF
}

run() {
  say "Fresh MLflow store (old one backed up)"
  [ -f mlflow.db ] && mv mlflow.db "mlflow.db.bak.$(date +%s)"
  rm -rf mlruns
  say "Executing notebook (20 API calls, ~2-3 min)"
  "$VENV/bin/jupyter" nbconvert --to notebook --execute --inplace llmops_assignment2.ipynb \
      --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=llmops-a2
  "$VENV/bin/jupyter" nbconvert --to html llmops_assignment2.ipynb
  "$VENV/bin/python" - <<'EOF'
import pandas as pd
print(pd.read_csv("prompt_evaluation_results.csv")[["prompt_version", "avg_latency_seconds", "avg_input_tokens",
      "avg_output_tokens", "avg_cost_per_request", "valid_json", "recommended"]].to_string(index=False))
EOF
}

ui() {
  if curl -s -o /dev/null "http://127.0.0.1:$PORT"; then say "MLflow UI already up on :$PORT"; return; fi
  say "Starting MLflow UI on http://127.0.0.1:$PORT"
  nohup "$VENV/bin/mlflow" ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port "$PORT" > mlflow_ui.log 2>&1 &
  echo $! > .mlflow_ui.pid
  for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$PORT" && break; sleep 1; done
  echo "  open http://127.0.0.1:$PORT in the lab browser"
}

shots() {
  ui
  say "Capturing MLflow screenshots"
  "$VENV/bin/python" scripts/capture_screenshots.py --url "http://127.0.0.1:$PORT"
}

report() {
  say "Building report/report.pdf from outputs/run_summary.json"
  "$VENV/bin/python" scripts/build_report.py --where "$WHERE"
}

package() {
  local ZIP="MLOps_Assignment2_${STUDENT_TAG:-Chodavarapu_Venkata_Hemalatha}.zip" STAGE; STAGE=$(mktemp -d)
  mkdir -p "$STAGE/llmops-assignment2"
  cp -r llmops_assignment2.ipynb llmops_assignment2.html requirements.txt README.md .env.example .gitignore \
        latency_tokens_cost.csv prompt_evaluation_results.csv outputs screenshots report scripts docs "$STAGE/llmops-assignment2/"
  rm -f "$ZIP"
  (cd "$STAGE" && "$ROOT/$VENV/bin/python" -m zipfile -c "$ROOT/$ZIP" llmops-assignment2)
  rm -rf "$STAGE"
  if "$VENV/bin/python" -m zipfile -l "$ZIP" | grep -q '\.env$'; then echo "ERROR: .env in zip" >&2; exit 1; fi
  say "Wrote $ZIP ($(du -h "$ZIP" | cut -f1))"
}

stop() {
  [ -f .mlflow_ui.pid ] && kill "$(cat .mlflow_ui.pid)" 2>/dev/null && rm -f .mlflow_ui.pid && say "MLflow UI stopped" || true
}

case "${1:-all}" in
  setup|run|ui|shots|report|package|stop) "$1" ;;
  all) setup; run; shots; report; package
       say "Done. MLflow UI is still running on :$PORT for manual lab screenshots (bash scripts/lab_run.sh stop when finished)." ;;
  *) echo "usage: $0 {all|setup|run|ui|shots|report|package|stop}"; exit 1 ;;
esac
