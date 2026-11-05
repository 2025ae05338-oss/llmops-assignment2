# Lab run: LLMOps Assignment 2

Total time is about 15 minutes; most of it is `pip install`. The notebook makes 20 Gemini API calls.

## 0. Get the code into the lab VM
```bash
cd ~/Documents
git clone https://github.com/2025ae05338-oss/llmops-assignment2.git
cd llmops-assignment2
```

## 1. Python 3.11+
The Rocky VM's default `python3` is 3.9. The script picks `python3.12`, `python3.11` or `python3.10` automatically. If none of those are on PATH, run this (it is also what was used for the earlier assignment):
```bash
mkdir -p ~/bin && ln -sf /usr/bin/python3.11 ~/bin/python3 && export PATH=~/bin:$PATH
```

## 2. Run everything
```bash
bash scripts/lab_run.sh all
```
- The script asks once for **GEMINI_API_KEY** with hidden input and saves it to `.env` (mode 600). Paste the **new, rotated** key.
- Steps it runs: venv → deps → API check → fresh MLflow DB → execute notebook → HTML export → MLflow UI on :5000 → 10 screenshots → `report/report.pdf` → `MLOps_Assignment2_Chodavarapu_VenkataHemalatha.zip`.
- Re-run a single step with `bash scripts/lab_run.sh run|shots|report|package`.

## 3. Manual lab screenshots (proof the run happened in the lab)
In the lab browser, open http://127.0.0.1:5000 and take 2-3 screenshots. **Crop out the address bar**, because the lab URL carries a session token. Save them into `screenshots/` with names starting `lab_`, for example:
- `lab_terminal_run.png`: the terminal showing the comparison table printed at the end of `lab_run.sh run`
- `lab_mlflow_prompts.png`: Prompts → eduassist_qa with `@production` on version 3
- `lab_mlflow_runs.png`: the experiment runs page

Then rebuild the report and zip. Any `lab_*.png` files are added to the appendix automatically:
```bash
bash scripts/lab_run.sh report && bash scripts/lab_run.sh package
```

## 4. Finish
```bash
bash scripts/lab_run.sh stop
rm .env          # do not leave the key on the shared VM
```
Submit `MLOps_Assignment2_Chodavarapu_VenkataHemalatha.zip`. Upload it to Drive or email it to yourself from the lab browser.
