from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
import uvicorn
import os
import json
import csv
import glob

app = FastAPI(title="SimAstra Visualizer UI")

ROOT = os.path.dirname(os.path.abspath(__file__))
UI_HTML = os.path.join(ROOT, "ui", "mission_control_console.html")
EVAL_DIR = os.path.join(ROOT, "eval_output")
RUNS_DIR = os.path.join(ROOT, "runs")


def _read_csv(path):
    if not os.path.exists(path):
        raise HTTPException(404, f"not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.get("/")
def serve_visualizer():
    if not os.path.exists(UI_HTML):
        raise HTTPException(500, "mission_control_console.html not found in ui/")
    return FileResponse(UI_HTML)


@app.get("/eval/summary")
def eval_summary():
    return _read_csv(os.path.join(EVAL_DIR, "eval_summary.csv"))


@app.get("/eval/results")
def eval_results():
    return _read_csv(os.path.join(EVAL_DIR, "eval_results.csv"))


@app.get("/eval/report")
def eval_report():
    path = os.path.join(EVAL_DIR, "EVAL_REPORT.md")
    if not os.path.exists(path):
        raise HTTPException(404, "no eval report yet — run eval_suite.py")
    return PlainTextResponse(open(path, encoding="utf-8").read())


@app.get("/logs")
def list_runs():
    """List available structured-log run files."""
    files = sorted(glob.glob(os.path.join(RUNS_DIR, "*", "sim.jsonl")), reverse=True)
    return [{"run": os.path.basename(os.path.dirname(f)),
             "lines": sum(1 for _ in open(f, encoding="utf-8"))} for f in files]


@app.get("/logs/{run}")
def get_run(run: str, limit: int = 500):
    """Return the last `limit` JSONL events of a run (oldest-first tail)."""
    safe = os.path.basename(run)  # prevent traversal
    path = os.path.join(RUNS_DIR, safe, "sim.jsonl")
    if not os.path.exists(path):
        raise HTTPException(404, f"unknown run: {safe}")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
    return [json.loads(l) for l in lines if l.strip()]


if __name__ == "__main__":
    print("[INFO] Starting FastAPI frontend server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
