"""Read-only dashboard snapshots and a process-local, single-mission worker."""
import json
import os
import threading
from pathlib import Path

from zathura import storage
from zathura.agents.memory import LocalBackend, MEMORY_PATH, PASTDEV_MIRROR_PATH
from zathura.engine import cached_targets
from zathura.orchestrator import run_research_loop

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows


def snapshot():
    events = read_jsonl(ROOT / "data/events.jsonl")
    try:
        records = [r.model_dump(mode="json") for r in storage.load_records()]
        warning = None
    except (ValueError, OSError):
        records, warning = [], "Experiment log is incomplete or invalid; retry after the writer finishes."
    backend = LocalBackend(PASTDEV_MIRROR_PATH if os.getenv("PASTDEV_API_KEY") else MEMORY_PATH)
    try:
        memories = backend.recall_memory("research-scientist", "", 200)
    except (ValueError, OSError):
        memories = []
        warning = "Memory mirror is being updated or is invalid; refresh to retry."
    loop_id = events[-1].get("loop_id") if events else None
    latest = [e for e in events if e.get("loop_id") == loop_id]
    baseline = next((e["payload"] for e in reversed(latest) if e.get("step") == "baseline"), None)
    ids = {e["payload"]["record"]["experiment_id"] for e in latest
           if e.get("step") == "iteration_complete" and "record" in e.get("payload", {})}
    current = [r for r in records if r["experiment_id"] in ids]
    return {"events": events, "latest": latest, "records": records, "current": current,
            "baseline": baseline, "memories": memories, "targets": cached_targets(),
            "holdout": read_jsonl(ROOT / "data/holdout.jsonl"), "warning": warning}


class Mission:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.iteration = 0
        self.error = None

    def status(self):
        with self.lock:
            return {"running": self.running, "iteration": self.iteration, "error": self.error}

    def on_event(self, event):
        with self.lock:
            self.iteration = event.get("payload", {}).get("iteration", self.iteration) if isinstance(event.get("payload"), dict) else self.iteration

    def launch(self, iterations):
        with self.lock:
            if self.running:
                return False
            self.running, self.iteration, self.error = True, 0, None
        threading.Thread(target=self._run, args=(iterations,), daemon=True, name="zathura-mission").start()
        return True

    def _run(self, iterations):
        try:
            run_research_loop(iterations, on_event=self.on_event)
        except Exception as exc:
            with self.lock:
                self.error = type(exc).__name__
        finally:
            with self.lock:
                self.running = False
