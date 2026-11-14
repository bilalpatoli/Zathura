"""Shared author-attributed memory; REST contract: https://past.dev/openapi.json."""
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

MEMORY_PATH = Path(__file__).resolve().parents[2] / "data" / "memory.jsonl"
PASTDEV_MIRROR_PATH = Path(__file__).resolve().parents[2] / "data" / "pastdev-memory.jsonl"
AGENTS = {"research-scientist", "skeptic"}


def _validate(agent_id, k=None):
    if agent_id not in AGENTS:
        raise ValueError(f"Unknown memory author/reader: {agent_id}")
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 200):
        raise ValueError("k must be between 1 and 200")


def _entry(agent_id, content, kind, refs):
    _validate(agent_id)
    if not content or not isinstance(kind, str) or not kind.strip():
        raise ValueError("Memory content and kind must not be empty")
    if refs is not None and (not isinstance(refs, list) or any(not isinstance(r, str) for r in refs)):
        raise ValueError("refs must be a list of memory ids")
    return {"memory_id": uuid4().hex, "agent_id": agent_id, "content": content,
            "kind": kind, "refs": refs or [], "timestamp": datetime.now(timezone.utc).isoformat()}


class LocalBackend:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else MEMORY_PATH

    def write_memory(self, agent_id, content, kind, refs=None):
        entry = _entry(agent_id, content, kind, refs)
        entry["write_via"] = "local_jsonl"
        self.store(entry)
        return entry["memory_id"]

    def store(self, entry):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    def recall_memory(self, agent_id, query, k=5):
        _validate(agent_id, k)
        if not self.path.exists():
            return []
        entries = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        words = set(re.findall(r"[a-z0-9_]+", query.lower()))
        def score(entry):
            tokens = set(re.findall(r"[a-z0-9_]+", json.dumps(entry["content"]).lower()))
            return len(words & tokens), entry["timestamp"]
        return [entry | {"read_via": "local_jsonl"} for entry in sorted(entries, key=score, reverse=True)[:k]]


class PastDevBackend:
    """Project-visible sources preserve authors via original-envelope metadata.

    Configured-backend errors propagate rather than splitting shared history.
    """
    base_url = "https://api.past.dev/api/v1"

    def __init__(self, api_key, mirror_path=None):
        self.api_key = api_key
        self.mirror = LocalBackend(mirror_path if mirror_path is not None else PASTDEV_MIRROR_PATH)

    def _request(self, method, path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base_url + path, data=body, method=method,
                                         headers={"Authorization": f"Bearer {self.api_key}",
                                                  "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    def write_memory(self, agent_id, content, kind, refs=None):
        entry = _entry(agent_id, content, kind, refs)
        receipt = self._request("POST", "/ingest", {
            "id": entry["memory_id"], "content": json.dumps(entry),
            "label": f"Zathura {agent_id}: {kind}", "timestamp": entry["timestamp"],
            "identity": agent_id, "audience": "project", "metadata": {"zathura": entry},
        })
        if receipt["status"] not in ("queued", "completed"):
            raise RuntimeError("Past.dev did not accept memory ingestion")
        entry["pastdev"] = {"ingestion_id": receipt["ingestionId"], "status": receipt["status"]}
        self.mirror.store(entry)
        return entry["memory_id"]

    def wait_until_ready(self, memory_id, timeout=60):
        entries = self.mirror.recall_memory("research-scientist", "", k=200)
        entry = next(m for m in entries if m["memory_id"] == memory_id)
        deadline = time.monotonic() + timeout
        status = entry["pastdev"]["status"]
        while status != "completed":
            if status == "failed":
                raise RuntimeError("Past.dev ingestion failed; memory was not indexed")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Past.dev ingestion still pending: {entry['pastdev']['ingestion_id']}")
            time.sleep(0.5)
            status = self._request("GET", f"/ingest/{entry['pastdev']['ingestion_id']}")["status"]
        return True

    def recall_memory(self, agent_id, query, k=5):
        local = self.mirror.recall_memory(agent_id, query, k)
        if local:
            return [entry | {"read_via": "pastdev_mirror"} for entry in local]
        remote = self.recall_remote(agent_id, query, k)
        for entry in remote:
            self.mirror.store(entry)
        return remote

    def recall_remote(self, agent_id, query, k=5):
        """Explicit server-only recall for synchronization and integration checks."""
        _validate(agent_id, k)
        response = self._request("POST", "/recall", {"query": query or "Zathura recent experiments",
                                                       "identity": agent_id, "limit": k, "level": "low"})
        entries = {}
        for result in response["results"]:
            for source in result["sources"]:
                metadata = source.get("metadata")
                entry = metadata.get("zathura") if isinstance(metadata, dict) else None
                if isinstance(entry, dict) and entry.get("agent_id") in AGENTS:
                    entries.setdefault(entry["memory_id"], entry)
        return [entry | {"read_via": "pastdev_remote"} for entry in list(entries.values())[:k]]


def get_backend():
    key = os.getenv("PASTDEV_API_KEY")
    return PastDevBackend(key) if key else LocalBackend()


def write_memory(agent_id, content, kind, refs=None):
    return get_backend().write_memory(agent_id, content, kind, refs)


def recall_memory(agent_id, query, k=5):
    entries = get_backend().recall_memory(agent_id, query, k)
    read_at = datetime.now(timezone.utc).isoformat()
    return [entry | {"read_at": read_at} for entry in entries]


def memory_receipts(entries):
    return [{"memory_id": entry["memory_id"], "author": entry["agent_id"],
             "read_via": entry.get("read_via", "not_recorded"), "read_at": entry.get("read_at")}
            for entry in entries]
