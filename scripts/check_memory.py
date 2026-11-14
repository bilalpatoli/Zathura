"""Live Past.dev cross-agent round trip; never accepts local recall as proof."""
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zathura.agents.memory import PastDevBackend, get_backend


def main():
    backend = get_backend()
    if not isinstance(backend, PastDevBackend):
        raise RuntimeError("PASTDEV_API_KEY is required for this live check")
    token = "Zathura roundtrip " + uuid4().hex
    written = []
    for author in ("research-scientist", "skeptic"):
        started = time.perf_counter()
        memory_id = backend.write_memory(author, f"{token}: shared-memory evidence authored by {author}.", "integration-check")
        print(json.dumps({"step": "write_accepted", "memory_id": memory_id, "author": author,
                          "runtime_s": time.perf_counter() - started}), flush=True)
        written.append((author, memory_id))
    for author, memory_id in written:
        backend.wait_until_ready(memory_id, timeout=120)
        reader = "skeptic" if author == "research-scientist" else "research-scientist"
        remote = backend.recall_remote(reader, token + " " + author, k=20)
        match = next((m for m in remote if m["memory_id"] == memory_id and m["agent_id"] == author), None)
        if match is None:
            raise AssertionError(f"Server-only recall by {reader} did not return {memory_id}")
        print(json.dumps({"step": "remote_cross_agent_recall", "reader": reader,
                          "memory_id": match["memory_id"], "author": match["agent_id"]}), flush=True)
    print("PASS: both cross-agent recalls came from Past.dev, not the local mirror.", flush=True)


if __name__ == "__main__":
    main()
