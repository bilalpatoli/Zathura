"""Read-only projections of recorded sponsor evidence; legacy gaps stay explicit."""

AGENT_NAMES = {"research-scientist": "Scientist", "skeptic": "Skeptic"}
TRANSPORT_NAMES = {"pastdev_remote": "past.dev remote", "pastdev_mirror": "past.dev local mirror",
                   "local_jsonl": "Local JSONL", "not_recorded": "Not recorded"}


def config_change(before, after):
    before, after = before or {}, after or {}
    changes = [f"{key}: {before[key]} → {value}" for key, value in after.items()
               if key in before and before[key] != value]
    return " · ".join(changes) or "Configuration change not recorded"


def proposal_chains(events):
    proposals = [e for e in events if e.get("step") in ("hypothesis", "revision")]
    latest = {e["payload"].get("iteration"): e for e in proposals}
    completed = {e["payload"].get("iteration"): e["payload"] for e in events
                 if e.get("step") == "iteration_complete"}
    return [{"proposal": e["payload"], "revision": e["step"] == "revision",
             "superseded": latest[e["payload"].get("iteration")] is not e,
             "completion": completed.get(e["payload"].get("iteration"))
                           if latest[e["payload"].get("iteration")] is e else None}
            for e in proposals]


def memory_activity(events, memories):
    by_id = {m["memory_id"]: m for m in memories}
    activity = []
    for event in events:
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        for read in payload.get("memory_reads", []):
            activity.append({"Time": read.get("read_at") or event.get("timestamp", ""),
                "Agent": AGENT_NAMES.get(event.get("agent"), event.get("agent", "Unknown")), "Action": "Read context",
                "Memory author": AGENT_NAMES.get(read.get("author"), read.get("author", "Unknown")),
                "Memory ID": read["memory_id"],
                "Transport": TRANSPORT_NAMES.get(read.get("read_via"), "Not recorded")})
        memory_id = payload.get("result_memory_id", payload.get("memory_id"))
        if memory_id:
            entry = by_id.get(memory_id, {})
            fallback_author = event.get("agent") if event.get("agent") in AGENT_NAMES else "Not recorded"
            author = entry.get("agent_id", fallback_author)
            activity.append({"Time": entry.get("timestamp", event.get("timestamp", "")),
                "Agent": AGENT_NAMES.get(author, author), "Action": "Write " + entry.get("kind", "memory"),
                "Memory author": AGENT_NAMES.get(author, author), "Memory ID": memory_id,
                "Transport": "past.dev accepted write" if entry.get("pastdev") else
                             "Local JSONL" if entry.get("write_via") == "local_jsonl" else "Write receipt not recorded"})
    return sorted(activity, key=lambda row: row["Time"])
