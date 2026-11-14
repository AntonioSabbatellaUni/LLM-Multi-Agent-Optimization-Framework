"""Tiny heartbeat writer: merges fields into <output_dir>/progress.json (atomic replace, thread-safe).
Read by tools/malbo_status.py. Never raises (progress must not break a run)."""
import json
import os
import threading
from datetime import datetime, timezone

_lock = threading.Lock()


def update(output_dir, **fields):
    if not output_dir:
        return
    path = os.path.join(str(output_dir), "progress.json")
    try:
        with _lock:
            try:
                cur = json.load(open(path))
            except Exception:
                cur = {}
            cur.update(fields)
            cur["updated_at"] = datetime.now(timezone.utc).isoformat()
            os.makedirs(str(output_dir), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cur, f, indent=1, default=str)
            os.replace(tmp, path)
    except Exception as e:  # pragma: no cover
        print(f"[run_progress] could not write progress.json: {type(e).__name__}")
