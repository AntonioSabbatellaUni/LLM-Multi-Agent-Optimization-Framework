"""OpenCode Go account usage (GET <base>/usage): rolling 5-hour / weekly / monthly windows -> status, percent, resetsAt.
    python3 malbo/tools/go_usage.py [--env .env] [--json] [--wait-seconds]
--wait-seconds prints the seconds until every window is usable again (0 = usable now; -1 = weekly/monthly exhausted beyond 1 day).
Stdlib only; the API key is read from OPENCODE_API_KEY / MALBO_API_KEY (env or the --env file) and never printed."""
import argparse, json, os, sys, urllib.request
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--env", default=".env"); ap.add_argument("--json", action="store_true"); ap.add_argument("--wait-seconds", action="store_true")
a = ap.parse_args()
if os.path.exists(a.env):
    for line in open(a.env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
key = os.getenv("OPENCODE_API_KEY") or os.getenv("MALBO_API_KEY")
base = os.getenv("MALBO_API_BASE", "https://opencode.ai/zen/go/v1").rstrip("/")


def usage():
    req = urllib.request.Request(base + "/usage", headers={"Authorization": f"Bearer {key}", "User-Agent": "malbo-status"})
    return json.load(urllib.request.urlopen(req, timeout=20))["usage"]


try:
    u = usage()
except Exception as e:
    print(f"usage unavailable: {type(e).__name__}", file=sys.stderr); sys.exit(2)
now = datetime.now(timezone.utc)
if a.wait_seconds:
    wait = 0
    for name, w in u.items():
        if w.get("status") != "ok":
            left = (datetime.fromisoformat(w["resetsAt"].replace("Z", "+00:00")) - now).total_seconds()
            if name != "rolling" and left > 86400:
                print(-1); sys.exit(0)
            wait = max(wait, int(left) + 120)
    print(wait); sys.exit(0)
if a.json:
    print(json.dumps(u)); sys.exit(0)
for name, w in u.items():
    t = datetime.fromisoformat(w["resetsAt"].replace("Z", "+00:00"))
    left = t - now
    h, m = divmod(int(left.total_seconds()) // 60, 60)
    print(f"Go {name:<8} {w.get('status'):<13} {w.get('percent'):>3}%   resets {t.strftime('%a %d %b %H:%M UTC')} (in {h}h{m:02d}m)")
