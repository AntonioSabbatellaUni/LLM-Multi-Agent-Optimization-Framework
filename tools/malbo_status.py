#!/usr/bin/env python3
"""MALBO run status monitor (stdlib only; uses `rich` for colours if installed). Runs on the host, no container needed.

  python3 malbo/tools/malbo_status.py <output_dir> [--watch N] [--json] [--no-rich] [--go-budget USD]

Reads only files the run writes:
  <output_dir>/progress.json          heartbeat of the optimizer/evaluator (phase, eval index, config, n_initial ...)
  <output_dir>/evaluations.jsonl      one record per finished evaluation (accuracy, cost_report, wall_time_s ...)
  <output_dir>/initial_design.json    planned initial configurations
  <output_dir>/current_session.json   -> session dir of the evaluation in progress
  <session>/progress.json, questions.jsonl, llm_calls.jsonl, events.jsonl   (question level, LLM calls, retries)
The session dir is resolved as written, else under smolagents_opt/examples/open_deep_research/experiments/<basename>
(the run happens in a container whose paths differ from the host's).
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSIONS = REPO / "smolagents_opt" / "examples" / "open_deep_research" / "experiments"
WINDOW_S = 600
_CACHE = {}  # path -> [offset, [slim records]] (incremental parsing for --watch; llm_calls.jsonl lines are big)


def ts(s):
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except Exception:
        return None


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def read_jsonl(p, slim=None):
    """Incremental: only complete new lines since the last call. `slim` reduces each record (None = keep as is)."""
    p = str(p)
    ent = _CACHE.setdefault(p, [0, []])
    try:
        size = os.path.getsize(p)
    except OSError:
        return []
    if size < ent[0]:
        ent[0], ent[1] = 0, []
    if size > ent[0]:
        with open(p, "rb") as f:
            f.seek(ent[0])
            data = f.read()
        end = data.rfind(b"\n")
        if end >= 0:
            for line in data[:end].splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ent[1].append(slim(r) if slim else r)
            ent[0] += end + 1
    return ent[1]


def slim_call(r):
    u = (r.get("usage") or {})
    oc = ((u.get("raw") or {}).get("opencode") or {})
    return {"t": ts(r.get("ts_end")), "status": r.get("http_status"), "error": bool(r.get("error")), "will_retry": bool(r.get("will_retry")),
            "agent": r.get("agent"), "task_id": r.get("task_id"), "model": r.get("model"),
            "list_cost": float(oc.get("list_cost") or 0), "out": u.get("completion_tokens") or 0}


def resolve_session(path):
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    q = SESSIONS / p.name
    return q if q.exists() else None


def is_peak(t):
    d = datetime.fromtimestamp(t, timezone.utc)
    return d.weekday() < 5 and (1 <= d.hour < 4 or 6 <= d.hour < 10)


def next_peak_change(t):
    """(is_peak, seconds until the status flips) scanning minute by minute (max 8 days)."""
    cur = is_peak(t)
    for m in range(1, 8 * 24 * 60):
        if is_peak(t + 60 * m) != cur:
            return cur, 60 * m - (t % 60)
    return cur, None


def short(opt):
    return str(opt) if opt else "?"


def eval_options(rec):
    if rec.get("options"):
        return rec["options"]
    cfg = rec.get("config") or {}
    out = {}
    for slot, role in (("manager", "manager"), ("search_agent", "search_agent"), ("docs", "text_inspector")):
        if role in cfg:
            out[slot] = cfg[role].get("pool_option") or cfg[role].get("model_id")
    return out


def simulate_eta(n, done, running_elapsed, p, med):
    """Greedy p-worker simulation: running questions finish after max(med-elapsed, 0.1*med), queued ones take med each."""
    if med is None:
        return None
    free = [max(med - e, 0.1 * med) for e in running_elapsed][:p]
    free += [0.0] * (p - len(free))
    queued = max(n - done - len(running_elapsed), 0)
    for _ in range(queued):
        i = free.index(min(free))
        free[i] += med
    return max(free) if (queued or running_elapsed) else 0.0


def collect(out_dir, go_budget=None, now=None):
    now = now or time.time()
    out = Path(out_dir)
    P = read_json(out / "progress.json") or {}
    E = read_jsonl(out / "evaluations.jsonl")
    D = read_json(out / "initial_design.json") or {}
    cur = read_json(out / "current_session.json") or {}
    n_init = P.get("n_initial") or (len(D.get("points", [])) or None)
    n_it = P.get("n_iterations") or 0
    q = P.get("batch_size") or 1
    total_evals = (n_init or 0) + n_it * q
    evals_done = len(E)
    finished = bool(P.get("done"))

    # per finished evaluation
    evals = []
    for r in E:
        cr = r.get("cost_report") or {}
        evals.append({"index": r.get("eval_index", r.get("config_index")), "options": eval_options(r), "accuracy": r.get("accuracy"),
                      "aa_cost": cr.get("aa_cost", r.get("cost_objective")), "go_expected": cr.get("go_expected"),
                      "go_list_cost": cr.get("go_list_cost"), "go_allowance": cr.get("go_allowance"), "mismatch": cr.get("mismatch"),
                      "wall_time_s": r.get("wall_time_s"), "n_calls": cr.get("n_calls")})
    sums = {k: sum(float(e[k] or 0) for e in evals) for k in ("aa_cost", "go_expected", "go_list_cost", "go_allowance")}

    # current evaluation (question level)
    sess = resolve_session(cur.get("session_dir"))
    S = read_json(sess / "progress.json") if sess else None
    es = ts(P.get("eval_started_at"))
    if S and es and (ts(S.get("started_at")) or 0) < es - 2 and not finished:  # session of the previous evaluation: the new one has not started its session yet
        sess, S = None, None
    quests = read_jsonl(sess / "questions.jsonl") if sess else []
    calls = read_jsonl(sess / "llm_calls.jsonl", slim_call) if sess else []
    events = read_jsonl(sess / "events.jsonl") if sess else []
    current = None
    in_flight_eval = (not finished) and (P.get("eval_index") is not None) and (P.get("eval_index") >= evals_done) and S and not S.get("finished")
    if S:
        n_q = S.get("n_questions") or 0
        par = S.get("parallel") or 1
        running = S.get("running") or {}
        durs = [x["duration_s"] for x in quests if x.get("duration_s")]
        med = statistics.median(durs) if durs else None
        start = ts(S.get("started_at"))
        run_el = [now - ts(v["started_at"]) for v in running.values() if ts(v.get("started_at"))]
        eta = simulate_eta(n_q, S.get("done", 0), run_el, par, med) if not S.get("finished") else 0.0
        cur_list = sum(c["list_cost"] for c in calls)
        current = {"session_dir": str(sess), "n_questions": n_q, "done": S.get("done", 0), "correct": S.get("correct", 0),
                   "running": len(running), "running_tasks": sorted(v["index"] + 1 for v in running.values()), "parallel": par,
                   "elapsed_s": (now - start) if start else None, "median_question_s": med, "eta_s": eta,
                   "finished": bool(S.get("finished")), "go_list_cost_so_far": cur_list, "in_flight": bool(in_flight_eval)}
    # totals incl. the evaluation in flight (list cost only; the others are computed at the end of an evaluation)
    live_list = current["go_list_cost_so_far"] if current and current["in_flight"] else 0.0

    # overall ETA
    walls = [e["wall_time_s"] for e in evals if e.get("wall_time_s")]
    overall = None
    if total_evals and not finished:
        remaining_after = max(total_evals - evals_done - (1 if current and current["in_flight"] else 0), 0)
        per_eval = statistics.median(walls) if walls else (
            (current["median_question_s"] * math.ceil(current["n_questions"] / current["parallel"])) if current and current.get("median_question_s") else None)
        cur_eta = current["eta_s"] if current and current["in_flight"] else 0.0
        if per_eval is not None and cur_eta is not None:
            overall = cur_eta + remaining_after * per_eval

    # activity in the last 10 minutes + totals
    w0 = now - WINDOW_S
    last = lambda items, key: [i for i in items if (i.get(key) or 0) >= w0]
    ev_t = [dict(e, _t=ts(e.get("ts"))) for e in events]
    win_calls = last(calls, "t")
    win_ev = [e for e in ev_t if (e["_t"] or 0) >= w0]
    activity = {"window_s": WINDOW_S, "calls": len(win_calls), "retries": len(win_ev),
                "http_429": sum(1 for c in win_calls if c.get("status") == 429),
                "errors": sum(1 for c in win_calls if c.get("error")),
                "total_calls": len(calls), "total_retries": len(events),
                "total_429": sum(1 for c in calls if c.get("status") == 429), "total_errors": sum(1 for c in calls if c.get("error")),
                "last_call_age_s": (now - max((c["t"] for c in calls if c.get("t")), default=0)) if calls else None}
    hb = max([x for x in (ts(P.get("updated_at")), ts((S or {}).get("updated_at")), max((c["t"] for c in calls if c.get("t")), default=None)) if x] or [0]) or None
    pk, secs = next_peak_change(now)
    peak = {"is_peak_now": pk, "seconds_to_change": secs}
    mism = any(e.get("mismatch") for e in evals) or (
        sums["go_list_cost"] > 0 and abs(sums["go_expected"] - sums["go_list_cost"]) > 0.02 * sums["go_list_cost"])
    return {"now": datetime.fromtimestamp(now, timezone.utc).isoformat(), "output_dir": str(out), "run_name": P.get("run_name") or out.name,
            "started_at": P.get("started_at"), "phase": P.get("phase"), "finished": finished, "seed": P.get("seed"),
            "initial_design": P.get("initial_design"), "heartbeat_age_s": (now - hb) if hb else None,
            "n_initial": n_init, "n_iterations": n_it, "bo_iteration": P.get("bo_iteration"), "evals_done": evals_done,
            "total_evals": total_evals, "current_index": P.get("eval_index"), "current_config": P.get("config"),
            "evals": evals, "totals": dict(sums, go_list_cost_with_current=sums["go_list_cost"] + live_list, mismatch=mism,
                                           go_budget=go_budget, go_budget_left=(None if go_budget is None else go_budget - sums["go_list_cost"] - live_list)),
            "current": current, "overall_eta_s": overall, "activity": activity, "peak": peak,
            "planned": [pt["options"] for pt in D.get("points", [])]}


def dur(s):
    if s is None:
        return "?"
    s = int(max(s, 0))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def bar(i, n, w=28):
    k = int(round(w * min(i, n) / n)) if n else 0
    return "[" + "#" * k + "-" * (w - k) + f"] {i}/{n or '?'}"


def render(d):
    """-> list of (style, text)."""
    L = []
    add = lambda t="", st=None: L.append((st, t))
    add(f"MALBO status  {d['run_name']}   ({d['output_dir']})", "bold")
    add(f"started {d['started_at'] or '?'}   phase {d['phase'] or '?'}   seed {d['seed']}   design {d['initial_design']}"
        f"   heartbeat {dur(d['heartbeat_age_s'])} ago" + ("   FINISHED" if d["finished"] else ""))
    pk = d["peak"]
    if pk["is_peak_now"]:
        add(f"!! GO PEAK HOURS now (weekday 01-04 / 06-10 UTC): deepseek/gemini cost x2 (ends in {dur(pk['seconds_to_change'])})", "bold red")
    elif pk["seconds_to_change"]:
        add(f"off-peak; next Go peak window (x2 deepseek/gemini) in {dur(pk['seconds_to_change'])}", "dim")
    try:   # account-wide Go windows (GET /usage) for every key (.env, .env.go2, ...), cached 60 s
        import subprocess, time as _t, glob as _g, json as _j
        g = globals().setdefault("_GO_USAGE", {"t": 0, "u": []})
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        if _t.time() - g["t"] > 60:
            g["u"] = []
            for envf in sorted(_g.glob(os.path.join(root, ".env")) + _g.glob(os.path.join(root, ".env.go*"))):
                r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "go_usage.py"), "--env", envf],
                                   capture_output=True, text=True, timeout=25).stdout.strip()
                rj = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "go_usage.py"), "--env", envf,
                                     "--json"], capture_output=True, text=True, timeout=25).stdout.strip()
                g["u"].append((os.path.basename(envf), r, _j.loads(rj) if rj else {}))
            g["t"] = _t.time()
        # $ left per key: monthly allowance MALBO_GO_MONTHLY (default 52), weekly = 50%, 5-hour = 20% of it
        M = float(os.getenv("MALBO_GO_MONTHLY", "52")); share = {"rolling": 0.2, "weekly": 0.5, "monthly": 1.0}
        al = [float(e["go_allowance"]) for e in d["evals"] if e.get("go_allowance")]
        per = sum(al) / len(al) if al else None
        now_t = week_t = 0.0
        for name, txt, u in g["u"]:
            add(f"-- key {name}", "dim")
            for line in txt.splitlines():
                add("  " + line, "red" if "limited" in line else "dim")
            left = {k: (0.0 if w.get("status") != "ok" else M * share.get(k, 1) * (1 - float(w.get("percent") or 0) / 100))
                    for k, w in u.items()}
            if left:
                now_t += min(left.values()); week_t += min(left.get("weekly", 0), left.get("monthly", 0))
        if per:
            add(f"Go capacity: now ${now_t:.2f} (~{now_t / per:.1f} points)   until weekly reset ${week_t:.2f} (~{week_t / per:.1f} points)"
                f"   at ${per:.2f} allowance/point (mean of completed)", "bold")
    except Exception:
        pass
    add()
    add(f"Evaluations  {bar(d['evals_done'], d['n_initial'] or d['total_evals'])} initial design"
        + (f"   BO iterations {d['bo_iteration'] or 0}/{d['n_iterations']}" if d["n_iterations"] else ""), "bold")
    c = d["current"]
    if d["current_config"] and not d["finished"]:
        add("  current #%s: %s" % (d["current_index"], "  ".join(f"{k}={short(v)}" for k, v in d["current_config"].items())))
    if c:
        add(f"Questions    {bar(c['done'], c['n_questions'])}   running now {c['running']} (q {c['running_tasks']}) of parallel {c['parallel']}"
            f"   correct {c['correct']}/{c['done']}" + ("   [evaluation finished]" if c["finished"] else ""), "bold")
        add(f"  elapsed {dur(c['elapsed_s'])}   median question {dur(c['median_question_s'])}   ETA this evaluation {dur(c['eta_s'])}"
            f"   overall ETA {dur(d['overall_eta_s'])}")
        add(f"  Go list cost so far this evaluation ${c['go_list_cost_so_far']:.4f}")
    else:
        add("Questions    (no session yet)")
        if d["overall_eta_s"] is not None:
            add(f"  overall ETA {dur(d['overall_eta_s'])}")
    add()
    add("Completed evaluations", "bold")
    add(f"  {'#':>2} {'acc%':>5} {'aa$':>8} {'go_list$':>8} {'go_exp$':>8} {'wall':>7}  config (manager / search / docs)")
    for e in d["evals"]:
        o = e["options"] or {}
        cfg = " / ".join(short(o.get(k)) for k in ("manager", "search_agent", "docs")) if o else "?"
        f = lambda v: "   n/a  " if v is None else f"{float(v):8.4f}"
        add(f"  {e['index']!s:>2} {float(e['accuracy'] or 0):5.1f} {f(e['aa_cost'])} {f(e['go_list_cost'])} {f(e['go_expected'])} {dur(e['wall_time_s']):>7}  {cfg}"
            + ("  MISMATCH" if e.get("mismatch") else ""), "red" if e.get("mismatch") else None)
    if not d["evals"]:
        add("  (none yet)", "dim")
    t = d["totals"]
    add()
    add(f"Totals (completed): AA cost ${t['aa_cost']:.4f}   Go expected ${t['go_expected']:.4f}   Go list ${t['go_list_cost']:.4f}   "
        f"Go allowance ${t['go_allowance']:.4f}   " + ("MISMATCH expected vs list >2%" if t["mismatch"] else "expected vs list consistent"),
        "red" if t["mismatch"] else "green")
    if t["go_budget"] is not None:
        add(f"  Go list incl. current ${t['go_list_cost_with_current']:.4f} of budget ${t['go_budget']:.2f} (left ${t['go_budget_left']:.4f})",
            "red" if t["go_budget_left"] < 0 else None)
    a = d["activity"]
    add(f"Last {a['window_s'] // 60} min: LLM calls {a['calls']}   retries {a['retries']}   HTTP 429 {a['http_429']}   errors {a['errors']}"
        f"      (this evaluation: calls {a['total_calls']}, retries {a['total_retries']}, 429 {a['total_429']}, errors {a['total_errors']};"
        f" last call {dur(a['last_call_age_s'])} ago)", "yellow" if (a["http_429"] or a["errors"]) else None)
    if not d["finished"] and a["last_call_age_s"] is not None and a["last_call_age_s"] > 300 and c and c["in_flight"]:
        add("!! no LLM call for >5 min while an evaluation is in flight: stalled?", "bold red")
    return L


def show(d, rich_console, clear):
    L = render(d)
    if clear:
        sys.stdout.write("\033[H\033[2J")
    if rich_console:
        for st, t in L:
            rich_console.print(t, style=st or "", highlight=False, markup=False, soft_wrap=True)
    else:
        for _, t in L:
            print(t)
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--watch", type=float, default=0, metavar="N", help="refresh every N seconds (Ctrl-C to stop)")
    ap.add_argument("--json", action="store_true", help="one-shot machine-readable snapshot")
    ap.add_argument("--no-rich", action="store_true")
    ap.add_argument("--go-budget", type=float, default=None, help="Go list-cost budget in USD to show the remainder")
    a = ap.parse_args()
    if not Path(a.output_dir).is_dir():
        sys.exit(f"not a directory: {a.output_dir}")
    if a.json:
        print(json.dumps(collect(a.output_dir, a.go_budget), indent=1, default=str))
        return
    console = None
    if not a.no_rich:
        try:
            from rich.console import Console
            console = Console()
        except Exception:
            pass
    try:
        while True:
            show(collect(a.output_dir, a.go_budget), console, clear=bool(a.watch))
            if not a.watch:
                break
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
