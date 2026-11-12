#!/usr/bin/env python3
"""Build the fixed GAIA validation subset used by MALBO (30 tasks, seed 42, stratified by Level).

Needs: huggingface_hub (+ pandas only if the snapshot has parquet but no jsonl). Needs HF_TOKEN
(env var, or the line HF_TOKEN=... in ~/.config/malbo/.env) with access to the gated gaia-benchmark/GAIA.
Does NOT use datasets / GAIA.py (broken with datasets>=4). The real repo ships 2023/validation/metadata.parquet
(no .jsonl) -> parquet fallback (needs pandas + pyarrow).

Run:   python make_gaia_subset.py [--n 30] [--seed 42] [--local-dir ./data_gaia_snapshot]
Out:   gaia_subset_seed42_n30.json   (task_ids + level + has_file)
       gaia_subset_seed42_n30_report.md  (dropped tasks and why)
Use in the evaluator:  export MALBO_TASK_IDS_FILE=<path to gaia_subset_seed42_n30.json>
"""
import argparse, json, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "gaia-benchmark/GAIA"


def get_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    envf = Path.home() / ".config" / "malbo" / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit("HF_TOKEN not found (env var or ~/.config/malbo/.env)")


def load_rows(root: Path):
    j = root / "2023" / "validation" / "metadata.jsonl"
    if j.exists():
        return [json.loads(l) for l in j.read_text().splitlines() if l.strip()]
    p = root / "2023" / "validation" / "metadata.parquet"
    if p.exists():
        import pandas as pd
        return pd.read_parquet(p).to_dict("records")
    sys.exit(f"no metadata.jsonl / metadata.parquet under {root}/2023/validation")


def allocate(counts: dict, n: int):
    """Largest-remainder proportional allocation of n over the levels."""
    tot = sum(counts.values())
    raw = {k: n * v / tot for k, v in counts.items()}
    alloc = {k: int(r) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - alloc[k], -int(k)), reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-dir", default=str(HERE / "data_gaia_snapshot"))
    ap.add_argument("--excluded", default=str(HERE / "gaia_excluded_tasks.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=a.local_dir,
                                  allow_patterns=["2023/validation/*"], token=get_token()))
    rows = load_rows(root)
    val_dir = root / "2023" / "validation"
    excl = {t["task_id"]: t for t in json.load(open(a.excluded))["tasks"]}

    dropped, keep = [], []
    for r in rows:
        tid, lvl, fn = r["task_id"], str(r["Level"]), (r.get("file_name") or "")
        if tid in excl:
            dropped.append((tid, lvl, f"excluded: {excl[tid]['kind']} - {excl[tid]['reason']}")); continue
        if fn:
            f = val_dir / fn
            if not f.exists() or f.stat().st_size == 0:
                dropped.append((tid, lvl, f"attachment missing/empty: {fn}")); continue
        keep.append({"task_id": tid, "level": int(lvl), "has_file": bool(fn)})

    by = defaultdict(list)
    for k in keep:
        by[str(k["level"])].append(k)
    counts = {l: len(v) for l, v in by.items()}
    alloc = allocate(counts, a.n)
    rng = random.Random(a.seed)
    sel = []
    for l in sorted(by):
        pool = sorted(by[l], key=lambda k: k["task_id"])  # order-independent of snapshot row order
        sel += rng.sample(pool, alloc[l])
    sel.sort(key=lambda k: (k["level"], k["task_id"]))

    out = Path(a.out) if a.out else HERE / f"gaia_subset_seed{a.seed}_n{a.n}.json"
    json.dump({"seed": a.seed, "n": a.n, "split": "2023/validation", "n_total_rows": len(rows),
               "n_after_filters": len(keep), "remaining_by_level": counts, "allocation": alloc,
               "tasks": sel}, open(out, "w"), indent=2)
    rep = out.with_name(out.stem + "_report.md")
    lines = [f"# GAIA subset report (seed {a.seed}, n {a.n})", "",
             f"Rows in validation: {len(rows)}; dropped: {len(dropped)}; remaining: {len(keep)} {counts}",
             f"Allocation: {alloc}; with attachment in subset: {sum(s['has_file'] for s in sel)}", "",
             "## Dropped tasks", "", "| task_id | Level | reason |", "|---|---|---|"]
    lines += [f"| `{t}` | {l} | {r} |" for t, l, r in sorted(dropped, key=lambda x: (x[1], x[0]))]
    rep.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} and {rep}")


if __name__ == "__main__":
    main()
