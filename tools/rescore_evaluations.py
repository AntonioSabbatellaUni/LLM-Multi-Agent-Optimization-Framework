"""Rescore finished evaluations offline (no API calls) with the relaxed GAIA scorer and write them as a new evaluations.jsonl,
to warm-start a run that uses MALBO_SCORER=relaxed. The source files are only read.

Per-question answers come from <src>/qcache (config.json + jsonl), or, for a record with `derived_from`, from that
evaluation's session questions.jsonl restricted to the task ids file. A question with no saved answer (timeout, error)
counts as wrong, as in the original score. The exact-match score is recomputed too; the stored accuracy must equal the exact or the relaxed one.

  python tools/rescore_evaluations.py outputs/go3_check30_sub20 outputs/go3v3_cheap20 --out outputs/go3v3_cheap20_rescored
"""
import argparse, glob, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "smolagents_opt/examples/open_deep_research"))
from scripts.gaia_scorer import rescore  # noqa: E402

SUBSETS = Path(__file__).resolve().parents[1] / "multi_agent_architectures/gaia_smolagents"


def qcache_answers(src):
    """{(manager, search_agent, docs): {task_id: result}} from <src>/qcache."""
    out = {}
    for f in glob.glob(os.path.join(src, "qcache", "*.config.json")):
        c = json.load(open(f))
        key = (c["manager"][2], c["search_agent"][2], c["reformulator"][2])
        rows = {}
        for line in open(f.replace(".config.json", ".jsonl")):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["task_id"]] = r["result"]
        out[key] = rows
    return out


def derived_answers(rec):
    d = rec["derived_from"]
    ids = json.load(open(SUBSETS / d["task_ids_file"]))
    ids = {t["task_id"] if isinstance(t, dict) else t for t in (ids if isinstance(ids, list) else ids["tasks"])}
    cur = json.load(open(Path(d["evaluation"]) / "current_session.json"))["session_dir"].replace("/work/", str(ROOT) + "/")
    rows = {}
    for line in open(Path(cur) / "questions.jsonl"):
        r = json.loads(line)
        if r["task_id"] in ids:
            rows[r["task_id"]] = r
    return rows, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-reused", action="store_true", help="also rescore reused copies (e.g. the warm-started design of a run)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    ap.add_argument("--n", type=int, default=20, help="questions per evaluation (missing answers count as wrong)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "evaluations.jsonl")
    if os.path.exists(dst) and not a.force:
        sys.exit(f"{dst} exists: not overwriting")
    written, problems = [], 0
    for src in a.sources:
        answers = qcache_answers(src)
        for line in open(os.path.join(src, "evaluations.jsonl")):
            rec = json.loads(line)
            if rec.get("accuracy") is None or (rec.get("reused") and not a.include_reused):
                continue
            o = rec["options"]
            if rec.get("derived_from"):
                rows, n = derived_answers(rec)
            else:
                rows, n = answers.get((o["manager"], o["search_agent"], o["docs"])), a.n
            if rows is None:
                print(f"  ⚠️ no per-question answers for {o} in {src}: skipped"); problems += 1
                continue
            rs = [rescore(r) for r in rows.values()]
            exact, relaxed, first = (100 * sum(r[k] for r in rs) / n for k in ("is_correct_exact", "is_correct", "is_correct_first_line"))
            if min(abs(exact - rec["accuracy"]), abs(relaxed - rec["accuracy"])) > 1e-6:   # stored by an exact or a relaxed run
                print(f"  ⚠️ stored {rec['accuracy']:.1f} != exact {exact:.1f} / relaxed {relaxed:.1f} for {o}"); problems += 1
            out = {k: v for k, v in rec.items()}
            out.update(accuracy=relaxed, accuracy_exact=exact, accuracy_stored=rec["accuracy"], accuracy_first_line=first, scorer="relaxed_v1",
                       rescored_from=os.path.join(src, "evaluations.jsonl"), n_answers_found=len(rows))
            written.append(out)
            print(f"  {o['manager']:32s} {o['search_agent']:32s} {o['docs']:32s} exact {rec['accuracy']:5.1f} -> relaxed {relaxed:5.1f}  (first line {first:5.1f})"
                  + (f"  it{rec['bo_iteration']}" if rec.get("bo_iteration") else ""))
    with open(dst, "w") as f:
        for r in written:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"{len(written)} records -> {dst}   problems: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
