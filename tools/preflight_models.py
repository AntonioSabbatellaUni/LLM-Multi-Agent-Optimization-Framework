"""Preflight: one short real call per pool option (model@effort) with exactly the parameters the agents send, so model/API
incompatibilities (e.g. `stop` rejected, effort value refused, tool calling broken) show up before an evaluation, not in it.
Run inside the container, cwd = malbo/:
    python tools/preflight_models.py [--config config.go3slots.yaml] [--options a,b] [--parallel 8] [--out outputs/preflight_<ts>]
Per option, through the real path (evaluator config -> loader.load_agent_models -> smolagents LiteLLMModel -> llm_logger wrapper):
  code : manager-style call (CodeAgent stop sequences), must return non-empty text
  tool : search_agent-style call (ToolCallingAgent: tools + stop sequences), must return a tool call
Writes <out>/preflight.json + llm_calls.jsonl; exit 0 if all OK, 1 otherwise."""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, os.getcwd())
from data_processor_v2 import load_llm_data, load_pool_df
from multi_agent_architectures.gaia_smolagents.evaluator import create_gaia_evaluator, effective_slots, isolated_executor_env

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.go3slots.yaml")
ap.add_argument("--options", default="")
ap.add_argument("--parallel", type=int, default=8)
ap.add_argument("--out", default=f"outputs/preflight_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
a = ap.parse_args()
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)

main = yaml.safe_load(open(a.config))
acp = main['evaluation']['architecture_config']; arch = yaml.safe_load(open(acp))
feats, csv = main['features'], main['data']['input_file']
full, pool = load_pool_df(csv, None, False), load_pool_df(csv, feats, True)
data, names, fn = load_llm_data(csv, feats, df=pool)
ev = create_gaia_evaluator(data, names, fn, len(effective_slots(arch)), config_path=acp, pool_df=pool, pool_df_full=full)
opts = [o for o in a.options.split(",") if o] or list(names)
nf = data.shape[1]

configs = {}
for o in opts:
    x = np.concatenate([data[names.index(o)]] * ev.n_agents)
    cfg = ev._complete_agent_configuration(ev._project_to_llm_configuration(x, nf))
    assert cfg['manager']['pool']['name'] == o, (o, cfg['manager']['pool']['name'])
    configs[o] = cfg

PROMPT_CODE = "Reply with the single word: ok"   # (must not contain a stop marker: models echo it in their reasoning and stop there)
PROMPT_TOOL = "Call the final_answer tool with the answer \"ok\". Do not write anything else."
results = {}

with isolated_executor_env(ev.executor_root_path):
    from loader import load_agent_models
    from utils import llm_logger, abort
    from smolagents.models import ChatMessage
    from smolagents.default_tools import FinalAnswerTool
    abort.check = lambda *x, **k: None          # preflight: one failing model must not stop the others
    llm_logger.set_log_path(out / "llm_calls.jsonl"); llm_logger.install()

    def msg(t):
        return [ChatMessage(role="user", content=[{"type": "text", "text": t}])]

    def one(o):
        cfg = configs[o]
        r = {"option": o, "model_id": cfg['manager']['model_id'], "effort": cfg['manager'].get('reasoning_effort')}
        try:
            models = load_agent_models({"manager": cfg["manager"], "search_agent": cfg["search_agent"]}, quiet=True)
        except Exception as e:
            r.update(ok=False, error=f"load: {type(e).__name__}: {e}"[:400]); return r
        for kind, role, call in (
                ("code", "manager", lambda m: m.generate(msg(PROMPT_CODE), stop_sequences=["<end_code>", "Observation:", "Calling tools:"])),
                ("tool", "search_agent", lambda m: m.generate(msg(PROMPT_TOOL), stop_sequences=["Observation:", "Calling tools:"],
                                                              tools_to_call_from=[FinalAnswerTool()]))):
            llm_logger.set_agent(role); llm_logger.set_task(f"preflight:{o}:{kind}")
            t0 = time.time()
            try:
                resp = call(models[role])
                content = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)[:300]
                tcs = resp.tool_calls or []
                if kind == "tool" and not tcs:
                    # smolagents can also parse a tool call out of the text (ToolCallingAgent fallback)
                    try:
                        resp = models[role].parse_tool_calls(resp); tcs = resp.tool_calls or []
                    except Exception:
                        pass
                good = bool(tcs) if kind == "tool" else bool((content or "").strip())
                r[kind] = {"ok": good, "latency_s": round(time.time() - t0, 1), "content": (content or "")[:160],
                           "tool_calls": [getattr(getattr(t, "function", None), "name", None) for t in tcs]}
            except Exception as e:
                r[kind] = {"ok": False, "latency_s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"[:400]}
        r["ok"] = all(r.get(k, {}).get("ok") for k in ("code", "tool"))
        return r

    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        for r in ex.map(one, opts):
            results[r["option"]] = r
            flag = "OK  " if r.get("ok") else "FAIL"
            det = "; ".join(f"{k}: " + ("ok" if r.get(k, {}).get("ok") else (r.get(k, {}).get("error") or f"bad output {r.get(k, {}).get('content', '')[:80]!r}"))
                            for k in ("code", "tool") if k in r) or r.get("error", "")
            print(f"{flag} {r['option']:<34} {det}"[:300], flush=True)

# cost of the preflight itself (Go list cost reported by the API)
cost = 0.0
for line in (out / "llm_calls.jsonl").read_text().splitlines() if (out / "llm_calls.jsonl").exists() else []:
    u = (json.loads(line).get("usage") or {}).get("raw") or {}
    cost += float((u.get("opencode") or {}).get("list_cost") or 0)
learned_no_stop = sorted(llm_logger._NO_STOP)
summary = {"n": len(results), "ok": sum(1 for r in results.values() if r.get("ok")), "go_list_cost": round(cost, 4),
           "models_without_stop": learned_no_stop, "models_tool_choice_auto": sorted(llm_logger._AUTO_TOOL_CHOICE), "results": results}
(out / "preflight.json").write_text(json.dumps(summary, indent=1, default=str))
print(f"\n{summary['ok']}/{summary['n']} options OK   preflight Go list cost ${cost:.4f}   models that reject `stop` (auto-dropped): {learned_no_stop}   tool_choice->auto: {sorted(llm_logger._AUTO_TOOL_CHOICE)}")
print(f"report: {out / 'preflight.json'}")
sys.exit(0 if summary["ok"] == summary["n"] else 1)
