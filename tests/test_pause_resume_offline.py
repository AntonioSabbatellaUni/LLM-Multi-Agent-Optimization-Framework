"""Offline test of pause/resume (no API): python tests/test_pause_resume_offline.py   (cwd = malbo/, inside the container)
A) llm_logger: a quota 429 -> no retries, InfraAbort, abort.reason kind=quota; later calls stop at once
B) evaluate_configuration with stubbed questions: pause after some questions -> EvaluationPaused, completed questions in the question
   cache; second run only runs the rest and merges accuracy + cost of both sessions
C) MALBO evaluator: a quota pause defers every later point using that model, other points still run, nothing is logged for the
   paused point; a global pause defers everything
D) BoTorchOptimizer initial design: NaN points -> SystemExit(75); resume in the same output dir reuses the finished points"""
import os, sys, json, tempfile, types
from pathlib import Path
import numpy as np

HERE = os.getcwd()
EXEC = os.path.join(HERE, '../smolagents_opt/examples/open_deep_research')
sys.path.insert(0, HERE)

# ---------------- A) llm_logger + abort
sys.path.insert(0, EXEC)
os.chdir(EXEC)
from utils import llm_logger, abort
import litellm

class FakeRL(Exception):
    status_code = 429

calls = {"n": 0}
def fake_completion(*a, **k):
    calls["n"] += 1
    raise FakeRL("Monthly usage limit exceeded for this model")
litellm.completion = fake_completion
llm_logger._state["installed"] = False
tmp = Path(tempfile.mkdtemp())
llm_logger.set_log_path(tmp / "llm_calls.jsonl")
llm_logger.install()
abort.reset()
llm_logger.set_task("t-A")
try:
    litellm.completion(model="openai/grok-4.7", messages=[{"role": "user", "content": "x"}]); raise SystemExit("expected InfraAbort")
except abort.InfraAbort:
    pass
assert calls["n"] == 1, calls                      # quota text -> no retry
r = abort.reason(); assert r["kind"] == "quota" and r["model"] == "openai/grok-4.7" and "t-A" in abort.failed_tasks(), r
llm_logger.set_task("t-B")
try:
    litellm.completion(model="openai/glm-5.3", messages=[]); raise SystemExit("expected InfraAbort")
except abort.InfraAbort:
    pass
assert calls["n"] == 1 and "t-B" in abort.failed_tasks()   # paused: the call is not even sent
print("A ok: quota -> pause without retries, later calls stop at once")

# ---------------- B) evaluate_configuration pause + resume via question cache
import optimization_interface as oi
N = 8
ds = [{"task_id": f"q{i}", "question": f"Q{i}", "true_answer": "a", "task": 1} for i in range(N)]
oi.load_gaia_dataset = lambda **k: ds
oi.load_agent_models = lambda cfg, quiet=False: {}
os.environ.pop("MALBO_TASK_IDS_FILE", None)
oi.create_deterministic_subset = lambda eval_ds, limits, seed: list(eval_ds)
qdir = Path(tempfile.mkdtemp()); os.environ["MALBO_QCACHE_DIR"] = str(qdir)
POOL = {"name": "m@x", "aa_in": 1.0, "aa_cache": 0.1, "aa_out": 2.0, "go_in": 1.0, "go_cache": 0.1, "go_out": 2.0, "go_peak": 1}
cfg = {"manager": {"model_id": "openai/m", "reasoning_effort": "max", "pool": POOL}}
plan = {"fail_from": 5}

class FakeRM:
    def setup_session(self, kind, name):
        d = Path(tempfile.mkdtemp()); llm_logger.set_log_path(d / "llm_calls.jsonl"); return d
    def start_run(self): pass
oi.EnhancedRunManager = FakeRM

def fake_answer(example, models, rm, retry_config, i, models_factory=None, parallel=False):
    tid = example["task_id"]
    if plan["fail_from"] is not None and i >= plan["fail_from"]:
        abort.trigger("llm", "openai/m", FakeRL("usage limit exceeded"), task_id=tid)
        return {"question_index": i, "task_id": tid, "is_correct": False, "prediction": None, "task_level": 1,
                "agent_error": "paused: usage limit", "question": "", "true_answer": "a"}
    llm_logger._write({"task_id": tid, "agent": "manager", "ts_start": "2026-11-14T12:00:00+00:00", "error": None,
                       "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "cached_tokens": 0}})
    return {"question_index": i, "task_id": tid, "is_correct": i % 2 == 0, "prediction": "a", "task_level": 1,
            "agent_error": None, "question": "", "true_answer": "a", "parsing_error": False, "iteration_limit_exceeded": False}
oi.answer_single_question_with_tracking = fake_answer
oi.calculate_final_metrics = lambda res: {"overall_accuracy": 100.0 * sum(r["is_correct"] for r in res) / len(res), "n": len(res)}

for par in (1, 4):
    for f in qdir.glob("*"): f.unlink()
    plan["fail_from"] = 5
    try:
        oi.evaluate_configuration(cfg, run_name="t", concurrency=par); raise SystemExit("expected EvaluationPaused")
    except abort.EvaluationPaused as e:
        assert e.n_done == 5 and e.n_total == N and e.reason["kind"] == "quota", (e.n_done, e.reason)
        assert json.loads((Path(e.session_dir) / "paused_report.json").read_text())["kept"] == 5
    cache = list(qdir.glob("*.jsonl")); assert len(cache) == 1 and len(cache[0].read_text().splitlines()) == 5
    plan["fail_from"] = None
    acc, cost, sd = oi.evaluate_configuration(cfg, run_name="t", concurrency=par)
    assert abs(acc - 50.0) < 1e-9, acc                           # q0,q2,q4 (cached) + q6 (new) correct of 8
    assert abs(cost - 8.0) < 1e-9, cost                          # 8 questions x 1M input x $1 (5 cached + 3 new)
    rep = json.loads((Path(sd) / "cost_report.json").read_text())
    assert rep["n_questions_cached"] == 5 and rep["n_calls"] == 8, rep
    print(f"B ok (parallel={par}): paused at 5/8, resumed -> acc {acc:.1f}% cost ${cost:.2f} over 8 questions")
# fake (0,0) on a crash is gone
oi.load_gaia_dataset = lambda **k: (_ for _ in ()).throw(RuntimeError("dataset down"))
try:
    oi.evaluate_configuration(cfg, run_name="t"); raise SystemExit("expected RuntimeError")
except RuntimeError:
    pass
print("B ok: a crash raises instead of returning (0 %, $0)")
os.chdir(HERE); os.environ.pop("MALBO_QCACHE_DIR", None)

# ---------------- C) MALBO evaluator pause handling
import yaml
from data_processor_v2 import load_llm_data, load_pool_df
from multi_agent_architectures.gaia_smolagents.evaluator import create_gaia_evaluator, effective_slots, EvaluationPaused
main = yaml.safe_load(open('config.go3slots.yaml'))
acp = main['evaluation']['architecture_config']; arch = yaml.safe_load(open(acp))
feats, csv = main['features'], main['data']['input_file']
full, pool = load_pool_df(csv, None, False), load_pool_df(csv, feats, True)
data, names, fn = load_llm_data(csv, feats, df=pool)

def make_ev(outdir):
    ev = create_gaia_evaluator(data, names, fn, len(effective_slots(arch)), config_path=acp, pool_df=pool, pool_df_full=full)
    ev.output_dir = outdir
    ev.executor_available = True
    return ev

def X(cfgs):
    return np.array([np.concatenate([data[names.index(c[s])] for s in ('manager', 'search_agent', 'docs')]) for c in cfgs])

pts = [{'manager': 'grok-4.7@xhigh', 'search_agent': 'gpt-6-luna@low', 'docs': 'glm-5.3-flash@max'},
       {'manager': 'glm-5.3@low', 'search_agent': 'grok-4.7@low', 'docs': 'mimo-v2.6-flash'},       # same go model -> deferred
       {'manager': 'claude-haiku-5-5@max', 'search_agent': 'gpt-6-luna@low', 'docs': 'glm-5.3-flash@max'}]
out = tempfile.mkdtemp(); ev = make_ev(out); seen = []
def fake_exec(config_dict, i):
    seen.append(config_dict['manager']['pool']['name'])
    if config_dict['manager']['model_id'] == 'openai/grok-4.7':
        raise EvaluationPaused({'kind': 'quota', 'model': 'openai/grok-4.7', 'error': 'usage limit exceeded'})
    ev.last_cost_report = {'aa_cost': 1.0, 'go_expected': 1, 'go_list_cost': 1, 'go_allowance': 1, 'mismatch': False}
    return 40.0, 1.0
ev._execute_real_benchmark = fake_exec
Y = ev.evaluate_agent_system(X(pts))
assert np.isnan(Y[0]).all() and np.isnan(Y[1]).all() and not np.isnan(Y[2]).any(), Y
assert seen == ['grok-4.7@xhigh', 'claude-haiku-5-5@max'], seen        # point 1 never started
recs = [json.loads(l) for l in open(os.path.join(out, 'evaluations.jsonl'))]
assert len(recs) == 1 and recs[0]['options']['manager'] == 'claude-haiku-5-5@max'
ev2 = make_ev(tempfile.mkdtemp()); seen.clear()
def fake_global(config_dict, i):
    seen.append(1); raise EvaluationPaused({'kind': 'infra', 'model': None, 'error': 'connection refused'})
ev2._execute_real_benchmark = fake_global
Y2 = ev2.evaluate_agent_system(X(pts)); assert np.isnan(Y2).all() and len(seen) == 1
print("C ok: quota pause defers only points with that model; global pause defers all; paused points not logged")

# ---------------- D) optimizer: exit 75 then resume in the same dir
import torch
from botorch_optimization import BoTorchOptimizer
out = tempfile.mkdtemp()
def run_design(fail_models):
    ev = make_ev(out)
    def fx(config_dict, i):
        if config_dict['manager']['model_id'] in fail_models:
            raise EvaluationPaused({'kind': 'quota', 'model': config_dict['manager']['model_id'], 'error': 'limit exceeded'})
        ev.last_cost_report = {'aa_cost': 0.5, 'go_expected': 1, 'go_list_cost': 1, 'go_allowance': 1, 'mismatch': False}
        return 50.0, 0.5
    ev._execute_real_benchmark = fx
    d = ev.n_agents * data.shape[1]
    opt = BoTorchOptimizer(ev, bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
                           ref_point=torch.tensor([-0.1, -50.0]).double(), output_dir=out, seed=42, initial_design='balanced')
    return opt._generate_initial_data(16)
try:
    run_design({'openai/grok-4.7', 'openai/kimi-k3.1-pro'}); raise SystemExit("expected exit 75")
except SystemExit as e:
    assert e.code == 75, e.code
done1 = [json.loads(l) for l in open(os.path.join(out, 'evaluations.jsonl'))]
prog = json.load(open(os.path.join(out, 'progress.json')))
assert prog['phase'] == 'paused' and len(prog['paused_points']) >= 2, prog
Xi, Yi = run_design(set())                                      # limits reset: resume
assert not torch.isnan(Yi).any() and Yi.shape == (16, 2)
recs = [json.loads(l) for l in open(os.path.join(out, 'evaluations.jsonl'))]
new = [r for r in recs[len(done1):] if not r.get('reused')]
assert len(new) == len(prog['paused_points']), (len(new), prog['paused_points'])   # only the deferred points are evaluated
print(f"D ok: first run {len(done1)} done + {len(prog['paused_points'])} deferred -> exit 75; resume evaluated only those")
print("ALL PAUSE/RESUME TESTS PASSED")
