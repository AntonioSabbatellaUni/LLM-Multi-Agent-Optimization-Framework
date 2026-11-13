"""Offline test (no API): python tests/test_go3slots_offline.py   (cwd = malbo/, run inside the container)
1) projection of random 15-d vectors -> per-role config (text_inspector == reformulator, efforts from the row, visual_qa fixed)
1b) balanced initial design (seed 42): deterministic, balanced per slot, visual_qa slot restricted to image models; random mode selectable
1c) llm_logger: thread-safe jsonl writes, retry backoff helper
2) cost function on an existing llm_calls.jsonl: go_list_cost reproduced, aa_cost ratio ~2.15x (deepseek)
   path: env LLM_CALLS (default /work/results/attachments_deepseek/llm_calls.jsonl)"""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), '../smolagents_opt/examples/open_deep_research'))
import yaml
from data_processor_v2 import load_llm_data, load_pool_df
from multi_agent_architectures.gaia_smolagents.evaluator import create_gaia_evaluator, effective_slots

main = yaml.safe_load(open('config.go3slots.yaml'))
acp = main['evaluation']['architecture_config']
arch = yaml.safe_load(open(acp))
feats, csv = main['features'], main['data']['input_file']
full, pool = load_pool_df(csv, None, False), load_pool_df(csv, feats, True)
assert len(full) == 22 and len(pool) == 21 and 'hy4@high' not in set(pool.Modello), (len(full), len(pool))
data, names, fn = load_llm_data(csv, feats, df=pool)
ev = create_gaia_evaluator(data, names, fn, len(effective_slots(arch)), config_path=acp, pool_df=pool, pool_df_full=full)
assert ev.n_agents == 3 and data.shape[1] == 5 and ev.n_agents * data.shape[1] == 15 and fn == ['AutomationBench', 'TerminalBench40', 'HLE', 'FrontierCode20', 'log_AA_CostPerTask']
rng = np.random.default_rng(0)
for k in range(8):
    cfg = ev._complete_agent_configuration(ev._project_to_llm_configuration(rng.random(15), 5))
    print(k, {r: (c['pool']['name'], c.get('reasoning_effort')) for r, c in ev._redact(cfg).items()})
    assert set(cfg) == {'manager', 'search_agent', 'text_inspector', 'reformulator', 'visual_qa'}
    assert cfg['text_inspector']['model_id'] == cfg['reformulator']['model_id'] and cfg['text_inspector']['pool'] == cfg['reformulator']['pool']
    for r in ('manager', 'search_agent', 'text_inspector'):
        row = pool[pool.Modello == cfg[r]['pool']['name']].iloc[0]
        assert cfg[r]['model_id'] == 'openai/' + row.go_model_id
        assert cfg[r]['reasoning_effort'] == ('' if row.reasoning_effort != row.reasoning_effort else row.reasoning_effort)
    assert cfg['visual_qa']['model_id'].endswith('deepseek-v4.1-flash') and cfg['visual_qa']['pool']['name'].startswith('deepseek-v4.1-flash')
# exact pool points project to themselves
x = np.concatenate([data[names.index('glm-5.3@low')], data[names.index('gpt-6-luna@low')], data[names.index('glm-5.3-flash@max')]])
cfg = ev._project_to_llm_configuration(x, 5)
assert [cfg[r]['pool']['name'] for r in ('manager', 'search_agent', 'text_inspector', 'reformulator')] == ['glm-5.3@low', 'gpt-6-luna@low'] + ['glm-5.3-flash@max'] * 2

# --- 1b) initial design
from botorch_optimization import design_pool_indices
n_init = 15 + 1
rows, counts = design_pool_indices(n_init, 3, len(names), 42, 'balanced', slot_names=ev.agent_roles)
rows2, _ = design_pool_indices(n_init, 3, len(names), 42, 'balanced', slot_names=ev.agent_roles)
rows3, _ = design_pool_indices(n_init, 3, len(names), 43, 'balanced', slot_names=ev.agent_roles)
assert rows == rows2 and rows != rows3 and len(rows) == 16
for sl, c in zip(ev.agent_roles, counts):
    print('balanced counts', sl, sorted(c.values(), reverse=True), 'options used', len(c), 'of', len(names))
    assert sum(c.values()) == n_init and max(c.values()) == 1 and len(c) == n_init   # 16 points < 21 options: every option at most once
assert len({tuple(r) for r in rows}) == 16 and len({r[0] for r in rows}) == 16 and len({r[1] for r in rows}) == 16
assert len({tuple(sorted(c)) for c in counts}) >= 2      # slots are permuted independently
big, bc = design_pool_indices(2 * len(names) + 5, 3, len(names), 42, 'balanced')
for c in bc:
    assert max(c.values()) - min(c.values()) <= 1 and len(c) == len(names)
img = np.array([(i % 3 == 0) for i in range(len(names))])
r4, c4 = design_pool_indices(20, 4, len(names), 42, 'balanced', slot_names=['manager', 'search_agent', 'docs', 'visual_qa'], supports_image=img)
assert all(img[r[3]] for r in r4) and max(c4[3].values()) - min(c4[3].values()) <= 1
rr, _ = design_pool_indices(16, 3, len(names), 42, 'random'); rr2, _ = design_pool_indices(16, 3, len(names), 42, 'random')
assert rr == rr2 and len(rr) == 16
print('first 3 design points:', [[names[i] for i in r] for r in rows[:3]])
# --- 1c) logger thread safety + backoff
import threading, json, tempfile
from utils import llm_logger, retry
d = tempfile.mkdtemp(); llm_logger.set_log_path(os.path.join(d, 'llm_calls.jsonl'))
def w(k):
    for j in range(50):
        llm_logger._write({'k': k, 'j': j, 'pad': 'x' * 5000})
ts_ = [threading.Thread(target=w, args=(k,)) for k in range(8)]
[t.start() for t in ts_]; [t.join() for t in ts_]
lines = open(os.path.join(d, 'llm_calls.jsonl')).read().splitlines()
assert len(lines) == 400 and all(json.loads(l)['pad'] for l in lines)
assert all(retry.llm_error_retryable(type('E', (Exception,), {'status_code': c})()) for c in (429, 500, 503)) and not retry.llm_error_retryable(type('E', (Exception,), {'status_code': 400})())
assert retry.search_error_retryable(Exception('https://html.duckduckgo.com/html 202 Ratelimit')) and not retry.search_error_retryable(Exception('No results found!'))
assert 1 <= retry.backoff_delay(1) <= 3.01 and retry.backoff_delay(10) <= 90
print('threaded logging + retry helpers OK')

from utils.pool_cost import evaluation_costs
p = full[full.go_model_id == 'deepseek-v4.1-flash'].iloc[0]
pr = ev._price_dict(p)
rep = evaluation_costs(os.getenv('LLM_CALLS', '/work/results/attachments_deepseek/llm_calls.jsonl'),
                       {r: pr for r in ('manager', 'search_agent', 'text_inspector', 'reformulator', 'visual_qa')})
print({k: v for k, v in rep.items() if k != 'per_role'})
assert abs(rep['go_list_cost'] - rep['go_expected']) < 1e-9 * max(1, rep['go_list_cost']) or not rep['mismatch'], rep
print('aa/go_list ratio', rep['aa_cost'] / rep['go_list_cost'])
print('OFFLINE TEST OK')
