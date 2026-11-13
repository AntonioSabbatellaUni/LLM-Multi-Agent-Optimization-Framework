"""Real smoke evaluation (API calls, ~cents): one hand-picked config on 2 GAIA tasks (text-only 11af4e1a + xlsx 5cfb274c).
Needs env OPENCODE_API_KEY (via --env-file), MALBO_API_BASE, MALBO_MANAGER_MAX_STEPS, MALBO_SEARCH_MAX_STEPS. cwd = malbo/."""
import os, sys
import numpy as np, yaml
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MALBO_TASK_IDS_FILE', os.path.abspath('tests/go3slots_smoke_2tasks.json'))
from data_processor_v2 import load_llm_data, load_pool_df
from multi_agent_architectures.gaia_smolagents.evaluator import create_gaia_evaluator, effective_slots
main = yaml.safe_load(open('config.go3slots.yaml')); acp = main['evaluation']['architecture_config']
arch = yaml.safe_load(open(acp)); feats, csv = main['features'], main['data']['input_file']
full, pool = load_pool_df(csv, None, False), load_pool_df(csv, feats, True)
data, names, fn = load_llm_data(csv, feats, df=pool)
out = os.getenv('SMOKE_OUT', 'outputs/smoke_go3slots')
ev = create_gaia_evaluator(data, names, fn, 3, config_path=acp, pool_df=pool, pool_df_full=full, output_dir=out)
pick = os.getenv('SMOKE_PICK', 'deepseek-v4.1-flash@max,gpt-6-luna@low,glm-5.3-flash@max').split(',')
X = np.concatenate([data[names.index(n)] for n in pick])[None, :]
print('Y =', ev.evaluate_agent_system(X))
