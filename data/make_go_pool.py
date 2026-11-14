"""Build a MALBO model-pool CSV for OpenCode Go with a `supports_image` column.

supports_image = 1 if the Go /models entry has an input modality starting with "image" (modalities.input).
Benchmark/cost columns are copied from --base (rows matched on the id after the last "/") when present, otherwise
filled with --placeholder (smoke-test only: replace with real numbers before a real sweep).

  python data/make_go_pool.py --ids deepseek-v4.1-flash iceman-contributor-free --out data/llm_data_go_smoke.csv
  python data/make_go_pool.py --all --base data/llm_data.csv --out data/llm_data_go.csv   # whole Go catalogue
"""
import argparse, json, urllib.request
import pandas as pd

URL = "https://opencode.ai/zen/go/v1/models"
ap = argparse.ArgumentParser()
ap.add_argument("--ids", nargs="*", default=[])
ap.add_argument("--all", action="store_true")
ap.add_argument("--base", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", default="openai/")
ap.add_argument("--placeholder", type=float, default=50.0)
ap.add_argument("--in-cost", type=float, default=0.1)
ap.add_argument("--out-cost", type=float, default=0.3)
a = ap.parse_args()

meta = {m["id"]: m for m in json.load(urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (malbo)"}), timeout=30))["data"]}
ids = list(meta) if a.all else a.ids
missing = [i for i in ids if i not in meta]
if missing:
    raise SystemExit(f"not in Go /models: {missing}")
cols = ["Modello", "MMLU", "HellaSwag", "HumanEval", "GSM8K", "GPQA_Diamond", "AIME_2025", "ARC", "TruthfulQA", "AGIEval",
        "BBH", "MATH", "MT_bench", "Media_Totale", "Costo_Input_1M", "Costo_Output_1M", "Costo_Blended_1M",
        "Context_Window", "Fonti"]
base = pd.read_csv(a.base) if a.base else None
rows = []
for i in ids:
    r = {c: a.placeholder for c in cols[1:13]}
    r.update(Media_Totale=a.placeholder, Costo_Input_1M=a.in_cost, Costo_Output_1M=a.out_cost,
             Costo_Blended_1M=(a.in_cost + a.out_cost) / 2, Context_Window="128K", Fonti="placeholder (Go smoke pool)")
    if base is not None:
        hit = base[base["Modello"].str.split("/").str[-1] == i]
        if len(hit):
            r.update({c: hit.iloc[0][c] for c in cols[1:] if c in hit.columns})
    r["Modello"] = a.prefix + i
    r["supports_image"] = int(any(str(m).startswith("image") for m in meta[i]["modalities"]["input"]))
    rows.append(r)
pd.DataFrame(rows)[cols + ["supports_image"]].to_csv(a.out, index=False)
print(pd.DataFrame(rows)[["Modello", "supports_image"]])
