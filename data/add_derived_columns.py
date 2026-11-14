"""Add derived columns to the Go pool CSV (stdlib only): log_AA_CostPerTask = ln(AA_CostPerTask_USD),
AA_Blended_1M = (3*Costo_Input_1M + Costo_Output_1M)/4.  python data/add_derived_columns.py SRC DST"""
import csv, math, sys
src, dst = sys.argv[1:3]
rows = list(csv.DictReader(open(src, newline="")))
cols = list(rows[0].keys())
for c in ("log_AA_CostPerTask", "AA_Blended_1M"):
    if c not in cols:
        cols.append(c)
for r in rows:
    r["log_AA_CostPerTask"] = repr(math.log(float(r["AA_CostPerTask_USD"])))
    r["AA_Blended_1M"] = repr(round((3 * float(r["Costo_Input_1M"]) + float(r["Costo_Output_1M"])) / 4, 10))
w = csv.DictWriter(open(dst, "w", newline=""), fieldnames=cols)
w.writeheader(); w.writerows(rows)
