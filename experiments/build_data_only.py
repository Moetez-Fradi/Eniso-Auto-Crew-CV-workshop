"""Run the notebook source up to (and including) the data-preparation cell of step 2.1, so training scripts use exactly the notebook's dataset.
usage: WS_DATA_SOURCE=local WS_DATA_A=data WS_DATA_B=data_sss python experiments/build_data_only.py
"""
import os, sys, builtins
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg")
import build_notebook as bn
builtins.display = print
ns = {"__name__": "__main__"}
for f in sorted(bn.SRC.glob("*.py")):
    for kind, src in bn.parse(f.read_text(encoding="utf-8")):
        if kind != "code":
            continue
        exec(compile(src, f.name, "exec"), ns)
        if "DATA_YAML_A" in ns:
            print("data ready:", ns["DATA_YAML"]); sys.exit(0)
