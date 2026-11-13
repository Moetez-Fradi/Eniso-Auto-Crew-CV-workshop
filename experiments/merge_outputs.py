"""Copy outputs from an executed notebook into a freshly built one, so text-only edits need no re-run.

Code cells are matched in order; a code cell whose source changed beyond comments is reported (its output may be stale).
usage: python experiments/merge_outputs.py [built.ipynb] [executed.ipynb]
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
built = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cv_production_workshop.ipynb"
execd = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "cv_production_workshop_executed.ipynb"


def code_only(src):
    lines = [re.sub(r"\s+#.*$", "", l).rstrip() for l in "".join(src).splitlines()]
    return [l for l in lines if l.strip() and not l.lstrip().startswith("#")]


nb_new = json.load(open(built, encoding="utf8"))
nb_old = json.load(open(execd, encoding="utf8"))
new_code = [c for c in nb_new["cells"] if c["cell_type"] == "code"]
old_code = [c for c in nb_old["cells"] if c["cell_type"] == "code"]
assert len(new_code) == len(old_code), f"code cell count differs: {len(new_code)} vs {len(old_code)}; re-run the notebook"
stale = 0
for i, (n, o) in enumerate(zip(new_code, old_code)):
    if code_only(n["source"]) != code_only(o["source"]):
        stale += 1; print(f"code cell {i} changed beyond comments: output may be stale")
    n["outputs"], n["execution_count"] = o.get("outputs", []), o.get("execution_count")
nb_new["metadata"] = {**nb_old.get("metadata", {}), **nb_new.get("metadata", {})}
json.dump(nb_new, open(execd, "w", encoding="utf8"), indent=1, ensure_ascii=False)
print(f"merged outputs of {len(new_code)} code cells into {execd.name}; {stale} possibly stale")
