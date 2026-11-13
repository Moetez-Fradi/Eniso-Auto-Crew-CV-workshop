"""Builds cv_production_workshop.ipynb from the percent-format files in notebook_src/.

Source format (jupytext-like):
    # %%                 -> starts a code cell
    # %% [markdown]      -> starts a markdown cell; every following line is "# text" (prefix stripped)

Usage:  python build_notebook.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "notebook_src"
OUT = HERE / "cv_production_workshop.ipynb"


def parse(text: str):
    cells, cur_kind, cur_lines = [], None, []

    def flush():
        if cur_kind is None:
            return
        lines = list(cur_lines)
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if cur_kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        cells.append((cur_kind, "\n".join(lines)))

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush(); cur_kind, cur_lines = "markdown", []
        elif line.startswith("# %%"):
            flush(); cur_kind, cur_lines = "code", []
        else:
            cur_lines.append(line)
    flush()
    return cells


def to_source(s: str):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build():
    cells = []
    for f in sorted(SRC.glob("*.py")):
        cells += parse(f.read_text(encoding="utf-8"))
    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        c = {"cell_type": kind, "metadata": {"id": f"c{i:03d}"}, "source": to_source(src)}
        if kind == "code":
            c["execution_count"] = None
            c["outputs"] = []
        nb_cells.append(c)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(nb_cells)} cells "
          f"({sum(k == 'code' for k, _ in cells)} code, {sum(k == 'markdown' for k, _ in cells)} markdown)")
    return cells


if __name__ == "__main__":
    cells = build()
    # syntax check every code cell
    import ast
    bad = 0
    for i, (kind, src) in enumerate(cells):
        if kind == "code":
            try:
                ast.parse(src)
            except SyntaxError as e:
                bad += 1
                print(f"SyntaxError in cell {i}: {e}")
    sys.exit(1 if bad else 0)
