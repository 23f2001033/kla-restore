"""Build the self-contained Kaggle notebook.

Embeds the verified source tree as a base64 zip so there is no external code
dataset that can go stale -- the cause of several failed launch attempts.
Regenerate with `python make_notebook.py` after changing anything under src/.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [
    "src/__init__.py", "src/data.py", "src/degrade.py", "src/model.py",
    "src/losses.py", "src/metrics.py", "train.py", "evaluate.py",
    "inference.py", "configs/base.yaml", "make_presentation.py",
    "analyze_degradation.py",
]


def payload() -> list[str]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(ROOT / f, f)
    b64 = base64.b64encode(buf.getvalue()).decode()
    chunk = 4000
    parts = [b64[i:i + chunk] for i in range(0, len(b64), chunk)]
    return ["_CODE_B64 = (\n"] + [f'    "{p}"\n' for p in parts] + [")\n"]


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


CELL1 = '''
# --- Cell 1: write the verified source tree into the session ---------------
import base64, io, os, zipfile, sys, shutil

WORK = "/kaggle/working/kla-restore"
if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_CODE_B64))) as z:
    z.extractall(WORK)
os.chdir(WORK)
sys.path.insert(0, WORK)

train_src = open("train.py").read()
model_src = open("src/model.py").read()
cfg_txt = open("configs/base.yaml").read()
lr_line = [l.strip() for l in cfg_txt.splitlines() if l.strip().startswith("lr:")]
checks = {
    "AdamW betas=(0.9, 0.999)":   "betas=(0.9, 0.999)" in train_src,
    "non-finite LOSS guard":      "math.isfinite(lv)" in train_src,
    "non-finite GRADIENT guard":  "isfinite(gnorm)" in train_src,
    "--no_amp escape hatch":      "--no_amp" in train_src,
    "LayerNorm2d forced to fp32": "enabled=False" in model_src,
    "lr = 5.0e-4":                lr_line == ["lr: 5.0e-4"],
    "pack_dataset present":       "def pack_dataset" in open("src/data.py").read(),
}
for k, v in checks.items():
    print(("  OK  " if v else "  FAIL") + "  " + k)
assert all(checks.values()), "embedded code is not the fixed version"
'''

CELL2 = '''# --- Cell 2: environment + data ------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # single GPU on purpose

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU. Settings -> Accelerator -> GPU."
print(torch.cuda.get_device_properties(0).name)

from pathlib import Path
def find_dataset(name):
    root = Path("/kaggle/input")
    if (root / name).is_dir():
        return root / name
    for pat in ("*/" + name, "datasets/*/" + name):
        m = [p for p in root.glob(pat) if p.is_dir()]
        if m:
            return m[0]
    present = [p.name for p in root.iterdir()] if root.is_dir() else "none"
    raise FileNotFoundError("dataset '%s' not attached. Present: %s" % (name, present))

DATA_ROOT = find_dataset("kla-semicon-2026")
DATA_SOURCE = DATA_ROOT / "train.zip" if (DATA_ROOT / "train.zip").is_file() else DATA_ROOT
print("data source:", DATA_SOURCE)

import subprocess
subprocess.run(["pip", "-q", "install", "lpips"], check=False)

from src.data import pack_dataset
meta = pack_dataset(DATA_SOURCE, "/kaggle/working/packed")
'''

CELL3 = '''# --- Cell 3: pre-flight gate, with automatic fp32 fallback ----------------
# Two earlier runs died to fp16 overflow inside the normalisation layers: a
# mean of x^2 overflows fp16 once activations reach only ~100, far below
# fp16's 65504 ceiling. That path is now forced to fp32, but the gate proves
# it empirically instead of trusting it -- and falls back to full fp32 if any
# other overflow path remains. Cost of a numerical failure: ~2 minutes.
import subprocess

BASE = ["python", "train.py", "--overfit", "2", "--steps", "2000",
        "--gate_db", "35", "--num_workers", "2", "--no_lpips",
        "--data_dir", "/kaggle/working/packed"]

AMP_FLAGS = []          # decided here, reused by the training cell
print("=" * 62)
print("GATE ATTEMPT 1: mixed precision")
print("=" * 62)
r = subprocess.run(BASE)

if r.returncode != 0:
    print("")
    print("=" * 62)
    print("fp16 gate failed -> retrying in fp32 (slower, no overflow path)")
    print("=" * 62)
    AMP_FLAGS = ["--no_amp"]
    r = subprocess.run(BASE + AMP_FLAGS)

assert r.returncode == 0, "GATE FAILED in both fp16 and fp32 -- do not train"
print("")
print("GATE PASSED -- training will use:", "fp32" if AMP_FLAGS else "mixed precision")
'''

CELL4 = '''# --- Cell 4: training (~6.5 h) -------------------------------------------
# Uses whichever precision passed the gate. Watch the loss over the first few
# hundred steps: it should fall steadily. Any non-finite loss or gradient is
# skipped rather than applied, and the run aborts after 20 consecutive bad
# updates instead of silently burning the budget.
import subprocess
r = subprocess.run(
    ["python", "train.py", "--config", "configs/base.yaml",
     "--data_dir", "/kaggle/working/packed",
     "--out_dir", "/kaggle/working/weights",
     "--hours", "6.5", "--num_workers", "2"] + AMP_FLAGS)
print("training exit code:", r.returncode)
assert r.returncode == 0, "training failed -- see the log above"
'''

CELL5 = '''# --- Cell 5: evaluate -----------------------------------------------------
import os, subprocess
assert os.path.exists("/kaggle/working/weights/best.pt"), \\
    "no best.pt -- training did not reach its first validation (step 5000)"
subprocess.run(["python", "evaluate.py",
                "--weights", "/kaggle/working/weights/best.pt",
                "--data_dir", "/kaggle/working/packed",
                "--out_dir", "/kaggle/working/results"], check=True)
'''

CELL6 = '''# --- Cell 6: stage the validation split for end-to-end timing -------------
import json, numpy as np, os, sys
sys.path.insert(0, "/kaggle/working/kla-restore")
from src.data import split_indices
meta = json.load(open("/kaggle/working/packed/meta.json"))
_, val_idx = split_indices(meta["n"])
lr = np.load("/kaggle/working/packed/lr.npy", mmap_mode="r")
os.makedirs("/kaggle/working/bench_in", exist_ok=True)
for i in val_idx:
    np.save("/kaggle/working/bench_in/%06d.npy" % i, np.asarray(lr[i]))
print(len(val_idx), "files staged")
'''

CELL7 = '''# Timed the way KLA times it: process start through last file written.
import time, subprocess
t0 = time.perf_counter()
subprocess.run(["python", "inference.py",
                "--input_dir", "/kaggle/working/bench_in",
                "--output_dir", "/kaggle/working/bench_out",
                "--weights", "/kaggle/working/weights/best.pt"], check=True)
print("wall clock, full pipeline: %.2f s" % (time.perf_counter() - t0))
'''

CELL8 = '''# --- Cell 8: regenerate the deck with real numbers, bundle artifacts ------
import glob, shutil, subprocess
shutil.copytree("/kaggle/working/results", "results", dirs_exist_ok=True)
subprocess.run(["python", "make_presentation.py"], check=True)
shutil.copy("solution_presentation.pptx", "/kaggle/working/")

print("weights:", glob.glob("/kaggle/working/weights/*"))
print("results:", glob.glob("/kaggle/working/results/*"))
subprocess.run(
    "cd /kaggle/working && zip -r submission_artifacts.zip weights results "
    "solution_presentation.pptx -x '*.pyc'", shell=True)
print("")
print("download: /kaggle/working/submission_artifacts.zip")
'''

HEADER = """# KLA SEMICON 2026 — restoration training

**Self-contained.** All source is embedded below, so there is no code dataset
that can go stale.

**Setup:** attach the official data as an Input named `kla-semicon-2026`
(zip *or* already-extracted — both work), set Accelerator to **GPU**, Internet
**On**. Then **Run All**.

Cell 3 is a hard gate. It first tries mixed precision; if that diverges it
automatically retries in fp32 and trains with whichever mode passed — so a
numerical failure costs about two minutes rather than a wasted night."""


def main() -> None:
    cells = [md(HEADER), code(payload() + [CELL1])]
    cells += [code(c) for c in (CELL2, CELL3, CELL4, CELL5, CELL6, CELL7, CELL8)]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "kaggle_train_standalone.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"[nb] wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
