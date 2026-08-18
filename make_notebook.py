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
    "inference.py", "run.py", "configs/base.yaml", "make_presentation.py",
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
    "streak cleared only on a real step": "NONFINITE_TOTAL_ABORT" in train_src,
    "lr = 5.0e-4":                lr_line == ["lr: 5.0e-4"],
    "fp32 default (amp: false)":  "amp: false" in cfg_txt,
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
subprocess.run(["pip", "-q", "install", "lpips", "python-pptx"], check=False)

from src.data import pack_dataset
meta = pack_dataset(DATA_SOURCE, "/kaggle/working/packed")
'''

CELL3 = '''# --- Cell 3: pre-flight gate ----------------------------------------------
# Runs in fp32 (the config default). Mixed precision repeatedly produced
# non-finite gradients on this stack, and a run that finishes beats one that
# is 2x faster and dies -- the model is only 0.68M parameters, so fp32 fits
# the time budget comfortably.
#
# The gate overfits 2 image pairs to prove the data path, loss and optimiser
# all work before committing hours of GPU time.
#
# Threshold is 28 dB, set from measurement rather than guessed: a 2000-step
# run reaches ~28.8 dB, while bicubic (i.e. doing nothing) scores 23.18 dB. A
# broken pipeline lands at or below bicubic, so 28 dB separates the two
# cleanly. An earlier 35 dB threshold was unreachable in 2000 steps and
# blocked a perfectly healthy run. ~2-4 minutes.
import subprocess

r = subprocess.run(
    ["python", "train.py", "--overfit", "2", "--steps", "2000",
     "--gate_db", "28", "--num_workers", "2", "--no_lpips",
     "--data_dir", "/kaggle/working/packed"])

assert r.returncode == 0, (
    "GATE FAILED (exit %d) -- do not start training. Non-zero means either the "
    "pipeline is broken or training diverged; the log above says which." % r.returncode)
print("")
print("GATE PASSED")
'''

CELL4 = '''# --- Cell 4: training (~4.5 h) -------------------------------------------
# Watch the loss over the first few hundred steps: it should fall steadily.
#
# Checkpoints every 1500 steps (not 5000): a Kaggle session that restarts
# clears /kaggle/working, and the first full run lost 6.4 hours of training
# that way. Download models/best.pt as soon as it appears -- do not leave the
# only copy in a session.
#
# num_workers=4 with prefetching: the first run managed 0.8 it/s on a T4 for a
# 0.68M model, which means the GPU was starving on the input pipeline rather
# than computing.
#
# Non-finite losses and gradients are skipped rather than applied, and the run
# aborts on 20 consecutive OR 200 total bad updates. Both bounds matter: an
# alternating good/bad cycle never builds a long streak, which is exactly how
# an earlier version spun ~8000 times without ever tripping its own guard.
import subprocess
r = subprocess.run(
    ["python", "train.py", "--config", "configs/base.yaml",
     "--data_dir", "/kaggle/working/packed",
     "--out_dir", "models",
     "--hours", "4.5", "--num_workers", "4",
     "--val_every", "1500"])
print("training exit code:", r.returncode)
assert r.returncode == 0, "training failed -- see the log above"
'''

CELL4B = '''# --- Cell 4b: SAVE THE CHECKPOINT NOW -------------------------------------
# Run this the moment models/best.pt first appears, and download the file from
# the Output panel. /kaggle/working is wiped when a session restarts, and
# "Save Version" is the only thing that persists it.
import os, shutil
src = "models/best.pt"
assert os.path.exists(src), "no checkpoint yet -- wait for the first validation"
shutil.copy(src, "/kaggle/working/best.pt")
print("%.1f MB -> /kaggle/working/best.pt  (download it from the Output panel)"
      % (os.path.getsize(src) / 1e6))
'''

CELL5 = '''# --- Cell 5: evaluate -----------------------------------------------------
# Retries without LPIPS if the perceptual metric fails for any reason: PSNR and
# SSIM are the numbers that matter, and losing them to an optional third metric
# would be the wrong trade.
import os, subprocess
assert os.path.exists("models/best.pt"), \\
    "no models/best.pt -- training has not reached its first validation yet"

BASE = ["python", "evaluate.py", "--weights", "models/best.pt",
        "--data_dir", "/kaggle/working/packed",
        "--out_dir", "/kaggle/working/results"]
if subprocess.run(BASE).returncode != 0:
    print("")
    print("evaluation failed -- retrying without LPIPS")
    subprocess.run(BASE + ["--no_lpips"], check=True)
print("")
print("metrics -> /kaggle/working/results/metrics.csv")
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
rc = subprocess.run(["python", "run.py",
                     "/kaggle/working/bench_in",
                     "/kaggle/working/bench_out"]).returncode
dt = time.perf_counter() - t0
if rc == 0:
    print("wall clock, full pipeline: %.2f s" % dt)
else:
    # Loud, but not fatal: cell 8 still needs to bundle the artifacts.
    print("!!! run.py FAILED (exit %d) -- the graded entry point is broken, fix "
          "before submitting" % rc)
'''

CELL8 = '''# --- Cell 8: regenerate the deck with real numbers, bundle artifacts ------
import glob, shutil, subprocess
shutil.copytree("/kaggle/working/results", "results", dirs_exist_ok=True)
if subprocess.run(["python", "make_presentation.py"]).returncode == 0:
    shutil.copy("solution_presentation.pptx", "/kaggle/working/")
    print("deck regenerated with the measured numbers")
else:
    print("deck generation failed (python-pptx missing?) -- run "
          "make_presentation.py locally instead; the metrics CSV is what it reads")

print("models:", glob.glob("models/*"))
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
    cells += [code(c) for c in (CELL2, CELL3, CELL4, CELL4B, CELL5, CELL6, CELL7, CELL8)]
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
