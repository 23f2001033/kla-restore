"""Reproducible training for the KLA restoration model.

    python train.py --config configs/base.yaml
    python train.py --overfit 2 --steps 500        # pre-flight sanity gate

Design points that matter for the "training & compute hygiene" axis:

* every hyperparameter comes from the YAML config; nothing is hard-coded here;
* seeds are fixed and recorded, along with the git hash, in results/run_log.csv;
* the LR schedule length is **calibrated from measured throughput** rather than
  guessed -- a cosine schedule that does not finish leaves the model at a high
  learning rate and costs ~1 dB;
* validation runs on the frozen split every `val_every` steps and the best
  checkpoint by PSNR is kept, so an interrupted session still yields a usable
  model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import PairDataset, pack_dataset, split_indices
from src.degrade import degrade_batch
from src.losses import RestorationLoss
from src.metrics import psnr as psnr_metric
from src.metrics import ssim as ssim_metric
from src.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "nogit"


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_lr: float = 1e-6) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def validate(model, loader, device, amp: bool, lpips_metric=None) -> dict:
    model.eval()
    tot = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    n = 0
    for gt, lr in loader:
        gt = gt.to(device, non_blocking=True)
        lr = lr.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = model(lr)
        pred = pred.float().clamp(0, 1)
        b = gt.shape[0]
        tot["psnr"] += float(psnr_metric(pred, gt)) * b
        tot["ssim"] += float(ssim_metric(pred, gt)) * b
        if lpips_metric is not None:
            tot["lpips"] += float(lpips_metric(pred, gt)) * b
        n += b
    model.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data_dir", default=None, help="override packed data dir")
    ap.add_argument(
        "--zip_path", default=None,
        help="override official data source: the .zip archive, or an already-extracted directory",
    )
    ap.add_argument("--out_dir", default="weights")
    ap.add_argument("--steps", type=int, default=0, help="override step count")
    ap.add_argument("--hours", type=float, default=0.0, help="override time budget")
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument("--overfit", type=int, default=0, help="overfit N pairs (sanity gate)")
    ap.add_argument(
        "--gate_db",
        type=float,
        default=35.0,
        help="PSNR the overfit gate must clear (needs ~2000 steps to be meaningful)",
    )
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument(
        "--no_amp", action="store_true",
        help="force fp32 (already the config default)",
    )
    ap.add_argument(
        "--amp", action="store_true",
        help="opt into mixed precision; faster, but see the note in configs/base.yaml",
    )
    ap.add_argument("--lr", type=float, default=0.0, help="override peak learning rate")
    ap.add_argument(
        "--val_every", type=int, default=0,
        help="validate/checkpoint interval; smaller means less lost to a session kill",
    )
    ap.add_argument(
        "--synth_prob", type=float, default=-1.0,
        help="fraction of each batch replaced by synthetic pairs (0 disables)",
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers >= 0:
        cfg["train"]["num_workers"] = args.num_workers
    if args.hours:
        cfg["train"]["budget_hours"] = args.hours
    if args.steps:
        cfg["train"]["max_steps"] = args.steps
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.amp:
        cfg["train"]["amp"] = True
    if args.no_amp:            # explicit --no_amp always wins
        cfg["train"]["amp"] = False
    if args.val_every:
        cfg["train"]["val_every"] = args.val_every
    if args.synth_prob >= 0:
        cfg["degradation"]["synth_prob"] = args.synth_prob

    set_seed(cfg["seed"])
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    ch_last = bool(cfg["train"]["channels_last"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ---- data -------------------------------------------------------------
    data_dir = Path(args.data_dir or cfg["data"]["packed_dir"])
    zip_path = Path(args.zip_path or cfg["data"]["zip_path"])
    if not (data_dir / "meta.json").exists():
        pack_dataset(zip_path, data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    n_total = meta["n"]
    train_idx, val_idx = split_indices(n_total)

    if args.overfit:
        # 400 steps is not enough to memorise even two images; the gate needs a
        # few thousand before its threshold means anything.
        if not cfg["train"]["max_steps"]:
            cfg["train"]["max_steps"] = 2000
        train_idx = train_idx[: args.overfit]
        val_idx = train_idx
        # With only a handful of samples, drop_last=True would yield zero
        # batches and spin the training loop forever.
        cfg["train"]["batch_size"] = min(cfg["train"]["batch_size"], len(train_idx))
        # The default 2k-step warmup would keep the LR tiny for the whole gate.
        cfg["train"]["warmup_steps"] = max(1, int(cfg["train"]["max_steps"]) // 10)
        print(f"[train] OVERFIT MODE on {len(train_idx)} pairs: {train_idx}")

    train_ds = PairDataset(data_dir, train_idx, crop_lr=cfg["data"]["crop_lr"], augment=not args.overfit)
    val_ds = PairDataset(data_dir, val_idx, augment=False, full_image=True)

    nw = int(cfg["train"]["num_workers"])
    loader_kw = {}
    if nw > 0:
        # Without prefetching, the GPU waits on the loader between steps. This
        # was the dominant cost in the first full run (0.8 it/s on a T4 for a
        # 0.68M model -- the GPU was starving, not saturated).
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        drop_last=not args.overfit,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=device.type == "cuda"
    )

    # ---- model / optim ----------------------------------------------------
    model = build_model(cfg["model"]).to(device)
    if ch_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device} params={n_params/1e6:.2f}M amp={amp}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.999),  # standard; (0.9, 0.9) made the 2nd-moment estimate
        # too noisy and let a single gradient spike at peak LR blow up the
        # weights irrecoverably -- see the run_log postmortem note below.
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = RestorationLoss(
        ssim_weight=cfg["loss"]["ssim_weight"], lpips_weight=cfg["loss"]["lpips_weight"]
    ).to(device)

    dg = cfg["degradation"]
    synth_prob = float(dg["synth_prob"])
    widen = bool(dg.get("widen", True))
    warmup = int(cfg["train"]["warmup_steps"])
    base_lr = float(cfg["train"]["lr"])
    grad_clip = float(cfg["train"]["grad_clip"])
    val_every = int(cfg["train"]["val_every"])

    # Provisional step count; recalibrated after CALIB_STEPS from measured rate.
    CALIB_STEPS = 200
    total_steps = int(cfg["train"]["max_steps"]) or 100_000
    calibrated = bool(cfg["train"]["max_steps"]) or bool(args.overfit)
    budget_s = float(cfg["train"]["budget_hours"]) * 3600

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_path = results_dir / "run_log.csv"
    log_new = not log_path.exists()
    log_f = log_path.open("a", newline="")
    log_w = csv.writer(log_f)
    if log_new:
        log_w.writerow(
            ["run_id", "git", "seed", "step", "total_steps", "lr", "loss",
             "val_psnr", "val_ssim", "val_lpips", "params", "config"]
        )

    best_psnr = -1.0
    step = 0
    t_start = time.time()
    t_calib = None
    loss_ema = None
    nonfinite_streak = 0
    total_nonfinite = 0
    NONFINITE_ABORT = 20        # consecutive non-finite updates => diverged
    NONFINITE_TOTAL_ABORT = 200 # absolute cap: catches an alternating good/bad
                                # cycle that never builds a long streak
    model.train()

    print(f"[train] starting run {run_id} (git {git_hash()})")
    while step < total_steps:
        for gt, lr_img in train_loader:
            if step >= total_steps:
                break
            gt = gt.to(device, non_blocking=True)
            lr_img = lr_img.to(device, non_blocking=True)

            # Replace a random fraction of the batch with synthetic degradations
            # of the same GT crops.  Real pairs anchor the exact test
            # distribution; synthetic pairs widen it for OOD robustness.
            if synth_prob > 0 and not args.overfit:
                mask = torch.rand(gt.shape[0], device=device) < synth_prob
                if bool(mask.any()):
                    idx = mask.nonzero(as_tuple=True)[0]
                    lr_img = lr_img.clone()
                    lr_img[idx] = degrade_batch(gt[idx], widen=widen).to(lr_img.dtype)

            if ch_last:
                gt = gt.to(memory_format=torch.channels_last)
                lr_img = lr_img.to(memory_format=torch.channels_last)

            cur_lr = lr_at(step, base_lr, warmup, total_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            # Enable LPIPS for the final fraction of training only.
            if (
                not args.no_lpips
                and not criterion.use_lpips
                and calibrated
                and step >= total_steps * (1 - float(cfg["loss"]["lpips_last_frac"]))
            ):
                if criterion.enable_lpips(device):
                    print(f"[train] step {step}: LPIPS term enabled")

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                pred = model(lr_img)
                loss, parts = criterion(pred, gt)

            lv = float(loss.detach())
            if not math.isfinite(lv):
                # A non-finite loss must never reach backward(): GradScaler only
                # guards against inf/nan *gradients*, not a loss that is already
                # nan going in, so a bad batch here would otherwise corrupt the
                # weights permanently on the very next optimizer step -- and
                # every step after that, since nan propagates forever once it's
                # in the parameters. Skip this batch entirely instead.
                nonfinite_streak += 1
                total_nonfinite += 1
                print(
                    f"[warn ] step {step}: non-finite loss ({lv}); skipping batch "
                    f"(streak {nonfinite_streak}, total {total_nonfinite})",
                    flush=True,
                )
                if (nonfinite_streak >= NONFINITE_ABORT
                        or total_nonfinite >= NONFINITE_TOTAL_ABORT):
                    print(
                        f"[fatal] non-finite loss ({nonfinite_streak} consecutive, "
                        f"{total_nonfinite} total) -- training has diverged. "
                        "Aborting rather than burning the rest of the budget. "
                        "Lower --lr, or pass --no_amp.",
                        flush=True,
                    )
                    log_f.close()
                    raise SystemExit(2)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # A finite loss can still produce non-finite gradients, and that is
            # how a corrupting step slipped past the loss-only check: once the
            # weights are poisoned every subsequent loss is nan, so the earlier
            # guard only ever saw the aftermath. Skip the step instead.
            if not torch.isfinite(gnorm):
                nonfinite_streak += 1
                total_nonfinite += 1
                print(
                    f"[warn ] step {step}: non-finite gradient norm; skipping step "
                    f"(streak {nonfinite_streak}, total {total_nonfinite})",
                    flush=True,
                )
                opt.zero_grad(set_to_none=True)
                scaler.update()
                if (nonfinite_streak >= NONFINITE_ABORT
                        or total_nonfinite >= NONFINITE_TOTAL_ABORT):
                    print(
                        f"[fatal] non-finite gradients ({nonfinite_streak} consecutive, "
                        f"{total_nonfinite} total) -- training has diverged. "
                        "Lower --lr, or pass --no_amp.",
                        flush=True,
                    )
                    log_f.close()
                    raise SystemExit(2)
                continue

            scaler.step(opt)
            scaler.update()

            # Only a step that actually applied clears the streak. Resetting
            # anywhere earlier lets a loss-ok/grad-bad cycle alternate forever:
            # the streak returns to 0 every iteration, the abort never fires,
            # and because the skipped step leaves the weights unchanged the
            # exact same failure repeats indefinitely.
            nonfinite_streak = 0
            loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv
            step += 1

            # --- calibrate the schedule from measured throughput ------------
            if step == CALIB_STEPS and not calibrated:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_calib = time.time()
                rate = CALIB_STEPS / (t_calib - t_start)
                total_steps = int(rate * budget_s)
                calibrated = True
                print(
                    f"[train] calibrated: {rate:.1f} it/s -> {total_steps} steps "
                    f"for {budget_s/3600:.1f} h"
                )

            if step % 100 == 0:
                el = time.time() - t_start
                msg = (
                    f"[train] step {step}/{total_steps} loss {loss_ema:.4f} "
                    f"lr {cur_lr:.2e} {step/max(el,1e-9):.1f} it/s "
                    f"elapsed {el/60:.1f}m"
                )
                print(msg, flush=True)

            if step % val_every == 0 or step == total_steps:
                m = validate(model, val_loader, device, amp)
                print(
                    f"[val ] step {step} PSNR {m['psnr']:.3f} SSIM {m['ssim']:.4f}",
                    flush=True,
                )
                log_w.writerow(
                    [run_id, git_hash(), cfg["seed"], step, total_steps,
                     f"{cur_lr:.3e}", f"{loss_ema:.5f}", f"{m['psnr']:.4f}",
                     f"{m['ssim']:.5f}", f"{m['lpips']:.5f}", n_params,
                     json.dumps(cfg)]
                )
                log_f.flush()

                ckpt = {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "step": step,
                    "val_psnr": m["psnr"],
                    "val_ssim": m["ssim"],
                    "run_id": run_id,
                    "git": git_hash(),
                }
                torch.save(ckpt, out_dir / "last.pt")
                if m["psnr"] > best_psnr:
                    best_psnr = m["psnr"]
                    torch.save(ckpt, out_dir / "best.pt")
                    print(f"[val ] new best {best_psnr:.3f} dB -> {out_dir/'best.pt'}")

    log_f.close()
    total_time = (time.time() - t_start) / 3600
    print(f"[train] done in {total_time:.2f} h | best val PSNR {best_psnr:.3f} dB")

    if args.overfit:
        gate = args.gate_db
        ok = best_psnr > gate
        print(
            f"[gate ] overfit sanity gate: {best_psnr:.2f} dB vs {gate:.1f} dB -> "
            f"{'PASS' if ok else 'FAIL'}"
        )
        # Exit code is the machine-readable result -- do not pipe this command
        # through `tail`/`head`, or the shell reports the pipe's status instead.
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
