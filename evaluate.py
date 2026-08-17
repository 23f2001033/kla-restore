"""Evaluate a checkpoint on the frozen validation split.

    python evaluate.py --weights weights/best.pt
    python evaluate.py --weights weights/best.pt --ood_dir path/to/external/images

Reports PSNR / SSIM / LPIPS against the bicubic baseline, writes per-image
numbers to results/metrics.csv, and saves qualitative figures including the
worst-performing (failure) cases.

Metric conventions are stated in src/metrics.py and repeated in the README:
PSNR data_range=1.0 on the clipped prediction; SSIM 11x11 Gaussian window,
sigma=1.5 (skimage's `gaussian_weights=True`, not its 7x7 uniform default).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import split_indices  # noqa: E402
from src.degrade import degrade_batch  # noqa: E402
from src.metrics import psnr as psnr_metric  # noqa: E402
from src.metrics import ssim as ssim_metric  # noqa: E402
from src.model import build_model  # noqa: E402


def bicubic_up(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)


@torch.no_grad()
def run_model(model, lr, device, amp: bool, batch: int = 16) -> torch.Tensor:
    outs = []
    for s in range(0, lr.shape[0], batch):
        x = lr[s : s + batch].to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            y = model(x)
        outs.append(y.float().clamp(0, 1).cpu())
    return torch.cat(outs)


def summarize(name, pred, gt, lpips_fn=None) -> dict:
    per_psnr = []
    per_ssim = []
    for i in range(pred.shape[0]):
        per_psnr.append(float(psnr_metric(pred[i : i + 1], gt[i : i + 1])))
        per_ssim.append(float(ssim_metric(pred[i : i + 1], gt[i : i + 1])))
    row = {
        "method": name,
        "psnr": float(np.mean(per_psnr)),
        "ssim": float(np.mean(per_ssim)),
        "lpips": float("nan"),
    }
    if lpips_fn is not None:
        vals = [
            float(lpips_fn(pred[i : i + 1], gt[i : i + 1])) for i in range(pred.shape[0])
        ]
        row["lpips"] = float(np.mean(vals))
    return row, per_psnr, per_ssim


def save_figures(lr, pred, gt, per_psnr, out_dir: Path, k: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(per_psnr)
    picks = list(order[:k]) + list(order[-k:])  # worst k (failures) + best k
    labels = ["FAILURE"] * k + ["BEST"] * k

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    for r, (i, lab) in enumerate(zip(picks, labels)):
        up = F.interpolate(lr[i : i + 1], scale_factor=2, mode="nearest")[0, 0]
        for c, (img, title) in enumerate(
            [
                (up, f"input (x2 nearest)"),
                (pred[i, 0], f"restored  {per_psnr[i]:.2f} dB"),
                (gt[i, 0], "ground truth"),
            ]
        ):
            ax = axes[r, c]
            ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{lab} #{i} {title}" if c == 0 else title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative.png", dpi=130)
    plt.close(fig)
    print(f"[eval] wrote {out_dir/'qualitative.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--data_dir", default="data/packed")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--ood_dir", default=None, help="dir of external grayscale images")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("cfg", {}).get("model", {}))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[eval] {args.weights} step={ckpt.get('step')} "
        f"train_val_psnr={ckpt.get('val_psnr')} params={n_params/1e6:.2f}M"
    )

    meta = json.loads((Path(args.data_dir) / "meta.json").read_text())
    _, val_idx = split_indices(meta["n"])
    gt_all = np.load(Path(args.data_dir) / "gt.npy", mmap_mode="r")
    lr_all = np.load(Path(args.data_dir) / "lr.npy", mmap_mode="r")
    gt = torch.from_numpy(np.asarray(gt_all[val_idx]).copy()).unsqueeze(1)
    lr = torch.from_numpy(np.asarray(lr_all[val_idx]).copy()).unsqueeze(1)
    print(f"[eval] validation split: {len(val_idx)} images")

    lpips_fn = None
    if not args.no_lpips:
        try:
            from src.metrics import LPIPSMetric

            lpips_fn = LPIPSMetric(device)
        except Exception as exc:
            print(f"[eval] LPIPS unavailable ({exc}); skipping")

    rows = []
    base = bicubic_up(lr).clamp(0, 1)
    r, _, _ = summarize("bicubic", base, gt, lpips_fn)
    rows.append(r)

    t0 = time.perf_counter()
    pred = run_model(model, lr, device, amp)
    dt = time.perf_counter() - t0
    r, per_psnr, per_ssim = summarize("model", pred, gt, lpips_fn)
    r["sec_per_image"] = dt / len(val_idx)
    rows.append(r)

    print("\n  method     PSNR      SSIM      LPIPS")
    for row in rows:
        print(
            f"  {row['method']:<9s} {row['psnr']:7.3f}  {row['ssim']:7.4f}  {row['lpips']:7.4f}"
        )
    print(f"\n  gain over bicubic: {rows[1]['psnr']-rows[0]['psnr']:+.2f} dB")

    # ---- OOD probe: degrade unseen external images with our engine ---------
    if args.ood_dir:
        from PIL import Image

        paths = sorted(
            p for p in Path(args.ood_dir).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )[:40]
        imgs = []
        for p in paths:
            im = Image.open(p).convert("L")
            w, h = im.size
            s = min(w, h) // 256 * 256 or 256
            im = im.crop((0, 0, min(w, s), min(h, s))).resize((256, 256), Image.BICUBIC)
            imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
        if imgs:
            ood_gt = torch.from_numpy(np.stack(imgs)).unsqueeze(1)
            torch.manual_seed(0)
            ood_lr = degrade_batch(ood_gt)
            ood_pred = run_model(model, ood_lr, device, amp)
            r_b, _, _ = summarize("ood_bicubic", bicubic_up(ood_lr).clamp(0, 1), ood_gt, lpips_fn)
            r_m, _, _ = summarize("ood_model", ood_pred, ood_gt, lpips_fn)
            rows += [r_b, r_m]
            print(
                f"\n  OOD ({len(imgs)} external images): "
                f"bicubic {r_b['psnr']:.2f} dB -> model {r_m['psnr']:.2f} dB "
                f"({r_m['psnr']-r_b['psnr']:+.2f})"
            )

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "psnr", "ssim", "lpips", "sec_per_image"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[eval] wrote {out_dir/'metrics.csv'}")

    if not args.no_figures:
        save_figures(lr, pred, gt, per_psnr, out_dir)


if __name__ == "__main__":
    main()
