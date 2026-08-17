"""Official submission entry point for the KLA restoration problem statement.

    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir>, restores it (denoise + 2x super-resolution),
and writes one .npy per input to <output-dir> under the same filename.

This file is deliberately SELF-CONTAINED: the model definition is inline rather
than imported from src/, so the script runs correctly even if only run.py,
requirements.txt, README.md and models/ are copied. It imports nothing beyond
torch and numpy, needs no internet access, no API keys, no additional
downloads, no user interaction and no manual configuration.

Output contract, per the submission checklist:
  * one output file per input file, identical filename
  * grayscale, shape (H, W)
  * float32 values inside [0, 1], guaranteed free of NaN and Inf
  * 2x the input resolution (128x128 -> 256x256, 256x256 -> 512x512)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# model (inline copy of src/model.py -- keep the two in sync)
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm, always evaluated in fp32.

    Variance squares its input, and under autocast fp16 accumulates that
    reduction in fp16 too -- mean(x^2) overflows to inf once activations reach
    only ~100, far below fp16's 65504 ceiling. Forcing fp32 here costs almost
    nothing and removes the failure mode.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            mu = x32.mean(dim=1, keepdim=True)
            var = x32.var(dim=1, keepdim=True, unbiased=False)
            y = (x32 - mu) / torch.sqrt(var + self.eps)
            y = (y * self.weight[None, :, None, None].float()
                 + self.bias[None, :, None, None].float())
        return y.to(x.dtype)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw, ffn = c * dw_expand, c * ffn_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the gated branch ONCE and reuse it for the attention weights.
        # Inlining this as a single nested expression evaluates the whole
        # conv1->conv2->gate chain twice, which is numerically identical but
        # costs ~40% more compute in the script whose runtime is scored.
        y = self.sg(self.conv2(self.conv1(self.norm1(x))))
        y = self.conv3(y * self.sca(y))
        x = x + y * self.beta

        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class RestoreNet(nn.Module):
    """Joint denoise + 2x super-resolution. Fully convolutional, so the same
    weights handle any input size."""

    def __init__(self, width: int = 64, blocks: int = 16, in_ch: int = 1,
                 global_skip: bool = True):
        super().__init__()
        self.global_skip = global_skip
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(blocks)])
        self.body_tail = nn.Conv2d(width, width, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1), nn.PixelShuffle(2))
        self.tail = nn.Conv2d(width, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.intro(x)
        feat = feat + self.body_tail(self.body(feat))
        out = self.tail(self.upsample(feat))
        if self.global_skip:
            out = out + torch.nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False)
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def find_checkpoint(explicit: str | None) -> Path:
    """Locate the weights. Searches models/ so the filename need not be fixed."""
    here = Path(__file__).resolve().parent
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = here / p
        if p.is_file():
            return p
        raise FileNotFoundError(f"weights not found: {p}")

    for folder in ("models", "weights"):
        d = here / folder
        if not d.is_dir():
            continue
        for name in ("best.pt", "model.pt", "final.pt"):
            if (d / name).is_file():
                return d / name
        found = sorted(list(d.glob("*.pt")) + list(d.glob("*.pth")))
        if found:
            return found[0]
    raise FileNotFoundError(
        f"no checkpoint found under {here/'models'} -- expected a .pt or .pth file")


def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    mcfg = ckpt.get("cfg", {}).get("model", {}) if isinstance(ckpt, dict) else {}
    model = RestoreNet(
        width=mcfg.get("width", 64),
        blocks=mcfg.get("blocks", 16),
        in_ch=mcfg.get("in_ch", 1),
        global_skip=mcfg.get("global_skip", True),
    )
    model.load_state_dict(state)

    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(
            f"checkpoint {ckpt_path} has non-finite parameters ({len(bad)} tensors, "
            f"e.g. {bad[0]}) -- refusing to produce garbage output.")

    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def read_npy(path: Path) -> np.ndarray:
    a = np.load(path)
    a = np.squeeze(a)                       # (H,W,1) or (1,H,W) -> (H,W)
    if a.ndim == 3:                         # any residual channel dim
        a = a[..., 0] if a.shape[-1] <= 4 else a[0]
    return np.ascontiguousarray(a, dtype=np.float32)


def write_npy(path: Path, arr: np.ndarray) -> None:
    np.save(path, arr.astype(np.float32, copy=False))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Restore degraded .npy images (denoise + 2x super-resolution)")
    # Positional, per the submission spec: python run.py <input-dir> <output-dir>
    ap.add_argument("input_dir", nargs="?", default=None)
    ap.add_argument("output_dir", nargs="?", default=None)
    # Flag forms kept as aliases so either calling convention works.
    ap.add_argument("--input_dir", dest="input_flag", default=None)
    ap.add_argument("--output_dir", dest="output_flag", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    args = ap.parse_args()

    in_dir = args.input_dir or args.input_flag
    out_dir = args.output_dir or args.output_flag
    if not in_dir or not out_dir:
        ap.error("usage: python run.py <input-dir> <output-dir>")

    t0 = time.perf_counter()
    in_path, out_path = Path(in_dir), Path(out_dir)
    if not in_path.is_dir():
        print(f"input directory does not exist: {in_path}", file=sys.stderr)
        raise SystemExit(1)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_path.iterdir()
                   if p.is_file() and p.suffix.lower() == ".npy")
    if not files:
        print(f"no .npy files found in {in_path}", file=sys.stderr)
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.fp32) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_model(find_checkpoint(args.weights), device)

    workers = min(16, (os.cpu_count() or 4) * 2)
    pool = ThreadPoolExecutor(max_workers=workers)
    arrays = list(pool.map(read_npy, files))

    # Group by resolution so each batch stacks cleanly.
    groups: dict[tuple[int, int], list[int]] = {}
    for i, a in enumerate(arrays):
        groups.setdefault(a.shape, []).append(i)

    written = 0
    futures = []
    with torch.inference_mode():
        for _, idxs in groups.items():
            for s in range(0, len(idxs), args.batch_size):
                chunk = idxs[s: s + args.batch_size]
                batch = torch.from_numpy(np.stack([arrays[i] for i in chunk])).unsqueeze(1)
                if device.type == "cuda":
                    batch = batch.pin_memory().to(device, non_blocking=True)
                    batch = batch.to(memory_format=torch.channels_last)
                else:
                    batch = batch.to(device)

                with torch.autocast("cuda", enabled=use_amp):
                    out = model(batch)

                # Guarantee the output contract: finite, then clamped to [0,1].
                out = out.float()
                out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
                out = out.clamp_(0.0, 1.0)
                arr = out.squeeze(1).cpu().numpy()

                for j, i in enumerate(chunk):
                    futures.append(pool.submit(write_npy, out_path / files[i].name, arr[j]))
                    written += 1

    for f in futures:
        f.result()
    pool.shutdown(wait=True)

    dt = time.perf_counter() - t0
    print(f"restored {written}/{len(files)} images in {dt:.2f}s "
          f"({written/dt:.1f} img/s, device={device.type}, "
          f"precision={'fp16' if use_amp else 'fp32'}) -> {out_path}")

    if written != len(files):
        print(f"ERROR: wrote {written} outputs for {len(files)} inputs", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
