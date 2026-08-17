"""Standalone inference for the KLA restoration task.

    python inference.py --input_dir <degraded images> --output_dir <restored images>

Runs as-is, with no source edits and no extra arguments required.

End-to-end runtime is a scored axis and KLA's stopwatch covers script startup,
model init, disk reads, inference and disk writes.  So:

* only torch/numpy/argparse are imported at module level -- no yaml, no lpips,
  no skimage, no matplotlib (each would add import time for zero benefit);
* the model config travels inside the checkpoint, so there is no config parse;
* files are read on a thread pool (npy reads release the GIL);
* inputs are grouped by resolution and run in batches, channels_last + fp16
  autocast, under inference_mode;
* writes go back out on the same thread pool.

I/O contract
------------
* Input  : a directory of ``.npy`` (float32) and/or ``.png``/``.tif`` images.
* Output : one file per input, **same basename and same extension**, written to
  ``--output_dir``.  ``.npy`` in -> float32 ``.npy`` out at 2x resolution.
* Inputs are **never** clipped -- NoisyLR legitimately exceeds [0,1].
* Outputs are **always** clamped to [0,1]: KLA scores the images exactly as
  saved and does not clip or renormalise on their side.
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

NPY_EXT = {".npy"}
IMG_EXT = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def _model_from_ckpt(ckpt: dict) -> torch.nn.Module:
    """Rebuild the network from the config stored in the checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.model import build_model

    return build_model(ckpt.get("cfg", {}).get("model", {}))


def _read_one(path: Path):
    ext = path.suffix.lower()
    if ext in NPY_EXT:
        arr = np.load(path)
        return arr.astype(np.float32, copy=False), None
    from PIL import Image  # only reached when the test set is image-based

    im = Image.open(path)
    arr = np.asarray(im)
    info = {"mode": im.mode, "dtype": arr.dtype}
    if arr.ndim == 3:  # collapse an RGB encoding of grayscale data
        arr = arr[..., 0]
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    return arr.astype(np.float32) / scale, info


def _write_one(path: Path, arr: np.ndarray, info) -> None:
    if info is None:
        np.save(path, arr.astype(np.float32, copy=False))
        return
    from PIL import Image

    if info["dtype"] == np.uint16:
        out = (arr * 65535.0).round().clip(0, 65535).astype(np.uint16)
    else:
        out = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Restore degraded images (denoise + x2 SR)")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parent / "weights" / "best.pt"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4) * 2))
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--compile", action="store_true", help="torch.compile (rarely pays off)")
    ap.add_argument("--tta", action="store_true", help="x8 self-ensemble: ~+0.2 dB, ~8x slower")
    args = ap.parse_args()

    t0 = time.perf_counter()
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (NPY_EXT | IMG_EXT)
    )
    if not files:
        print(f"no input images found in {in_dir}", file=sys.stderr)
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.fp32) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    except Exception:
        # Older torch, or a checkpoint holding non-tensor objects.  Never let
        # this abort a scored run.
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = _model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)

    pool = ThreadPoolExecutor(max_workers=args.workers)
    loaded = list(pool.map(_read_one, files))

    # Group by resolution so every batch is a single stacked tensor.
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (arr, _) in enumerate(loaded):
        groups.setdefault(arr.shape, []).append(i)

    writes = []
    with torch.inference_mode():
        for shape, idxs in groups.items():
            for s in range(0, len(idxs), args.batch_size):
                chunk = idxs[s : s + args.batch_size]
                batch = torch.from_numpy(
                    np.stack([loaded[i][0] for i in chunk])
                ).unsqueeze(1)
                if device.type == "cuda":
                    batch = batch.pin_memory().to(device, non_blocking=True)
                    batch = batch.to(memory_format=torch.channels_last)
                else:
                    batch = batch.to(device)

                with torch.autocast("cuda", enabled=use_amp):
                    if args.tta:
                        acc = None
                        for k in range(4):
                            for flip in (False, True):
                                x = torch.rot90(batch, k, (-2, -1))
                                if flip:
                                    x = torch.flip(x, (-1,))
                                y = model(x)
                                if flip:
                                    y = torch.flip(y, (-1,))
                                y = torch.rot90(y, -k, (-2, -1))
                                acc = y.float() if acc is None else acc + y.float()
                        out = acc / 8.0
                    else:
                        out = model(batch)

                out = out.float().clamp_(0, 1).squeeze(1).cpu().numpy()
                for j, i in enumerate(chunk):
                    writes.append(
                        pool.submit(
                            _write_one,
                            out_dir / files[i].name,
                            out[j],
                            loaded[i][1],
                        )
                    )

    for w in writes:
        w.result()
    pool.shutdown(wait=True)

    dt = time.perf_counter() - t0
    print(
        f"restored {len(files)} images in {dt:.2f}s "
        f"({len(files)/dt:.1f} img/s, batch={args.batch_size}, device={device.type}, "
        f"amp={'fp16' if use_amp else 'fp32'}) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
