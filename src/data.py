"""Dataset packing, train/val split and the training Dataset.

Packing: the official archive holds 6400 small ``.npy`` files.  Loading those
individually every epoch is the wrong I/O pattern -- we pack once into two
contiguous memmaps (``gt.npy`` 3200x256x256, ``lr.npy`` 3200x128x128) so the
loader is essentially free.

Split: KLA's own slides show sample 000000 -> source ``0001.png`` and sample
000500 -> source ``0186.png``, i.e. samples are ordered by source image with
roughly 2-3 samples per source.  So a random per-index split leaks
near-duplicate content into train, and a contiguous tail slice hands validation
a few sources that appear nowhere else.  We instead hold out **10 evenly-spaced
contiguous blocks of 32**: contiguous keeps same-source samples together (no
leakage), spacing covers the full content diversity.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Entries look like ``train/GT/001804.npy``.  The archive was zipped on macOS and
# also contains ``__MACOSX/._*`` sidecars and a ``.DS_Store`` -- globbing without
# this filter feeds those to np.load and crashes.
ENTRY_RE = re.compile(r"^train/(GT|NoisyLR)/(\d{6})\.npy$")

GT_SIZE = 256
LR_SIZE = 128
N_VAL_BLOCKS = 10
VAL_BLOCK_SIZE = 32


def val_indices(n_total: int, n_blocks: int = N_VAL_BLOCKS, block: int = VAL_BLOCK_SIZE) -> list[int]:
    """Frozen validation indices: `n_blocks` contiguous blocks, evenly spaced."""
    stride = n_total // n_blocks
    idx: list[int] = []
    for b in range(n_blocks):
        start = b * stride
        idx.extend(range(start, min(start + block, n_total)))
    return sorted(idx)


def split_indices(n_total: int) -> tuple[list[int], list[int]]:
    val = val_indices(n_total)
    val_set = set(val)
    train = [i for i in range(n_total) if i not in val_set]
    return train, val


def _finish_packing(gt, lr, ids: list[str], out_dir: Path, meta_path: Path) -> dict:
    """Shared tail end of packing: integrity checks, split, meta.json."""
    n = len(ids)
    assert gt.shape == (n, GT_SIZE, GT_SIZE) and lr.shape == (n, LR_SIZE, LR_SIZE)
    assert gt.min() >= -1e-6 and gt.max() <= 1.0 + 1e-6, "GT must lie in [0,1]"
    assert lr.max() > 1.0, "LR should exceed 1.0 (speckle) -- packing looks wrong"

    train_idx, val_idx = split_indices(n)
    assert not (set(train_idx) & set(val_idx)), "train/val overlap"

    meta = {
        "n": n,
        "ids": ids,
        "gt_size": GT_SIZE,
        "lr_size": LR_SIZE,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "val_indices": val_idx,
        "gt_range": [float(gt.min()), float(gt.max())],
        "lr_range": [float(lr.min()), float(lr.max())],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[data] packed {n} pairs -> {out_dir} | "
        f"train {len(train_idx)} / val {len(val_idx)} | "
        f"GT [{meta['gt_range'][0]:.3f},{meta['gt_range'][1]:.3f}] "
        f"LR [{meta['lr_range'][0]:.3f},{meta['lr_range'][1]:.3f}]"
    )
    return meta


def _already_packed(out_dir: Path) -> dict | None:
    gt_path, lr_path, meta_path = out_dir / "gt.npy", out_dir / "lr.npy", out_dir / "meta.json"
    if meta_path.exists() and gt_path.exists() and lr_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[data] reusing packed arrays in {out_dir} ({meta['n']} pairs)")
        return meta
    return None


def pack_from_zip(zip_path: str | Path, out_dir: str | Path) -> dict:
    """Unpack the official archive into contiguous memmaps.  Idempotent."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    cached = _already_packed(out_dir)
    if cached:
        return cached

    with zipfile.ZipFile(zip_path) as zf:
        found: dict[str, set[str]] = {"GT": set(), "NoisyLR": set()}
        for name in zf.namelist():
            m = ENTRY_RE.match(name)
            if m:
                found[m.group(1)].add(m.group(2))

        ids = sorted(found["GT"] & found["NoisyLR"])
        if not ids:
            raise RuntimeError(f"no train/GT + train/NoisyLR pairs found in {zip_path}")
        missing = found["GT"] ^ found["NoisyLR"]
        if missing:
            print(f"[data] warning: {len(missing)} unpaired ids ignored")

        n = len(ids)
        gt = np.lib.format.open_memmap(
            out_dir / "gt.npy", mode="w+", dtype=np.float32, shape=(n, GT_SIZE, GT_SIZE)
        )
        lr = np.lib.format.open_memmap(
            out_dir / "lr.npy", mode="w+", dtype=np.float32, shape=(n, LR_SIZE, LR_SIZE)
        )
        for i, sid in enumerate(ids):
            gt[i] = np.load(io.BytesIO(zf.read(f"train/GT/{sid}.npy")))
            lr[i] = np.load(io.BytesIO(zf.read(f"train/NoisyLR/{sid}.npy")))
            if (i + 1) % 500 == 0:
                print(f"[data] packed {i+1}/{n}")
        gt.flush()
        lr.flush()

    return _finish_packing(gt, lr, ids, out_dir, meta_path)


def _locate_extracted_train_dir(root: Path) -> Path:
    """Find the ``train/`` directory (containing GT/ and NoisyLR/) under `root`.

    Kaggle unzips an uploaded .zip by default, so the dataset root may itself
    *be* ``train/``, or may contain it one level down, or the whole thing may
    sit under an extra wrapper directory from how the zip was created.
    """
    candidates = [root, root / "train"]
    candidates += [p / "train" for p in root.iterdir() if p.is_dir()]
    candidates += [p for p in root.iterdir() if p.is_dir()]
    for c in candidates:
        if (c / "GT").is_dir() and (c / "NoisyLR").is_dir():
            return c
    raise RuntimeError(
        f"could not find a GT/ + NoisyLR/ pair under {root} "
        f"(looked in: {[str(c) for c in candidates]})"
    )


def pack_from_dir(data_dir: str | Path, out_dir: str | Path) -> dict:
    """Pack an already-extracted GT/ + NoisyLR/ tree into contiguous memmaps.

    Handles the case where a Kaggle Dataset was created from a .zip and
    Kaggle auto-extracted it on upload, so there is no .zip file to read --
    just ``train/GT/*.npy`` and ``train/NoisyLR/*.npy`` directly on disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    cached = _already_packed(out_dir)
    if cached:
        return cached

    train_dir = _locate_extracted_train_dir(Path(data_dir))
    gt_dir, lr_dir = train_dir / "GT", train_dir / "NoisyLR"

    gt_ids = {p.stem for p in gt_dir.glob("*.npy") if re.fullmatch(r"\d{6}", p.stem)}
    lr_ids = {p.stem for p in lr_dir.glob("*.npy") if re.fullmatch(r"\d{6}", p.stem)}
    ids = sorted(gt_ids & lr_ids)
    if not ids:
        raise RuntimeError(f"no matching GT/NoisyLR pairs found under {train_dir}")
    missing = gt_ids ^ lr_ids
    if missing:
        print(f"[data] warning: {len(missing)} unpaired ids ignored")

    n = len(ids)
    gt = np.lib.format.open_memmap(
        out_dir / "gt.npy", mode="w+", dtype=np.float32, shape=(n, GT_SIZE, GT_SIZE)
    )
    lr = np.lib.format.open_memmap(
        out_dir / "lr.npy", mode="w+", dtype=np.float32, shape=(n, LR_SIZE, LR_SIZE)
    )
    for i, sid in enumerate(ids):
        gt[i] = np.load(gt_dir / f"{sid}.npy")
        lr[i] = np.load(lr_dir / f"{sid}.npy")
        if (i + 1) % 500 == 0:
            print(f"[data] packed {i+1}/{n}")
    gt.flush()
    lr.flush()

    return _finish_packing(gt, lr, ids, out_dir, meta_path)


def pack_dataset(source: str | Path, out_dir: str | Path) -> dict:
    """Pack the official data regardless of whether `source` is the .zip
    archive or an already-extracted directory (Kaggle auto-extracts uploaded
    zips by default)."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        return pack_from_zip(source, out_dir)
    if source.is_dir():
        return pack_from_dir(source, out_dir)
    raise FileNotFoundError(f"{source} is neither a .zip file nor a directory")


class PairDataset(Dataset):
    """Returns aligned (GT crop, real NoisyLR crop) pairs.

    The training loop replaces a random fraction of the LR crops with
    synthetically degraded versions of the same GT crop -- see
    :func:`src.degrade.degrade_batch`.

    Memmaps are opened lazily inside each worker: on Windows DataLoader workers
    are spawned, not forked, and an already-open memmap does not survive
    pickling.
    """

    def __init__(
        self,
        data_dir: str | Path,
        indices: list[int],
        crop_lr: int = 64,
        augment: bool = True,
        full_image: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.indices = list(indices)
        self.crop_lr = crop_lr
        self.augment = augment
        self.full_image = full_image
        self._gt = None
        self._lr = None

    def _arrays(self):
        if self._gt is None:
            self._gt = np.load(self.data_dir / "gt.npy", mmap_mode="r")
            self._lr = np.load(self.data_dir / "lr.npy", mmap_mode="r")
        return self._gt, self._lr

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gt_all, lr_all = self._arrays()
        j = self.indices[i]

        if self.full_image:
            # np.array (not asarray): memmap slices are read-only views and
            # torch.from_numpy warns on those.
            gt = np.array(gt_all[j], dtype=np.float32)
            lr = np.array(lr_all[j], dtype=np.float32)
        else:
            c = self.crop_lr
            max_off = lr_all.shape[1] - c
            y = np.random.randint(0, max_off + 1)
            x = np.random.randint(0, max_off + 1)
            lr = np.array(lr_all[j, y : y + c, x : x + c], dtype=np.float32)
            gt = np.array(
                gt_all[j, 2 * y : 2 * (y + c), 2 * x : 2 * (x + c)], dtype=np.float32
            )

            if self.augment:
                k = np.random.randint(4)
                if k:
                    gt, lr = np.rot90(gt, k), np.rot90(lr, k)
                if np.random.rand() < 0.5:
                    gt, lr = gt[:, ::-1], lr[:, ::-1]
                if np.random.rand() < 0.5:
                    gt, lr = gt[::-1], lr[::-1]
                gt, lr = np.ascontiguousarray(gt), np.ascontiguousarray(lr)

        return torch.from_numpy(gt).unsqueeze(0), torch.from_numpy(lr).unsqueeze(0)
