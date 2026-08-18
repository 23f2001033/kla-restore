"""Evaluation metrics: PSNR, SSIM, LPIPS.

Conventions (stated explicitly because evaluators can differ):

* PSNR with ``data_range=1.0``, computed on the **clipped** prediction.
* SSIM with an 11x11 Gaussian window, sigma=1.5, ``data_range=1.0`` -- the
  Wang et al. convention, matching skimage's
  ``gaussian_weights=True, sigma=1.5, use_sample_covariance=False``.
  skimage's *default* (7x7 uniform) yields a different number.
* LPIPS-VGG on 3-channel-replicated grayscale, inputs mapped to [-1, 1].
"""

from __future__ import annotations

import torch

from .losses import ssim as _ssim


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image PSNR, averaged over the batch.

    Averaging PSNR *per image* (not over one pooled MSE) is the standard
    convention and is what a per-image leaderboard mean reports.
    """
    pred = pred.float().clamp(0, data_range)
    target = target.float()
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return (10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))).mean()


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return _ssim(pred.float().clamp(0, data_range), target.float(), data_range=data_range)


class LPIPSMetric:
    """Thin lazy wrapper -- imported only where LPIPS is actually needed, never
    in the inference script (every import counts against end-to-end runtime)."""

    def __init__(self, device, net: str = "vgg"):
        import lpips  # noqa: PLC0415

        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=net).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Callers often hand us CPU tensors (evaluate.py moves predictions back
        # to host memory before scoring), while the LPIPS network lives on the
        # GPU. Move the inputs rather than assuming they already match.
        p3 = pred.to(self.device).float().clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        t3 = target.to(self.device).float().repeat(1, 3, 1, 1) * 2 - 1
        return self.model(p3, t3).mean().cpu()
