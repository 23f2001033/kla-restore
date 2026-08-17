"""Training losses.

    L = Charbonnier + 0.15 * (1 - SSIM)   [+ 0.05 * LPIPS in the final fine-tune]

* Charbonnier (smooth L1) rather than L2: better PSNR/SSIM trade-off and far
  more robust to the heavy-tailed speckle outliers in this data.
* **Single-scale SSIM, not MS-SSIM.**  MS-SSIM's default 5 scales need images of
  at least ~161px; our GT training crops are 128x128, so MS-SSIM would either
  error or silently fall back to garbage.  MS-SSIM is fine for *reporting* on
  full 256x256 images -- just not as a crop-level training loss.
* LPIPS is a scored metric but running VGG every step costs ~30% throughput for
  little gain, so it is switched on only for the last ~10% of training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean SSIM over the batch.

    11x11 Gaussian window, sigma=1.5, data_range=1.0 -- i.e. the Wang et al.
    convention, equivalent to skimage's
    ``gaussian_weights=True, sigma=1.5, use_sample_covariance=False``.
    (skimage's *default* is a 7x7 uniform window, which gives a different
    number; we report the Gaussian convention and say so.)
    """
    c = x.shape[1]
    dtype = x.dtype if x.is_floating_point() else torch.float32
    g = _gaussian_window(window_size, sigma, x.device, dtype)
    win_x = g.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    win_y = g.view(1, 1, -1, 1).expand(c, 1, -1, 1)

    def blur(t: torch.Tensor) -> torch.Tensor:
        t = F.conv2d(t, win_x, groups=c)
        return F.conv2d(t, win_y, groups=c)

    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x, mu_y = blur(x), blur(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = blur(x * x) - mu_x2
    sig_y = blur(y * y) - mu_y2
    sig_xy = blur(x * y) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2)
    return (num / den).mean()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((x - y) ** 2 + self.eps2).mean()


class RestorationLoss(nn.Module):
    """Combined loss.  ``lpips_weight`` starts at 0 and is raised for the
    final fine-tune phase via :meth:`enable_lpips`."""

    def __init__(self, ssim_weight: float = 0.15, lpips_weight: float = 0.05):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self._lpips = None
        self.use_lpips = False

    def enable_lpips(self, device) -> bool:
        """Lazily construct the LPIPS network.  Returns False (and stays off)
        if the package is unavailable, so training never dies over an optional
        loss term."""
        if self._lpips is None:
            try:
                import lpips  # noqa: PLC0415 -- optional, deliberately lazy

                self._lpips = lpips.LPIPS(net="vgg").to(device).eval()
                for p in self._lpips.parameters():
                    p.requires_grad_(False)
            except Exception as exc:  # pragma: no cover
                print(f"[loss] LPIPS unavailable ({exc}); continuing without it")
                return False
        self.use_lpips = True
        return True

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # SSIM/LPIPS are ill-behaved in fp16; compute the loss in fp32.
        pred = pred.float()
        target = target.float()

        l_pix = self.charb(pred, target)
        l_ssim = 1.0 - ssim(pred.clamp(0, 1), target)
        total = l_pix + self.ssim_weight * l_ssim
        parts = {"charb": l_pix.detach(), "ssim": l_ssim.detach()}

        if self.use_lpips and self._lpips is not None:
            # lpips expects 3-channel input in [-1, 1].  Feeding [0,1] is a
            # silent accuracy bug, so the rescale is explicit.
            p3 = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
            t3 = target.repeat(1, 3, 1, 1) * 2 - 1
            l_lp = self._lpips(p3, t3).mean()
            total = total + self.lpips_weight * l_lp
            parts["lpips"] = l_lp.detach()

        return total, parts
