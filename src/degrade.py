"""Degradation engine for the KLA restoration task.

The forward model was reverse-engineered from the provided training pairs
(see results/degradation_analysis.md).  Fitting ``LR - area_downsample(GT)``
over 60 sampled pairs gives:

* downsampling is a 2x2 area (box) average -- lowest residual of every kernel
  tested (0.084 vs 0.087 bilinear, 0.093 nearest);
* the noise is applied *after* downsampling -- the residual is spatially white
  (lag-1 autocorrelation ~= -0.04);
* speckle is multiplicative Gamma: residual variance fits ``var(r) = d^2/L + s^2``
  and the skew of ``r/d`` measures +0.344, matching Gamma's ``2/sqrt(L)`` at
  L ~= 35;
* per image L ~= 16..150 (median 37) and sigma ~= 0.00..0.15 (median 0.02).
  KLA's own slides quote L=16.86/sigma=0.008594 and L=18.13/sigma=0.001065.

The problem statement says the three degradations "may have been applied in any
order", so this engine randomises the order, the resampling kernel and the noise
levels over ranges deliberately wider than measured.  Everything runs batched on
the GPU so synthetic pairs cost almost nothing during training.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Resampling kernels.  'area' dominates because that is what the real data uses;
# the others exist so the model does not overfit to one specific kernel.
KERNELS = ("area", "bilinear", "bicubic", "gaussian")
KERNEL_WEIGHTS = (0.55, 0.15, 0.15, 0.15)

# Degradation orders.  'ds_speckle_gauss' is the measured real-data order.
ORDERS = ("ds_speckle_gauss", "speckle_ds_gauss", "ds_gauss_speckle")
ORDER_WEIGHTS = (0.6, 0.2, 0.2)

# Empirical noise-parameter distributions.
#
# A first attempt sampled L log-uniformly over [10,200] and sigma uniformly over
# [0,0.18].  That produced synthetic pairs markedly noisier than the real ones
# (residual std 0.137 vs 0.091, skew 0.16 vs 0.35) -- the uniform sigma was ~4x
# too hot and would have biased the model toward over-smoothing.
#
# So instead of guessing a parametric family, we fit (L, sigma) per image on all
# 3200 real pairs and store the resulting quantiles at 5% intervals.  Sampling
# interpolates these tables, which reproduces the true distribution by
# construction, and then applies a random widening factor so the tails extend
# in both directions (the brief warns test noise "levels may vary within a
# similar range").
#
# Measured: L is tightly concentrated (p5=22, p50=36, p95=57), and sigma is
# heavily skewed to zero (21% of images below 0.005, median 0.021, p95 0.073).
L_QUANTILES = (
    5.00, 22.33, 25.18, 26.88, 28.34, 29.76, 31.01, 32.21, 33.68, 34.81, 36.07,
    37.52, 38.98, 40.58, 42.26, 44.08, 45.83, 47.95, 50.84, 57.14, 400.00,
)
SIGMA_QUANTILES = (
    0.00000, 0.00000, 0.00000, 0.00000, 0.00239, 0.00860, 0.01137, 0.01410,
    0.01630, 0.01871, 0.02134, 0.02384, 0.02658, 0.02966, 0.03303, 0.03764,
    0.04270, 0.04932, 0.05630, 0.07269, 0.19655,
)

# Multiplicative widening applied on top of the empirical draw.
L_WIDEN = (0.55, 1.8)          # log-uniform; smaller L = heavier speckle
SIGMA_WIDEN = (0.5, 2.0)       # uniform
L_CLAMP = (8.0, 300.0)
SIGMA_CLAMP = (0.0, 0.25)


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _blur_then_stride(x: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
    """Gaussian pre-filter followed by stride-2 subsampling."""
    radius = 2
    k = _gaussian_kernel1d(sigma, radius, x.device, x.dtype)
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), ky, groups=c)
    return x[:, :, ::2, ::2]


def downsample(x: torch.Tensor, kernel: str) -> torch.Tensor:
    """Halve spatial resolution with the named kernel."""
    if kernel == "area":
        return F.avg_pool2d(x, 2)
    if kernel == "gaussian":
        return _blur_then_stride(x)
    # antialias=True matters: without it bicubic/bilinear alias badly at 2x and
    # would not resemble any sane acquisition model.
    return F.interpolate(
        x, scale_factor=0.5, mode=kernel, align_corners=False, antialias=True
    )


def apply_speckle(x: torch.Tensor, looks: torch.Tensor) -> torch.Tensor:
    """Multiplicative Gamma speckle with mean 1 and variance 1/L.

    ``looks`` is a per-sample tensor broadcastable to ``x``.  Gamma(L, rate=L)
    has mean 1, variance 1/L and skew 2/sqrt(L) -- matching the +0.344 skew
    measured on the real residuals.
    """
    conc = looks.expand_as(x)
    noise = torch.distributions.Gamma(conc, conc).sample()
    return x * noise


def apply_gaussian(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def _sample_choice(weights, n: int, device) -> torch.Tensor:
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    return torch.multinomial(w.expand(n, -1), 1).squeeze(1)


def _sample_quantiles(table, n: int, device) -> torch.Tensor:
    """Draw n samples by linearly interpolating an empirical quantile table."""
    q = torch.tensor(table, device=device, dtype=torch.float32)
    pos = torch.rand(n, device=device) * (len(table) - 1)
    lo = pos.floor().long().clamp(max=len(table) - 2)
    frac = pos - lo
    return q[lo] + frac * (q[lo + 1] - q[lo])


def sample_noise_params(n: int, device, widen: bool = True):
    """Sample per-image (looks, sigma) from the measured empirical distribution."""
    looks = _sample_quantiles(L_QUANTILES, n, device)
    sigma = _sample_quantiles(SIGMA_QUANTILES, n, device)
    if widen:
        u = torch.rand(n, device=device)
        looks = looks * torch.exp(
            u * (math.log(L_WIDEN[1]) - math.log(L_WIDEN[0])) + math.log(L_WIDEN[0])
        )
        sigma = sigma * (
            torch.rand(n, device=device) * (SIGMA_WIDEN[1] - SIGMA_WIDEN[0]) + SIGMA_WIDEN[0]
        )
    looks = looks.clamp(*L_CLAMP).view(n, 1, 1, 1)
    sigma = sigma.clamp(*SIGMA_CLAMP).view(n, 1, 1, 1)
    return looks, sigma


def degrade_batch(gt: torch.Tensor, widen: bool = True) -> torch.Tensor:
    """Degrade a batch of clean images (B,1,H,W) -> (B,1,H/2,W/2).

    The output is deliberately **not** clipped: the real NoisyLR data reaches
    1.66 and dips below 0, and the brief calls that out as intentional.
    Clipping here would create a train/test mismatch on exactly those values.
    """
    b = gt.shape[0]
    device = gt.device
    out = torch.empty(
        (b, gt.shape[1], gt.shape[2] // 2, gt.shape[3] // 2),
        device=device,
        dtype=gt.dtype,
    )

    kern_id = _sample_choice(KERNEL_WEIGHTS, b, device)
    order_id = _sample_choice(ORDER_WEIGHTS, b, device)
    looks, sigma = sample_noise_params(b, device, widen=widen)

    for oi, order in enumerate(ORDERS):
        sel = order_id == oi
        if not bool(sel.any()):
            continue
        idx = sel.nonzero(as_tuple=True)[0]
        x_hr = gt[idx]
        l_i = looks[idx]
        s_i = sigma[idx]
        k_sub = kern_id[idx]

        if order == "speckle_ds_gauss":
            # Speckle applied before a 2x2 average has its variance cut ~4x, so
            # scale L down to keep the *realised* LR noise inside the target
            # band.  Without this correction half the sampled range would land
            # on near-clean images.
            x_hr = apply_speckle(x_hr, torch.clamp(l_i / 4.0, min=1.0))

        # Apply each kernel to its own sub-group.
        x_lr = torch.empty(
            (x_hr.shape[0], x_hr.shape[1], x_hr.shape[2] // 2, x_hr.shape[3] // 2),
            device=device,
            dtype=x_hr.dtype,
        )
        for ki, kern in enumerate(KERNELS):
            ksel = k_sub == ki
            if not bool(ksel.any()):
                continue
            kidx = ksel.nonzero(as_tuple=True)[0]
            x_lr[kidx] = downsample(x_hr[kidx], kern)

        if order == "ds_speckle_gauss":
            x_lr = apply_speckle(x_lr, l_i)
            x_lr = apply_gaussian(x_lr, s_i)
        elif order == "speckle_ds_gauss":
            x_lr = apply_gaussian(x_lr, s_i)
        else:  # ds_gauss_speckle
            x_lr = apply_gaussian(x_lr, s_i)
            x_lr = apply_speckle(x_lr, l_i)

        out[idx] = x_lr

    return out
