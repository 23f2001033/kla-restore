# Degradation analysis

Fitted on all 3200 provided training pairs (GT 256x256, NoisyLR 128x128).

## 1. Downsampling kernel

| kernel | residual std | lag-1 autocorr of residual |
|---|---|---|
| area | 0.0910 | -0.053 |
| bilinear | 0.0940 | -0.056 |
| bicubic | 0.0924 | -0.057 |
| nearest | 0.1006 | -0.027 |

**Kernel = area** (lowest residual).  The near-zero autocorrelation shows the noise is spatially white, i.e. applied *after* downsampling -- noise applied at high resolution and then box-averaged would be visibly correlated.

## 2. Noise model

Per-image least squares of `var(r) = d^2/L + sigma^2`.  The presence of a `d^2` term is the signature of multiplicative (speckle) noise; the constant term is the additive Gaussian component.

| quantile | speckle looks L | Gaussian sigma |
|---|---|---|
| p5 | 22.3 | 0.0000 |
| p10 | 25.2 | 0.0000 |
| p25 | 29.8 | 0.0086 |
| p50 | 36.1 | 0.0213 |
| p75 | 44.1 | 0.0376 |
| p90 | 50.8 | 0.0563 |
| p95 | 57.1 | 0.0727 |
| p99 | 89.1 | 0.1304 |

Fraction of images with sigma < 0.005: **21.3%**

Mean skew of `r/d` = **+0.357**.  Gamma-distributed speckle with L looks has skew `2/sqrt(L)`; at the median L=36 that predicts +0.333 -- a close match, confirming **multiplicative Gamma speckle** rather than, say, multiplicative Gaussian (which is symmetric).

## 3. Baseline anchors

| method | PSNR |
|---|---|
| bicubic upsample of NoisyLR (do-nothing baseline) | 23.18 dB |
| bicubic upsample of *clean* LR (perfect-denoise ceiling) | 32.25 dB |
| pixel-replicate of clean LR | 30.10 dB |

A learned model should clear the do-nothing baseline by a wide margin and should beat the perfect-denoise ceiling, since it also learns super-resolution rather than interpolating.

## 4. Synthetic engine fidelity

| source | range | residual std | skew of r/d | per-image residual std (p10/p50/p90) |
|---|---|---|---|---|
| real pairs | [-0.13, +1.90] | 0.0852 | +0.367 | 0.045 / 0.079 / 0.114 |
| synthetic (exact) | [-0.72, +2.00] | 0.0945 | +0.299 | 0.050 / 0.081 / 0.135 |
| synthetic (widened) | [-1.01, +2.47] | 0.1010 | +0.483 | 0.045 / 0.086 / 0.145 |

The synthetic generator reproduces the real noise statistics closely, and the widened variant extends the tails in both directions as intended for OOD robustness.
