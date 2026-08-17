"""NAFNet-style restoration network with a PixelShuffle x2 head.

Design notes (rationale for the write-up):

* The whole body runs at **LR resolution** and upsamples once at the very end.
  That is ~4x cheaper than a U-Net operating at output resolution, and
  end-to-end throughput is a scored axis.
* NAF blocks (depthwise conv + SimpleGate + simplified channel attention) give
  near-transformer restoration quality with pure convolution.  Attention-based
  backbones (SwinIR/Restormer) buy very little at 128x128 tile sizes and cost a
  lot of H100 throughput.
* **No BatchNorm anywhere.**  All normalisation is channel-wise LayerNorm, so
  the model is batch-size independent -- the evaluator's batch size cannot
  change our outputs.
* Fully convolutional with no fixed-size assumption, so the same weights handle
  128->256 and 256->512.  The brief says test data may include either.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor.

    Always computed in fp32, even under autocast.  Variance squares the
    activations, and fp16 accumulates that reduction in fp16 too -- so a mean of
    x^2 overflows to ``inf`` once activations merely reach ~100, far below
    fp16's 65504 ceiling.  The normalisation then yields inf/nan, which
    propagates into the loss and (via one optimizer step) corrupts the weights
    permanently.  This is what took down two training runs; the fp32 cast costs
    almost nothing and removes the failure mode entirely.
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
            y = y * self.weight[None, :, None, None].float() + self.bias[None, :, None, None].float()
        return y.to(x.dtype)


class SimpleGate(nn.Module):
    """Split channels in half and multiply -- NAFNet's activation-free gate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw = c * dw_expand
        ffn = c * ffn_expand

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        # Simplified channel attention: global context -> per-channel gain.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        # Learnable residual scales, initialised at zero so each block starts as
        # an identity map.  This is what makes a 10-block stack trainable at
        # lr=1e-3 from scratch without warm-up tricks.
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        return x + y * self.gamma


class RestoreNet(nn.Module):
    """Joint denoise + x2 super-resolution.

    Args:
        width: feature width of the body.
        blocks: number of NAF blocks.
        in_ch: input channels (1 -- the data is grayscale).
        global_skip: add a bilinear-upsampled copy of the input to the output.
            This carries some input noise through, which the final conv must
            cancel, but it lets the network learn a residual instead of the
            whole image and converges far faster.  Deliberate trade-off given a
            single-shot training budget; ablation lives in results/.
    """

    def __init__(
        self,
        width: int = 64,
        blocks: int = 16,
        in_ch: int = 1,
        global_skip: bool = True,
    ):
        super().__init__()
        self.global_skip = global_skip
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(blocks)])
        self.body_tail = nn.Conv2d(width, width, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tail = nn.Conv2d(width, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.intro(x)
        feat = feat + self.body_tail(self.body(feat))
        feat = self.upsample(feat)
        out = self.tail(feat)
        if self.global_skip:
            out = out + F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
        return out


def build_model(cfg: dict | None = None) -> RestoreNet:
    cfg = cfg or {}
    return RestoreNet(
        width=cfg.get("width", 64),
        blocks=cfg.get("blocks", 16),
        in_ch=cfg.get("in_ch", 1),
        global_skip=cfg.get("global_skip", True),
    )


if __name__ == "__main__":
    m = build_model()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.2f}M")
    for size in (64, 128, 256):
        y = m(torch.randn(1, 1, size, size))
        print(f"  {size}x{size} -> {tuple(y.shape[-2:])}")
