# 3-minute demo video — script and recording notes

Optional for the portal (the Video link field is not marked required). If you
record one, this is a slide-by-slide script using only numbers that are in the
deck and reproducible from the repo.

---

## Fastest way to record (no extra software)

PowerPoint can narrate and export video by itself:

1. Open `solution_presentation.pptx`
2. **Slide Show → Record → From Beginning**
3. Talk through the slides using the script below; arrow keys advance
4. **Esc** when done
5. **File → Export → Create a Video** → *Full HD (1080p)* → **Create Video** (a few minutes)
6. Upload the MP4 to YouTube as **Unlisted**, or Google Drive with link sharing on
7. Paste that link into the Video link field

Alternatives: **Win + G** (Windows Game Bar) records the screen, or OBS if you
already use it. Neither is better than the above for a slide narration.

Tips: one continuous take is fine — small stumbles do not matter. Speak slightly
slower than feels natural. Aim for **under 3 minutes**; judges watch many of these.

---

## Script (~2 min 50 s)

### Slide 1 — Title (0:00–0:15)

> Our submission for the KLA problem statement: restoring degraded semiconductor
> inspection images. The task is to take a noisy, half-resolution image and
> recover a clean one at full resolution — joint denoising and 2× super-resolution
> in a single network.

### Slide 3 — Degradation model (0:15–0:50) — *the most important slide*

> The brief names three degradations — speckle noise, Gaussian noise and
> downsampling — but does not disclose their parameters or the order they were
> applied in. Rather than guess, we recovered the forward model from the data.
>
> Fitting the residual between the degraded image and the downsampled ground
> truth across all 3,200 pairs told us three things: the downsampling is a 2×2
> area average; the noise is applied *after* downsampling, because the residual
> is spatially white; and the speckle is multiplicative Gamma — the measured skew
> of 0.357 matches Gamma's predicted 0.333 almost exactly. Multiplicative
> Gaussian noise would have been symmetric.
>
> That let us generate synthetic training pairs that match the real distribution
> rather than a plausible guess. Our first attempt, using assumed ranges, was
> 50% noisier than reality and would have taught the model to over-smooth.

### Slide 6 — Architecture (0:50–1:20)

> The network is a NAFNet-style body of 16 blocks running entirely at low
> resolution, with a single PixelShuffle upsample at the end. Keeping the body at
> low resolution is about four times cheaper than a U-Net working at output
> resolution, and throughput is a scored axis.
>
> It is 0.68 million parameters, uses no BatchNorm anywhere — so the evaluator's
> batch size cannot change our outputs, which we verified — and it is fully
> convolutional, so the same weights handle both 128-to-256 and 256-to-512.

### Slide 8 — Validation design (1:20–1:45)

> One detail we think matters. KLA's own slides show that consecutive samples come
> from the same source photograph, roughly two to three crops each. So a random
> split leaks near-duplicate content into training and inflates the score, while a
> contiguous tail slice gives validation sources that appear nowhere in training.
>
> We hold out ten evenly-spaced contiguous blocks: contiguous keeps same-source
> samples together, and the spacing covers the full content diversity. The split
> is frozen in the config.

### Slide 9 — Results (1:45–2:20)

> On the 320-image held-out split, PSNR goes from 22.9 dB for bicubic to 27.5 dB,
> a gain of 4.6 dB. SSIM improves from 0.55 to 0.76, and LPIPS drops from 0.45 to
> 0.39.
>
> Improving on all three at once is the part we would highlight — perceptual
> metrics often move against PSNR, and here they do not. End-to-end inference runs
> at 21.7 milliseconds per image, about 46 images per second on a T4.

### Slide 11 — Limitations (2:20–2:45)

> Being straight about the limits: the model is undertrained. It stopped at step
> 7,500, short of convergence, because we fell back to fp32 after mixed precision
> produced non-finite gradients — a trade that bought a run which finished rather
> than one that diverged, at the cost of throughput.
>
> The dominant failure mode is heavy speckle over fine texture, where the noise
> variance approaches the local signal variance and the model smooths both. That
> is the conservative choice, and the right one here — the brief asks for
> restoration without hallucinating structure, so we deliberately used no GAN.

### Slide 12 — Close (2:45–3:00)

> Everything is reproducible: one command regenerates the degradation analysis,
> another reproduces training, and inference runs from a clean clone with just an
> input and an output directory. Thank you.

---

## If you have only 60 seconds

Slide 3 (degradation model) and slide 9 (results). Those two carry the work.
