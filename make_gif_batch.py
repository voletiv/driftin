"""Generate a batch GIF: 8 images being generated simultaneously.

DDPM side: all 8 denoise together over 50 steps.
Drift side: all 8 appear instantly in 1 step.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from drifting_vs_diffusion.config import UNetConfig
from drifting_vs_diffusion.models.unet import UNet
from drifting_vs_diffusion.training.ddpm_utils import DDPMSchedule

DEVICE = torch.device("cuda")
UPSCALE = 4  # 32 -> 128 each
CELL = 32 * UPSCALE  # 128px per image
N_IMGS = 8  # 2x4 grid

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def load_model(ckpt_path):
    cfg = UNetConfig()
    model = UNet(
        in_ch=cfg.in_ch, out_ch=cfg.out_ch, base_ch=cfg.base_ch,
        ch_mult=cfg.ch_mult, num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions, dropout=0.0,
        num_heads=cfg.num_heads,
    ).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("ema", ckpt["model"])
    clean = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean)
    return model


def tensor_to_pil(t):
    t = t.clamp(-1, 1)
    t = (t + 1) / 2
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.resize((CELL, CELL), Image.NEAREST)


def make_grid(tensors, ncol=4):
    """Arrange PIL images in a grid."""
    pils = [tensor_to_pil(t) for t in tensors]
    nrow = (len(pils) + ncol - 1) // ncol
    w = ncol * CELL + (ncol - 1) * 4
    h = nrow * CELL + (nrow - 1) * 4
    grid = Image.new("RGB", (w, h), (15, 15, 15))
    for i, pil in enumerate(pils):
        r, c = divmod(i, ncol)
        grid.paste(pil, (c * (CELL + 4), r * (CELL + 4)))
    return grid


def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def ddim_intermediates(model, schedule, z, n_steps=50):
    intermediates = [z.clone()]
    T = schedule.T
    step_seq = list(range(0, T, T // n_steps))
    step_seq_next = [-1] + step_seq[:-1]
    alphas_bar = schedule.alpha_bar
    x = z.clone()
    with torch.no_grad():
        for i in reversed(range(len(step_seq))):
            t_cur = step_seq[i]
            t_next = step_seq_next[i]
            t_batch = torch.full((x.shape[0],), t_cur, device=x.device, dtype=torch.long)
            pred_noise = model(x, t_batch)
            ab_cur = alphas_bar[t_cur]
            x0_pred = (x - (1 - ab_cur).sqrt() * pred_noise) / ab_cur.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            if t_next < 0:
                x = x0_pred
            else:
                ab_next = alphas_bar[t_next]
                x = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * pred_noise
            intermediates.append(x.clone())
    return intermediates


def make_frame(ddpm_grid, drift_grid, step, total, elapsed_ms, drift_ms,
               drift_done=True, done=False):
    """Vertical layout: DDPM top, Drift bottom, padded to square."""
    gw, gh = ddpm_grid.size
    MARGIN = 20
    LABEL_H = 26
    INFO_H = 42
    GAP = 14
    TITLE_H = 38
    FOOTER_H = 42

    content_w = gw + MARGIN * 2
    content_h = TITLE_H + (LABEL_H + gh + INFO_H) * 2 + GAP + FOOTER_H

    # Pad to square
    S = max(content_w, content_h)
    frame = Image.new("RGB", (S, S), (15, 15, 15))
    draw = ImageDraw.Draw(frame)

    # Offset to center content in the square
    ox = (S - content_w) // 2
    oy = (S - content_h) // 2

    font_title = get_font(18)
    font_label = get_font(14)
    font_time = get_font(12)
    font_big = get_font(22)

    cx = S // 2

    # Title
    draw.text((cx, oy + 8), "Batch Generation: DDPM vs Drifting",
              fill=(230, 230, 230), font=font_title, anchor="mt")

    # --- Top row: DDPM ---
    y_ddpm_label = oy + TITLE_H
    y_ddpm_img = y_ddpm_label + LABEL_H
    x_img = ox + MARGIN
    color_ddpm = (100, 200, 255)

    draw.text((cx, y_ddpm_label + 2), "DDPM (50 steps)",
              fill=color_ddpm, font=font_label, anchor="mt")
    draw.rectangle([x_img - 2, y_ddpm_img - 2,
                    x_img + gw + 1, y_ddpm_img + gh + 1],
                   outline=color_ddpm, width=2)
    frame.paste(ddpm_grid, (x_img, y_ddpm_img))

    step_text = f"Step {step}/{total}" if not done else f"Done! ({total} steps)"
    draw.text((cx, y_ddpm_img + gh + 4), step_text,
              fill=(180, 180, 180), font=font_time, anchor="mt")
    draw.text((cx, y_ddpm_img + gh + 20), f"{elapsed_ms:.0f} ms",
              fill=(180, 180, 180), font=font_time, anchor="mt")

    # --- Bottom row: Drift ---
    y_drift_label = y_ddpm_img + gh + INFO_H + GAP
    y_drift_img = y_drift_label + LABEL_H
    color_drift = (120, 230, 140)

    draw.text((cx, y_drift_label + 2), "Drifting (1 step)",
              fill=color_drift, font=font_label, anchor="mt")
    draw.rectangle([x_img - 2, y_drift_img - 2,
                    x_img + gw + 1, y_drift_img + gh + 1],
                   outline=color_drift, width=2)
    frame.paste(drift_grid, (x_img, y_drift_img))

    if drift_done:
        draw.text((cx, y_drift_img + gh + 4), "Done! (1 step)",
                  fill=color_drift, font=font_time, anchor="mt")
        draw.text((cx, y_drift_img + gh + 20), f"{drift_ms:.1f} ms",
                  fill=color_drift, font=font_time, anchor="mt")
    else:
        draw.text((cx, y_drift_img + gh + 4), "Waiting...",
                  fill=(180, 180, 180), font=font_time, anchor="mt")

    # Footer: speedup
    if done:
        speedup = elapsed_ms / drift_ms
        draw.text((cx, oy + content_h - 6),
                  f"Drifting: {speedup:.0f}x faster",
                  fill=color_drift, font=font_big, anchor="mb")

    return frame


def main():
    print("Loading models...")
    ddpm = load_model("outputs/ddpm_h100/checkpoints/ddpm_final.pt")
    drift = load_model("outputs/drift_dinov2/checkpoints/drift_final.pt")
    schedule = DDPMSchedule(T=1000).to(DEVICE)

    torch.manual_seed(42)
    z = torch.randn(N_IMGS, 3, 32, 32, device=DEVICE)

    # Benchmark
    print("Benchmarking...")
    with torch.no_grad():
        for _ in range(3):
            _ = drift(z)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        drift_out = drift(z)
    end.record()
    torch.cuda.synchronize()
    drift_ms = start.elapsed_time(end)

    with torch.no_grad():
        for _ in range(2):
            _ = ddim_intermediates(ddpm, schedule, z.clone())
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        intermediates = ddim_intermediates(ddpm, schedule, z.clone())
    end.record()
    torch.cuda.synchronize()
    ddpm_ms = start.elapsed_time(end)
    per_step = ddpm_ms / 50

    print(f"Drift: {drift_ms:.1f} ms for {N_IMGS} images")
    print(f"DDPM: {ddpm_ms:.1f} ms for {N_IMGS} images")

    drift_grid = make_grid(drift_out)
    noise_grid = make_grid(intermediates[0])

    print("Generating frames...")
    frames = []

    # Both start as noise
    for _ in range(3):
        f = make_frame(noise_grid, noise_grid, 0, 50, 0, drift_ms, drift_done=False)
        frames.append(f)

    # Step 1: drift snaps to final, DDPM begins
    f = make_frame(make_grid(intermediates[1]), drift_grid,
                   1, 50, per_step, drift_ms, drift_done=True)
    frames.append(f)

    # Steps 2-50: DDPM continues, drift frozen
    for step in range(2, 51):
        f = make_frame(make_grid(intermediates[step]), drift_grid,
                       step, 50, step * per_step, drift_ms,
                       drift_done=True, done=(step == 50))
        frames.append(f)

    # Hold final
    for _ in range(20):
        frames.append(frames[-1].copy())

    # Global palette to prevent per-frame requantization flicker
    print("Quantizing to global palette...")
    palette_img = frames[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette_img, dither=Image.Dither.NONE) for f in frames]

    gif_path = "outputs/drifting_vs_diffusion_batch.gif"
    quantized[0].save(gif_path, save_all=True, append_images=quantized[1:],
                      duration=80, loop=0)
    print(f"Saved: {gif_path} ({len(frames)} frames, {os.path.getsize(gif_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
