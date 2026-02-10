#!/usr/bin/env python3
"""Build a GIF from sample images in a folder, with step number on each frame.

Step number is parsed from the filename (e.g. drift_step0010000.png -> 10000).
Usage:
    python samples_to_gif.py <folder>
    python samples_to_gif.py outputs/drift/20260210_010550/samples
"""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Match step number in filenames like drift_step0010000.png or step_0001000.png
STEP_PATTERN = re.compile(r"step[_\s]*(\d+)", re.IGNORECASE)


def get_font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def parse_step_from_path(path: Path) -> int | None:
    """Extract step number from filename. Returns None if no match."""
    m = STEP_PATTERN.search(path.name)
    return int(m.group(1)) if m else None


def collect_image_paths(folder: Path):
    """Return list of (path, step) sorted by step. Only includes files with a parseable step."""
    pairs = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        step = parse_step_from_path(p)
        if step is not None:
            pairs.append((p, step))
    pairs.sort(key=lambda x: x[1])
    return pairs


def add_step_label(img: Image.Image, step: int, corner: str = "top-left") -> Image.Image:
    """Draw step number on a copy of the image. corner: top-left, top-right, bottom-left, bottom-right."""
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    # Font size scales roughly with image size
    size = max(12, min(48, out.width // 16))
    font = get_font(size)
    text = str(step)
    # Padding and background for readability
    pad = max(4, size // 4)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if corner == "top-left":
        x, y = pad, pad
    elif corner == "top-right":
        x, y = out.width - tw - pad * 2, pad
    elif corner == "bottom-left":
        x, y = pad, out.height - th - pad * 2
    else:  # bottom-right
        x, y = out.width - tw - pad * 2, out.height - th - pad * 2
    draw.rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        fill=(0, 0, 0),
        outline=(255, 255, 255),
        width=max(1, size // 12),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return out


def samples_to_gif(
    folder: Path,
    output_path: Path | None = None,
    *,
    duration: int = 200,
    corner: str = "top-left",
) -> None:
    """Build a GIF from sample images in folder with step labels. Callable from training scripts."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        return
    pairs = collect_image_paths(folder)
    if not pairs:
        return
    out_path = Path(output_path).resolve() if output_path else folder / "samples.gif"
    frames = []
    for path, step in pairs:
        img = Image.open(path)
        frame = add_step_label(img, step, corner=corner)
        frames.append(frame)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build a GIF from sample images with step numbers on each frame."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing sample images (e.g. .../samples)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output GIF path (default: <folder>/samples.gif)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=200,
        help="Frame duration in ms (default: 200)",
    )
    parser.add_argument(
        "--corner",
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="top-left",
        help="Corner for step label (default: top-left)",
    )
    args = parser.parse_args()
    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")
    pairs = collect_image_paths(folder)
    if not pairs:
        raise SystemExit(
            f"No images with parseable step number found in {folder}. "
            "Expected filenames like drift_step0010000.png"
        )
    out_path = args.output or folder / "samples.gif"
    samples_to_gif(folder, out_path, duration=args.duration, corner=args.corner)
    print(f"Saved {out_path} ({len(pairs)} frames, {args.duration} ms/frame)")


if __name__ == "__main__":
    main()
