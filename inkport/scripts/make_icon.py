"""Build Inkport icon assets from the brand logo (requires Pillow).

assets/brand/inkport-logo.jpg (logo on white)
  -> assets/brand/inkport-logo.png   transparent, trimmed
  -> scripts/icon-source.png         1024x1024 square with padding (input for `pnpm tauri icon`)
  -> public/logo.png                 256x256 for the window/favicon
  -> src/assets/logo.png             256x256 for the in-app header
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "brand" / "inkport-logo.jpg"


TOP = (141, 219, 219)  # darker brand teal (upper shape)
BOTTOM = (176, 232, 231)  # lighter brand teal (lower shape)


def white_to_alpha(img: Image.Image) -> Image.Image:
    """Clean vector-like PNG from a JPEG logo on white: alpha from the red channel's distance to
    white (ramping to fully opaque well inside each shape, which removes JPEG speckle), and every
    pixel recoloured with the exact brand teal of the shape it belongs to."""
    img = img.convert("RGB")
    w, h = img.size
    src = img.load()
    alpha = [[0.0] * w for _ in range(h)]
    label = [[None] * w for _ in range(h)]
    lo, hi = 8.0, 0.8 * (255 - BOTTOM[0])
    for y in range(h):
        for x in range(w):
            r = src[x, y][0]
            a = min(1.0, max(0.0, (255 - r - lo) / (hi - lo)))
            alpha[y][x] = a
            if a >= 1.0:
                label[y][x] = TOP if r < (TOP[0] + BOTTOM[0]) / 2 else BOTTOM
    out = Image.new("RGBA", img.size)
    dst = out.load()
    for y in range(h):
        for x in range(w):
            a = alpha[y][x]
            if a <= 0:
                dst[x, y] = (0, 0, 0, 0)
                continue
            colour = label[y][x]
            if colour is None:  # anti-aliased edge: borrow the nearest solid pixel's colour
                best = None
                for rad in range(1, 9):
                    for dy in range(-rad, rad + 1):
                        for dx in (-rad, rad) if abs(dy) != rad else range(-rad, rad + 1):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < h and 0 <= xx < w and label[yy][xx] is not None:
                                best = label[yy][xx]
                                break
                        if best:
                            break
                    if best:
                        break
                colour = best or BOTTOM
            dst[x, y] = (*colour, round(a * 255))
    return out


def square(img: Image.Image, size: int, pad: float) -> Image.Image:
    inner = round(size * (1 - 2 * pad))
    w, h = img.size
    scale = inner / max(w, h)
    resized = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2), resized)
    return canvas


def main() -> None:
    logo = white_to_alpha(Image.open(SRC))
    logo = logo.crop(logo.getbbox())
    logo.save(ROOT / "assets" / "brand" / "inkport-logo.png")
    square(logo, 1024, 0.08).save(ROOT / "scripts" / "icon-source.png")
    small = square(logo, 256, 0.04)
    small.save(ROOT / "public" / "logo.png")
    small.save(ROOT / "src" / "assets" / "logo.png")
    print("logo", logo.size, "-> icon-source.png, public/logo.png, src/assets/logo.png")


if __name__ == "__main__":
    main()
