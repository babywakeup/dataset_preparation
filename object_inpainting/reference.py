import math

from PIL import Image, ImageOps


def extract_reference(
    target: Image.Image, object_mask: Image.Image, bbox: tuple[int, int, int, int], resolution: int, padding: float
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    margin = math.ceil(max(x1 - x0, y1 - y0) * padding)
    crop_box = (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(target.width, x1 + margin),
        min(target.height, y1 + margin),
    )
    foreground = Image.composite(target, Image.new("RGB", target.size, "white"), object_mask).crop(crop_box)
    foreground = ImageOps.contain(foreground, (resolution, resolution), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (resolution, resolution), "white")
    canvas.paste(foreground, ((resolution - foreground.width) // 2, (resolution - foreground.height) // 2))
    return canvas
