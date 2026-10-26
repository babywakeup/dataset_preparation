from PIL import Image


def align_square(
    target: Image.Image, object_mask: Image.Image, edit_mask: Image.Image, resolution: int
) -> tuple[Image.Image, Image.Image, Image.Image]:
    bbox = edit_mask.getbbox()
    if bbox is None:
        raise ValueError("Edit mask is empty.")
    width, height = target.size
    side = min(width, height)
    x0, y0, x1, y1 = bbox
    if x1 - x0 > side or y1 - y0 > side:
        raise ValueError("Object cannot fit into an aligned square crop without truncation.")
    left = max(0, min(width - side, (x0 + x1 - side) // 2))
    top = max(0, min(height - side, (y0 + y1 - side) // 2))
    crop_box = (left, top, left + side, top + side)
    size = (resolution, resolution)
    return (
        target.crop(crop_box).resize(size, Image.Resampling.LANCZOS),
        object_mask.crop(crop_box).resize(size, Image.Resampling.NEAREST),
        edit_mask.crop(crop_box).resize(size, Image.Resampling.NEAREST),
    )
