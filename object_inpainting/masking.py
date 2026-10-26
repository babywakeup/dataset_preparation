from PIL import Image, ImageFilter


def expand_mask(object_mask: Image.Image, radius: int) -> Image.Image:
    if radius == 0:
        return object_mask.copy()
    return object_mask.filter(ImageFilter.MaxFilter(2 * radius + 1))


def make_source(target: Image.Image, edit_mask: Image.Image) -> Image.Image:
    return Image.composite(Image.new("RGB", target.size, (128, 128, 128)), target, edit_mask)
