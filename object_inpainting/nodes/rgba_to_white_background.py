from PIL import Image

from .base import Node


class RGBAToWhiteBackgroundNode(Node):
    INPUT_TYPES = {"rgba": "RGBA_IMAGE"}
    OUTPUT_TYPES = {"rgb": "RGB_IMAGE"}
    CATEGORY = "dataset"

    def execute(self, rgba: Image.Image) -> dict:
        if not isinstance(rgba, Image.Image):
            raise ValueError("RGBA 输入必须是图片对象。")

        rgba = rgba.convert("RGBA")
        white_background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(white_background, rgba)
        return {"rgb": composed.convert("RGB")}
