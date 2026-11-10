from PIL import Image

from ..node_contracts import RGBAToWhiteBackgroundInputs, RGBAToWhiteBackgroundOutputs
from ..port_types import RGBAImage
from .base import Node


class RGBAToWhiteBackgroundNode(Node):
    INPUT_SCHEMA = RGBAToWhiteBackgroundInputs
    OUTPUT_SCHEMA = RGBAToWhiteBackgroundOutputs
    CATEGORY = "dataset"

    def execute(self, rgba: RGBAImage) -> RGBAToWhiteBackgroundOutputs:
        if not isinstance(rgba, Image.Image):
            raise ValueError("RGBA 输入必须是图片对象。")

        rgba = rgba.convert("RGBA")
        white_background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(white_background, rgba)
        return {"rgb": composed.convert("RGB")}
