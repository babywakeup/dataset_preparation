from ..node_contracts import MaskToRGBAInputs, MaskToRGBAOutputs
from ..port_types import MaskImage, RGBImage
from .base import Node


class MaskToRGBANode(Node):
    INPUT_SCHEMA = MaskToRGBAInputs
    OUTPUT_SCHEMA = MaskToRGBAOutputs
    CATEGORY = "dataset"
    PRESERVE_MASK_VALUES = True

    def execute(self, image: RGBImage, mask: MaskImage) -> MaskToRGBAOutputs:
        if image.size != mask.size:
            raise ValueError("原图与 Mask 尺寸必须一致。")
        if mask.mode not in {"1", "L"}:
            raise ValueError("Mask 必须是黑白或 8 位灰度图片。")

        rgba = image.convert("RGBA")
        rgba.putalpha(mask.convert("L"))
        rgba.save("temp_rgba.png", format="PNG")  # 临时保存以确保 alpha 通道正确
        return {"rgba": rgba}
