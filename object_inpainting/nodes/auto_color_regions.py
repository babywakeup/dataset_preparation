from PIL import Image

from ..color_regions import ColorRegionOutputs, render_color_regions, segment_colors
from .base import Node


class AutoColorRegionsNode(Node):
    """在输入 Mask 内自动分色，固定输出端口，不需要预先指定主色数量。"""

    INPUT_TYPES = {"image": "RGB_IMAGE", "mask": "MASK", "detail": "INT"}
    INPUT_DEFAULTS = {"detail": 1}
    OUTPUT_TYPES = {"regions": "COLOR_REGIONS", "overlay": "RGB_IMAGE", "palette": "RGB_IMAGE", "labels": "LABEL_MAP"}
    CATEGORY = "dataset"
    PRESERVE_MASK_VALUES = True

    def execute(self, image: Image.Image, mask: Image.Image, detail: int = 1) -> ColorRegionOutputs:
        """接收 RGB image、同尺寸 1/L mask 和 0/1/2 精细度，返回四个固定类型输出。

        Args:
            image (PIL.Image.Image): 原始 RGB 图片，不在此节点内缩放。
            mask (PIL.Image.Image): 同尺寸 1/L Mask，>127 表示要分析的前景。
            detail (int): 0 粗、1 中、2 细；默认 1，不代表颜色数量。

        Returns:
            ColorRegionOutputs: regions (ColorRegionSet)、overlay (RGB Image)、
            palette (RGB Image)、labels (L Image，整数编号而非二值 Mask)。
        """
        return render_color_regions(segment_colors(image, mask, detail))
