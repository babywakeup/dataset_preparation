import numpy as np
from PIL import Image

from ..node_contracts import ColorSimilarityInputs, ColorSimilarityOutputs
from ..port_types import RGBImage
from .base import Node


class ColorSimilarityNode(Node):
    """基于 skimage.color.deltaE_ciede2000 评估两张纯色 RGB 图片的颜色相似度得分。"""

    INPUT_SCHEMA = ColorSimilarityInputs
    OUTPUT_SCHEMA = ColorSimilarityOutputs
    CATEGORY = "dataset"

    def execute(self, image1: RGBImage, image2: RGBImage) -> ColorSimilarityOutputs:
        """接收两张纯色 RGB 图片，提取代表颜色并计算 CIEDE2000 颜色差异得分。

        Args:
            image1 (PIL.Image.Image): 第一张纯色 RGB 图片。
            image2 (PIL.Image.Image): 第二张纯色 RGB 图片。

        Returns:
            ColorSimilarityOutputs: 包含 score (float)，即 CIEDE2000 色差得分。
            值域为 [0, ~119.03]（在标准 sRGB 颜色范围内），完全相同时为 0.0，值越小越相似。
        """
        if not isinstance(image1, Image.Image) or not isinstance(image2, Image.Image):
            raise ValueError("image1 与 image2 必须为 PIL Image 图片。")

        if image1.width == 0 or image1.height == 0 or image2.width == 0 or image2.height == 0:
            raise ValueError("输入图片尺寸不可为空。")

        try:
            from skimage.color import deltaE_ciede2000, rgb2lab
        except ImportError as error:
            raise ValueError("颜色相似度计算需要 scikit-image，请安装相关依赖。") from error

        # 纯色图片无需创建全图大数组或求空间均值，直接获取单像素 RGB 并用 uint8 构造单像素输入
        rgb1 = image1.convert("RGB").getpixel((0, 0))
        rgb2 = image2.convert("RGB").getpixel((0, 0))

        lab1 = rgb2lab(np.array([[rgb1]], dtype=np.uint8))
        lab2 = rgb2lab(np.array([[rgb2]], dtype=np.uint8))

        score = float(deltaE_ciede2000(lab1, lab2)[0, 0])
        return {"score": score}
