from PIL import Image

from .base import Node


class CropNode(Node):
    INPUT_TYPES = {
        "image": "RGB_IMAGE",
        "bbox": "BBOX",
    }
    OUTPUT_TYPES = {
        "cropped": "RGB_IMAGE",
    }
    CATEGORY = "dataset"

    def execute(self, image: Image.Image, bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("裁剪框必须是 [左, 上, 右, 下]。")

        if any(type(value) is not int for value in bbox):
            raise ValueError("裁剪坐标必须是整数。")

        left, top, right, bottom = bbox
        width, height = image.size

        if not (0 <= left < right <= width):
            raise ValueError(f"横向范围必须满足 0 ≤ 左 < 右 ≤ {width}。")

        if not (0 <= top < bottom <= height):
            raise ValueError(f"纵向范围必须满足 0 ≤ 上 < 下 ≤ {height}。")

        return {
            "cropped": image.crop((left, top, right, bottom)),
        }
