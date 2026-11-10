"""CPU-only images and scores for exercising conditional loops."""

from random import randrange
import time
from typing import TypedDict

from PIL import Image, ImageStat

from ..port_types import RGBImage
from .base import Node

MARKER_SIZE = 32


def validate_image(image: RGBImage) -> None:
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("image 必须是 RGB 模式的 PIL 图片。")
    if image.width == 0 or image.height == 0:
        raise ValueError("image 尺寸不可为空。")


class LoopDemoImageInputs(TypedDict):
    seed: int
    image: RGBImage


class LoopDemoImageOutputs(TypedDict):
    image: RGBImage
    value: int


class LoopDemoScoreInputs(TypedDict):
    image: RGBImage


class LoopDemoScoreOutputs(TypedDict):
    score: float


class LoopDemoImageNode(Node):
    """Mark an optional source image with a seed-selected color for loop tests."""

    INPUT_SCHEMA = LoopDemoImageInputs
    OUTPUT_SCHEMA = LoopDemoImageOutputs
    INPUT_DEFAULTS = {"seed": 0}
    OPTIONAL_INPUTS = {"image"}

    def execute(self, seed: int = 0, image: RGBImage = None) -> LoopDemoImageOutputs:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < -1:
            raise ValueError("seed 必须是非负整数或 -1（随机种子）。")
        if seed == -1:
            seed = randrange(2**63)
        if image is not None:
            validate_image(image)
        time.sleep(10)  # Simulate a time-consuming operation
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (122, 122, 122), (255, 255, 0), (0, 255, 255), (255, 0, 255))
        output = image.copy() if image is not None else Image.new("RGB", (MARKER_SIZE, MARKER_SIZE))
        output.paste(colors[seed % len(colors)], (0, 0, min(MARKER_SIZE, output.width), min(MARKER_SIZE, output.height)))
        return {"image": output, "value": seed}


class LoopDemoScoreNode(Node):
    """Score blue intensity in the top-left demo marker, from 0 to 255."""

    INPUT_SCHEMA = LoopDemoScoreInputs
    OUTPUT_SCHEMA = LoopDemoScoreOutputs

    def execute(self, image: RGBImage) -> LoopDemoScoreOutputs:
        validate_image(image)
        marker = image.crop((0, 0, min(MARKER_SIZE, image.width), min(MARKER_SIZE, image.height)))
        return {"score": float(ImageStat.Stat(marker).mean[2])}
