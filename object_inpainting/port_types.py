"""公共端口类别和 Python 数据类型；图像本体始终是 PIL Image。"""

from enum import Enum
from pathlib import Path
from typing import Annotated, TypeAlias

from PIL import Image


class PortType(str, Enum):
    """图执行 / Studio 使用的端口类别，值保持 JSON 可序列化。"""

    RGB_IMAGE = "RGB_IMAGE"
    RGBA_IMAGE = "RGBA_IMAGE"
    MASK = "MASK"
    IMAGE = "IMAGE"
    IMAGE_LIST = "IMAGE_LIST"
    LABEL_MAP = "LABEL_MAP"
    COLOR_REGIONS = "COLOR_REGIONS"
    BBOX = "BBOX"
    PATH = "PATH"
    STRING = "STRING"
    INT = "INT"
    INT_LIST = "INT_LIST"
    FLOAT = "FLOAT"
    CONFIG = "CONFIG"
    OBJECT_RECORD = "OBJECT_RECORD"
    PREPARED_SAMPLE = "PREPARED_SAMPLE"
    JSON_RECORD = "JSON_RECORD"


class ImageRole(str, Enum):
    """同为 RGB 的数据仍需区分用途；不是额外的像素包装对象。"""

    RGB = "rgb_image"
    MASKED_RGB = "masked_rgb_image"
    REFERENCE_RGB = "reference_rgb_image"
    OVERLAY_RGB = "overlay_rgb_image"
    PALETTE_RGB = "palette_rgb_image"
    COLOR_RGB = "color_rgb_image"


RGBImage: TypeAlias = Annotated[Image.Image, PortType.RGB_IMAGE, ImageRole.RGB]
MaskedRGBImage: TypeAlias = Annotated[Image.Image, PortType.RGB_IMAGE, ImageRole.MASKED_RGB]
ReferenceRGBImage: TypeAlias = Annotated[Image.Image, PortType.RGB_IMAGE, ImageRole.REFERENCE_RGB]
OverlayRGBImage: TypeAlias = Annotated[Image.Image, PortType.RGB_IMAGE, ImageRole.OVERLAY_RGB]
PaletteRGBImage: TypeAlias = Annotated[Image.Image, PortType.RGB_IMAGE, ImageRole.PALETTE_RGB]
ColorRGBImage: TypeAlias = Annotated[Image.Image, PortType.RGB_IMAGE, ImageRole.COLOR_RGB]
RGBAImage: TypeAlias = Annotated[Image.Image, PortType.RGBA_IMAGE]
MaskImage: TypeAlias = Annotated[Image.Image, PortType.MASK]
LabelMap: TypeAlias = Annotated[Image.Image, PortType.LABEL_MAP]
ImageList: TypeAlias = Annotated[list[RGBImage], PortType.IMAGE_LIST]
BoundingBox: TypeAlias = Annotated[tuple[int, int, int, int] | list[int], PortType.BBOX]
FloatBoundingBox: TypeAlias = Annotated[tuple[float, float, float, float] | list[float] | tuple[int, int, int, int] | list[int], PortType.BBOX]
FilePath: TypeAlias = Annotated[str | Path, PortType.PATH]
PathString: TypeAlias = Annotated[str, PortType.PATH]
JsonRecord: TypeAlias = dict[str, object]
# 上传路径在 Studio 层解码；FLUX 也支持直接传路径。空字符串是合法的可选输入。
OptionalReferenceImage: TypeAlias = Annotated[Image.Image | str | Path | None, PortType.RGB_IMAGE, ImageRole.REFERENCE_RGB]
OptionalImage: TypeAlias = Annotated[Image.Image | str | None, PortType.IMAGE]
OptionalImageList: TypeAlias = Annotated[list[Image.Image] | None, PortType.IMAGE_LIST]
RegionIndices: TypeAlias = Annotated[list[int] | None, PortType.INT_LIST]
