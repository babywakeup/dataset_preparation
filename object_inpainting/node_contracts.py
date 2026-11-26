"""节点入口 / 出口合同。TypedDict 保持字典执行协议，字段注解生成端口表。"""

from pathlib import Path
from typing import Annotated, TypedDict, get_args, get_origin, get_type_hints, is_typeddict

from .color_regions import (
    ColorRegionOutputs as ColorRegionOutputs,
    ColorRegionSet,
    SelectedRegionOutputs as SelectedRegionOutputs,
)
from .config import PipelineConfig
from .port_types import (
    BoundingBox, ColorRGBImage, FilePath, FloatBoundingBox, ImageList, JsonRecord, MaskImage,
    MaskedRGBImage, OptionalImage, OptionalImageList, OptionalReferenceImage,
    PathString, PortType, ReferenceRGBImage, RegionIndices, RGBAImage, RGBImage,
)
from .quality import PreparedSample
from .records import ObjectRecord


class RecordInputInputs(TypedDict):
    record: ObjectRecord


class RecordInputOutputs(TypedDict):
    target_path: FilePath
    mask_path: FilePath
    object_name: str


class LoadTargetInputs(TypedDict):
    path: FilePath


class LoadTargetOutputs(TypedDict):
    target: RGBImage


class LoadInstanceMaskInputs(TypedDict):
    path: FilePath
    target: RGBImage


class LoadInstanceMaskOutputs(TypedDict):
    object_mask: MaskImage


class SelectObjectInputs(TypedDict):
    object_mask: MaskImage
    config: PipelineConfig


class SelectObjectOutputs(TypedDict):
    object_mask: MaskImage
    bbox: BoundingBox


class ExtractReferenceInputs(TypedDict):
    target: RGBImage
    object_mask: MaskImage
    bbox: BoundingBox
    config: PipelineConfig


class ExtractReferenceOutputs(TypedDict):
    reference: ReferenceRGBImage


class ExpandMaskInputs(TypedDict):
    object_mask: MaskImage
    config: PipelineConfig


class ExpandMaskOutputs(TypedDict):
    edit_mask: MaskImage


class AlignSquareInputs(TypedDict):
    target: RGBImage
    object_mask: MaskImage
    edit_mask: MaskImage
    config: PipelineConfig


class AlignSquareOutputs(TypedDict):
    target: RGBImage
    object_mask: MaskImage
    edit_mask: MaskImage


class GrayMaskInputs(TypedDict):
    target: RGBImage
    edit_mask: MaskImage


class GrayMaskOutputs(TypedDict):
    source: MaskedRGBImage


class BuildPromptInputs(TypedDict):
    object_name: str


class BuildPromptOutputs(TypedDict):
    prompt: str


class AssembleSampleInputs(TypedDict):
    target: RGBImage
    source: MaskedRGBImage
    edit_mask: MaskImage
    reference: ReferenceRGBImage
    prompt: str


class SampleOutputs(TypedDict):
    sample: PreparedSample


class ValidateSampleInputs(TypedDict):
    sample: PreparedSample
    object_mask: MaskImage
    config: PipelineConfig


class SaveSampleInputs(TypedDict):
    sample: PreparedSample
    record: ObjectRecord
    config: PipelineConfig
    output: FilePath


class SaveSampleOutputs(TypedDict):
    entry: JsonRecord
    manifest_path: Path
    split: str


class WriteJsonlInputs(TypedDict):
    entry: JsonRecord
    path: FilePath


class WriteJsonlOutputs(TypedDict):
    entry: JsonRecord


class SavePreviewInputs(TypedDict):
    sample: PreparedSample
    path: FilePath


class SavePreviewOutputs(TypedDict):
    path: Path


class CropInputs(TypedDict):
    image: RGBImage
    bbox: BoundingBox


class CropOutputs(TypedDict):
    cropped: RGBImage


class SAM2MaskInputs(TypedDict):
    image: RGBImage
    bbox: FloatBoundingBox


class SAM2MaskOutputs(TypedDict):
    object_mask: MaskImage
    quality: str


class SAM3TextMaskInputs(TypedDict):
    image: RGBImage
    text: str
    threshold: float
    mask_threshold: float


class SAM3TextMaskOutputs(TypedDict):
    object_mask: MaskImage
    bbox: BoundingBox
    quality: str


class MaskDilationInputs(TypedDict):
    mask: MaskImage
    dilation_steps: int
    dilation_kernel: int


class MaskDilationOutputs(TypedDict):
    mask: MaskImage
    bbox: BoundingBox
    quality: str


class MaskToRGBAInputs(TypedDict):
    image: RGBImage
    mask: MaskImage


class MaskToRGBAOutputs(TypedDict):
    rgba: RGBAImage


class RGBAToWhiteBackgroundInputs(TypedDict):
    rgba: RGBAImage


class RGBAToWhiteBackgroundOutputs(TypedDict):
    rgb: ReferenceRGBImage


class FluxKleinReferenceInputs(TypedDict):
    prompt: str
    reference_1: OptionalReferenceImage
    reference_2: OptionalReferenceImage
    reference_3: OptionalReferenceImage
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int


class FluxKleinReferenceOutputs(TypedDict):
    image: RGBImage


class FluxRandomRecolorInputs(TypedDict):
    target: RGBImage
    mask: MaskedRGBImage
    seed: int
    num_inference_steps: int
    guidance_scale: float
    preserve_unmasked: int


class RandomRecolorOutputs(TypedDict):
    source: RGBImage
    masked_source: MaskedRGBImage
    mask: MaskImage
    sampled_color: ColorRGBImage
    color_name: str
    actual_seed: int
    generation_prompt: str


class SaveImagesInputs(TypedDict):
    img1: OptionalImage
    img2: OptionalImage
    img3: OptionalImage
    img4: OptionalImage
    images: OptionalImageList
    dir: FilePath
    filename: str


class SaveImageItem(TypedDict):
    """Per-instance metadata, not a graph input or an image wrapper."""

    id: str
    name: str


class SaveImagesOutputs(TypedDict):
    paths: dict[str, str]


class FolderImagesInputs(TypedDict):
    dir: FilePath
    recursive: int
    index: int


class FolderImagesOutputs(TypedDict):
    image: RGBImage
    path: PathString
    filename: str
    index: int
    total: int


class AutoColorRegionsInputs(TypedDict):
    image: RGBImage
    mask: MaskImage
    detail: int


class SelectColorRegionInputs(TypedDict):
    regions: ColorRegionSet
    region_id: int


class ColorRegionSwatchesInputs(TypedDict):
    regions: ColorRegionSet
    indices: RegionIndices


class SwatchColors(TypedDict):
    indices: list[int]
    region_ids: list[int]
    rgb: list[tuple[int, int, int]]
    fractions: list[float]
    size: tuple[int, int]


class ColorSwatchOutputs(TypedDict):
    """Default one-swatch template; instance schemas extend img1 to imgN."""

    img1: RGBImage
    colors: SwatchColors


class ColorSimilarityInputs(TypedDict):
    image1: RGBImage
    image2: RGBImage


class ColorSimilarityOutputs(TypedDict):
    score: float


def port_schema(contract: type) -> dict[str, str]:
    """从字段注解生成图端口类型，无法识别的类型立即报错，不静默回退。"""
    if not is_typeddict(contract):
        raise TypeError(f"节点合同必须是 TypedDict 类：{contract!r}")
    native = {
        str: PortType.STRING, int: PortType.INT, float: PortType.FLOAT,
        Path: PortType.PATH, PipelineConfig: PortType.CONFIG,
        ObjectRecord: PortType.OBJECT_RECORD, PreparedSample: PortType.PREPARED_SAMPLE,
        ColorRegionSet: PortType.COLOR_REGIONS,
    }
    result = {}
    for name, annotation in get_type_hints(contract, include_extras=True).items():
        if get_origin(annotation) is Annotated:
            markers = [item for item in get_args(annotation)[1:] if isinstance(item, PortType)]
            if len(markers) != 1:
                raise TypeError(f"{contract.__name__}.{name} 必须声明唯一 PortType。")
            port_type = markers[0]
        elif get_origin(annotation) is dict or is_typeddict(annotation):
            port_type = PortType.JSON_RECORD
        else:
            port_type = native.get(annotation)
            if port_type is None:
                raise TypeError(f"{contract.__name__}.{name} 的类型没有端口类别：{annotation!r}")
        result[name] = port_type.value
    return result
