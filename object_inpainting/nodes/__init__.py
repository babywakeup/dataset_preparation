from .align_square import AlignSquareNode
from .assemble_sample import AssembleSampleNode
from .auto_color_regions import AutoColorRegionsNode
from .base import Node
from .build_prompt import BuildPromptNode
from .crop import CropNode
from .expand_mask import ExpandMaskNode
from .extract_reference import ExtractReferenceNode
from .flux_klein_reference import FluxKleinReferenceNode
from .flux_random_recolor import FluxRandomRecolorNode
from .folder_images import FolderImagesNode
from .gray_mask import GrayMaskNode
from .load_instance_mask import LoadInstanceMaskNode
from .load_target import LoadTargetNode
from .mask_to_rgba import MaskToRGBANode
from .record_input import RecordInputNode
from .rgba_to_white_background import RGBAToWhiteBackgroundNode
from .sam2_mask import SAM2MaskNode
from .sam3_text_mask import SAM3TextMaskNode
from .save_images import SaveImagesNode
from .save_preview import SavePreviewNode
from .save_sample import SaveSampleNode
from .select_color_region import SelectColorRegionNode
from .select_object import SelectObjectNode
from .validate_sample import ValidateSampleNode
from .write_jsonl import WriteJsonlNode


NODE_CLASS_MAPPINGS = {
    node.__name__: node
    for node in (
        RecordInputNode,
        LoadTargetNode,
        LoadInstanceMaskNode,
        SelectObjectNode,
        ExtractReferenceNode,
        ExpandMaskNode,
        AlignSquareNode,
        GrayMaskNode,
        BuildPromptNode,
        AssembleSampleNode,
        ValidateSampleNode,
        SaveSampleNode,
        WriteJsonlNode,
        SavePreviewNode,
        CropNode,
        SAM2MaskNode,
        SAM3TextMaskNode,
        MaskToRGBANode,
        RGBAToWhiteBackgroundNode,
        FluxKleinReferenceNode,
        FluxRandomRecolorNode,
        SaveImagesNode,
        FolderImagesNode,
        AutoColorRegionsNode,
        SelectColorRegionNode,
    )
}
