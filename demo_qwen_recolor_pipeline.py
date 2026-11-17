"""Qwen-Image-Edit-2511：原图、涂灰图、色块三参考图换色。

参考 https://huggingface.co/Qwen/Qwen-Image-Edit-2511 的 Quick Start。
目标颜色只由第三张色块图提供，不在 prompt 中写颜色名、RGB 或色号。
"""
import torch
from PIL import Image, ImageOps
from diffusers import QwenImageEditPlusPipeline

model_path = '../../hf_cache/Qwen-Image-Edit-2511'

# True：逐层 CPU offload，显存占用较低，但较慢。
# False：按组件 offload，速度较快，需要足够显存容纳整个 Transformer。
use_sequential_cpu_offload = False

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
if use_sequential_cpu_offload:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()
pipeline.set_progress_bar_config(disable=False)

# 图1：待改色的原图；图2：对应的涂灰定位图；图3：目标颜色的色块。
with Image.open('./test/原图A/00019_00.png') as image:
    original = ImageOps.exif_transpose(image).convert('RGB')
with Image.open('./test/经过涂灰的原图AB/00019_00.png') as image:
    masked = ImageOps.exif_transpose(image).convert('RGB')
with Image.open('./test/主颜色B/00019_00.png') as image:
    swatch = ImageOps.exif_transpose(image).convert('RGB')
# 最近邻放大色块，保持其 RGB 像素值；不把 RGB 数值填进 prompt。
swatch = swatch.resize((128, 128), Image.Resampling.NEAREST)
images = [original, masked, swatch]

# Picture编号与Qwen pipeline自动添加的图片标签一致；目标颜色只从图3取。
prompt = (
    'Edit Picture 1 in place. Sample the color directly from the solid color swatch in Picture 3 '
    'and apply it as the base fabric color of the shirt in Picture 1. '
    'Picture 2 is only a location guide: the covered area, identified by its difference from Picture 1, '
    'marks the shirt region to edit. Use Picture 3 as the only source of the target color. '
    'Preserve the original fabric texture, folds, seams, printed logo and lighting from Picture 1. '
    'Keep the person, pose, background, camera framing and image dimensions unchanged. '
    'Do not zoom, crop or reframe. Return one edited version of Picture 1, '
    'without reference panels, the swatch or location markings.'
)

# 与官方 Quick Start 保持一致的主要采样参数。
inputs = {
    'image': images,
    'prompt': prompt,
    'generator': torch.Generator(device='cpu').manual_seed(43),
    'true_cfg_scale': 4.0,
    'negative_prompt': ' ',
    'num_inference_steps': 40,
    'guidance_scale': 1.0,
    'num_images_per_prompt': 1,
    'height': original.height,
    'width': original.width,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save('qwen_recolor_result.png')
print('Saved: qwen_recolor_result.png')
