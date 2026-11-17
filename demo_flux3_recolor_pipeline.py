"""对齐工作流随机换色节点的 FLUX.3 / FLUX.2 直接推理对照。"""
import gc
import json
import secrets
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageOps
from diffusers import ModularPipeline, Flux2KleinPipeline
from diffusers.utils import load_image
from object_inpainting.nodes.flux_random_recolor import choose_recolor, recover_recolor_mask

# 与 object-inpainting-workflow (11)new.json 的随机换色节点一致。
seed = -1  # -1 每次随机；复跑时填下面 run_config.json 记录的 actual_seed。
num_inference_steps = 8
guidance_scale = 4.0
actual_seed = secrets.randbelow(2**32) if seed == -1 else seed

target = load_image('./test/原图A/00019_00.png')
masked_target = load_image('./test/经过涂灰的原图AB/00019_00.png')
binary_mask = recover_recolor_mask(target, masked_target)
color_name, rgb = choose_recolor(target, binary_mask, actual_seed)
sampled_color = Image.new('RGB', (128, 128), rgb)
width, height = target.size
padding = (0, 0, (-width) % 16, (-height) % 16)
images = [
    ImageOps.expand(target, border=padding, fill=(128, 128, 128)),
    ImageOps.expand(masked_target, border=padding, fill=(128, 128, 128)),
    sampled_color,
]
# 这次复现工作流，按用户要求保留color_name；不是“仅色块”能力实验。
prompt = (
    "Edit image 1, the original photograph. Image 2 is the same photograph with only the editable "
    "clothing region covered in uniform gray. The region replaced by gray compared with image 1 "
    "marks where to recolor; all other regions must stay unchanged, including naturally gray objects. "
    "Image 3 is the desired color swatch. "
    f"Recolor only the marked clothing region to new color, matching image 3 color. "
    "Use image 1, not the gray fill in image 2, to preserve all original clothing details. "
    "Preserve the exact person's identity, pose, garment silhouette, seams, buttons, prints, logos, "
    "fabric weave, texture, folds, shadows, highlights, camera framing, and background. "
    "Change fabric color only, not material or geometry. Do not add objects, text, mask markings, "
    "or the swatch to the output. Return only the edited photograph."
)
output_dir = Path('outputs/flux_workflow_aligned_demo') / str(actual_seed)
output_dir.mkdir(parents=True, exist_ok=False)
for name in ('flux3_recolor_result.png', 'flux2_recolor_result.png'):
    if Path(name).is_file():
        shutil.copy2(name, output_dir / ('previous_' + name))
for i, image in enumerate(images, 1):
    image.save(output_dir / f'condition_{i}.png')
binary_mask.save(output_dir / 'mask.png')
metadata = dict(seed=seed, actual_seed=actual_seed, color_name=color_name, sampled_color_rgb=rgb,
    prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
    condition_order=['original', 'gray', 'swatch'], size=[width, height], completed=[])
(output_dir / 'run_config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'actual_seed={actual_seed}, color={color_name}, rgb={rgb}, output={output_dir}', flush=True)

model_path = Path('../../hf_cache/models--black-forest-labs--flux-3-image-klein-base')
if (model_path / 'refs/main').is_file():
    model_path = model_path / 'snapshots' / (model_path / 'refs/main').read_text().strip()

pipe = ModularPipeline.from_pretrained(str(model_path), local_files_only=True)
# 本地未下载可选的 prompt 扩写模型；保持我们填写的 prompt 原文。
pipe._blocks.sub_blocks.pop('prompt_upsampler', None)
pipe.load_components(
    names=['text_encoder', 'tokenizer', 'transformer', 'vae'],
    dtype={'vae': torch.float32, 'default': torch.bfloat16},
    local_files_only=True,
    pretrained_model_name_or_path={
        'tokenizer': str(model_path / 'text_encoder'),
        'default': str(model_path),
    },
    subfolder={'tokenizer': ''},
)
pipe.to('cuda')


image = pipe(
    prompt=prompt,
    image=images,
    width=images[0].width,
    height=images[0].height,
    generator=torch.Generator(device='cpu').manual_seed(actual_seed),
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    output='images',
)[0]
image = image.crop((0, 0, width, height)).convert('RGB')
image.save('flux3_recolor_result.png')
image.save(output_dir / 'flux3_raw.png')
Image.composite(image, target, binary_mask).save(output_dir / 'flux3_composited.png')
metadata['completed'].append('flux3')
(output_dir / 'run_config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

# 释放 FLUX.3 再加载 FLUX.2。CPU offload避免9B与Qwen同时常驻GPU。
del pipe
gc.collect()
torch.cuda.empty_cache()
pipe = Flux2KleinPipeline.from_pretrained(
    '../../hf_cache/FLUX.2-klein-base-9B', torch_dtype=torch.bfloat16, local_files_only=True,
)
pipe.enable_model_cpu_offload()
image = pipe(
    prompt=prompt,
    image=images,
    width=images[0].width,
    height=images[0].height,
    generator=torch.Generator(device='cpu').manual_seed(actual_seed),
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    output_type='pil',
).images[0]
image = image.crop((0, 0, width, height)).convert('RGB')
image.save('flux2_recolor_result.png')
image.save(output_dir / 'flux2_raw.png')
Image.composite(image, target, binary_mask).save(output_dir / 'flux2_composited.png')
metadata['completed'].append('flux2')
(output_dir / 'run_config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Both results saved to {output_dir}', flush=True)
