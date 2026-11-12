"""Three-reference inference and visual validation with the training preprocessing."""

import argparse
import copy
import math
from pathlib import Path

from PIL import Image

from . import CONDITION_FOLDERS, TARGET_FOLDER
from .config import read_json, write_json, prepare_output, validate_local_model
from .data import load_rgb, prepare_with_policy


def load_run_config(adapter):
    config = read_json(Path(adapter) / 'recolor_config.json')
    if (config.get('format_version') != 1 or config.get('condition_folders') != list(CONDITION_FOLDERS)
            or config.get('target_folder') != TARGET_FOLDER or config.get('grid') != 16):
        raise ValueError('适配器配置版本、条件顺序或网格不兼容')
    if not 256 <= config.get('max_area', 0) <= 1024 * 1024:
        raise ValueError('适配器缺少有效的预处理面积预算')
    return config


def build_pipeline(components, device):
    from diffusers.modular_pipelines.flux3 import Flux3AutoBlocks
    blocks = Flux3AutoBlocks()
    # Fixed task instruction: do not invoke the optional prompt-upsample model.
    # Text is encoded explicitly once with the exact training text settings.
    blocks.sub_blocks.pop('prompt_upsampler')
    blocks.sub_blocks.pop('text_encoder')
    pipeline = blocks.init_pipeline()
    pipeline.update_components(transformer=components['transformer'], vae=components['vae'],
                               scheduler=copy.deepcopy(components['scheduler']))
    pipeline.to(device)
    return pipeline


def generate_sample(pipeline, conditions, config, seed, steps, embeddings):
    import torch
    _, references = prepare_with_policy(None, conditions, config.get('preprocessing') or {'max_area': config['max_area']})
    device = pipeline._execution_device
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        # Explicit canvas prevents implicit sizing/crop-back from the reference branch.
        image = pipeline(
            image=references, prompt_embeds=embeddings[0], negative_prompt_embeds=embeddings[1],
            height=references[0].height, width=references[0].width,
            generator=generator, num_inference_steps=steps, output_type='pil', output='images',
        )[0]
    return image, references


def save_comparison(path, target, references, generated):
    # Order: gray scene, B scene, A swatch, target A (if present), generated A.
    images = [*references]
    if target is not None:
        images.append(target)
    images.append(generated)
    width, height = generated.size
    canvas = Image.new('RGB', (width * len(images), height), 'white')
    for index, image in enumerate(images):
        tile = image.copy()
        tile.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.paste(tile, (width * index + (width - tile.width) // 2, (height - tile.height) // 2))
    canvas.save(path)


def generate_validation(pipeline, root, names, config, output, seed, steps, embeddings):
    if not names:
        raise ValueError('验证清单为空')
    output = prepare_output(output, protected=[root])
    manifest = []
    for index, name in enumerate(names):
        if Path(name).name != name or name in {'', '.', '..'}:
            raise ValueError(f'无效验证文件名: {name}')
        target, conditions = load_validation_sample(root, name, config)
        generated, references = generate_sample(pipeline, conditions, config, seed + index, steps, embeddings)
        sample_dir = output / name
        sample_dir.mkdir()
        target.save(sample_dir / 'target.png')
        for i, image in enumerate(references, 1):
            image.save(sample_dir / f'condition_{i}.png')
        generated.save(sample_dir / 'generated.png')
        save_comparison(sample_dir / 'comparison.png', target, references, generated)
        manifest.append({'name': name, 'seed': seed + index, 'steps': steps})
    write_json(output / 'manifest.json', {
        'samples': manifest, 'condition_folders': list(CONDITION_FOLDERS),
        'comparison_order': [*CONDITION_FOLDERS, TARGET_FOLDER, 'generated'],
        'prompt': config.get('prompt'), 'max_area': config['max_area'],
    })


def load_validation_sample(root, name, config):
    return prepare_with_policy(load_rgb(Path(root) / TARGET_FOLDER / name),
        [load_rgb(Path(root) / folder / name) for folder in CONDITION_FOLDERS],
        config.get('preprocessing') or {'max_area': config['max_area']})


def main(argv=None):
    parser = argparse.ArgumentParser(description='使用 FLUX.3 换色 LoRA 进行三参考图推理；只加载本地权重')
    parser.add_argument('--pretrained_model_name_or_path', help='不指定则使用训练配置中的本地基础模型路径')
    parser.add_argument('--lora_dir', required=True, help='训练输出目录或 checkpoint 目录')
    parser.add_argument('--output_dir', required=True, help='新的空目录')
    parser.add_argument('--gray_image')
    parser.add_argument('--generated_b_image')
    parser.add_argument('--color_a_image')
    parser.add_argument('--dataset_root', help='与 --split_json 一起使用，生成留出验证集结果')
    parser.add_argument('--split_json', help='训练生成的 split.json；默认 lora_dir/split.json')
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--prompt', help='可覆盖训练默认指令')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], default='bf16')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    paths = [args.gray_image, args.generated_b_image, args.color_a_image]
    if bool(args.dataset_root) == any(paths) or (any(paths) and not all(paths)):
        parser.error('请提供完整三张图片，或只提供 --dataset_root 进行验证集推理')
    if min(args.limit, args.num_inference_steps) < 1 or args.seed < 0:
        parser.error('limit/num_inference_steps 必须为正，seed 不能为负')
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error('guidance_scale 必须为有限数且 >=1')
    config = load_run_config(args.lora_dir)
    model_path, signature = validate_local_model(args.pretrained_model_name_or_path or config['model_path'])
    if signature != config['model_config_signature']:
        raise ValueError('基础模型配置与训练时不一致')
    if args.prompt is not None:
        config['prompt'] = args.prompt
    protected = [args.lora_dir, model_path]
    if args.dataset_root:
        protected.append(args.dataset_root)
        split = read_json(args.split_json or Path(args.lora_dir) / 'split.json')
        names = split['validation'][:args.limit]
        if not names:
            raise ValueError('验证集为空')
        for name in names:
            if Path(name).name != name:
                raise ValueError(f'无效验证文件名: {name}')
            load_validation_sample(args.dataset_root, name, config)
    else:
        protected.extend(paths)
        _, conditions = prepare_with_policy(None, [load_rgb(p) for p in paths],
            config.get('preprocessing') or {'max_area': config['max_area']})
    output = prepare_output(args.output_dir, protected=protected)
    import torch
    from .modeling import load_components, encode_prompt
    device = torch.device(args.device)
    if device.type == 'cpu' and args.mixed_precision != 'no':
        raise ValueError('CPU 推理请使用 --mixed_precision no')
    dtype = {'no': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.mixed_precision]
    components = load_components(model_path, dtype, device)
    text_embeddings = []
    for prompt in (config['prompt'], ''):
        embedding, _ = encode_prompt(components['text_encoder'], components['tokenizer'], prompt, device,
                                     dtype, config['text_encoder_out_layers'], config['max_sequence_length'])
        text_embeddings.append(embedding)
    components.pop('text_encoder').to('cpu')
    components.pop('tokenizer')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    pipeline = build_pipeline(components, device)
    pipeline.load_lora_weights(str(Path(args.lora_dir).resolve()), local_files_only=True)
    pipeline.update_components(guider=pipeline.guider.new(guidance_scale=args.guidance_scale))
    if args.dataset_root:
        generate_validation(pipeline, args.dataset_root, names, config, output, args.seed,
                            args.num_inference_steps, text_embeddings)
    else:
        image, references = generate_sample(pipeline, conditions, config, args.seed,
                                             args.num_inference_steps, text_embeddings)
        image.save(output / 'generated.png')
        for i, reference in enumerate(references, 1):
            reference.save(output / f'condition_{i}.png')
        save_comparison(output / 'comparison.png', None, references, image)
    write_json(output / 'inference_config.json', {**config, 'seed': args.seed,
               'num_inference_steps': args.num_inference_steps, 'guidance_scale': args.guidance_scale})
    print(f'生成结果已保存到 {output}')
