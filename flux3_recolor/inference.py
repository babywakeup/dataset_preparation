"""Three-reference inference and visual validation with the training preprocessing."""

import argparse
import copy
import math
from pathlib import Path

from PIL import Image

from . import task_folders
from .config import read_json, write_json, prepare_output, validate_local_model
from .data import load_rgb, prepare_with_policy


def load_run_config(adapter):
    config = read_json(Path(adapter) / 'recolor_config.json')
    if (config.get('format_version') != 1 or config.get('dataset_format') != 'recolor_paths_v1'
            or config.get('condition_order') != ['gray', 'source', 'swatch'] or config.get('grid') != 16):
        raise ValueError('适配器配置版本、条件顺序或网格不兼容')
    if not 256 <= config.get('max_area', 0) <= 1024 * 1024:
        raise ValueError('适配器缺少有效的预处理面积预算')
    from .component_io import load_component_config
    components = load_component_config(adapter)
    if (components['text_encoder']['type'] == 'learned') != bool(config.get('train_text_encoder')):
        raise ValueError('文本组件类型与训练配置不一致')
    if config.get('train_text_encoder', False):
        from .text_encoder import load_text_embeddings
        load_text_embeddings(adapter)
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
    from .text_encoder import TextEncoderModel
    if isinstance(components.get('text_encoder'), TextEncoderModel):
        pipeline.recolor_text_encoder = components['text_encoder']
    return pipeline


def generate_sample(pipeline, conditions, config, seed, steps, embeddings=None):
    import torch
    if embeddings is None:
        from .validation import pipeline_text_embeddings
        embeddings = pipeline_text_embeddings(pipeline, pipeline.recolor_embeddings)
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
    # Order: gray scene, source scene, target swatch, target (if present), generated.
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


def select_validation_records(records, limit):
    """Take evenly spaced rows, including both endpoints, in saved dataset order."""
    count = min(len(records), limit)
    if count < 1:
        return records.select([]) if hasattr(records, 'select') else []
    indices = [i * (len(records) - 1) // (count - 1) for i in range(count)] if count > 1 else [0]
    return records.select(indices) if hasattr(records, 'select') else [records[i] for i in indices]


def generate_validation(pipeline, root, records, config, output, seed, steps, embeddings):
    """Render saved validation rows without reconstructing directions."""
    from .data import read_path_sample
    if not len(records):
        raise ValueError('验证清单为空')
    output = prepare_output(output, protected=[root])
    manifest = []
    policy = config.get('preprocessing') or {'max_area': config['max_area']}
    for index, row in enumerate(records):
        target, conditions = read_path_sample(row, policy)
        generated, references = generate_sample(pipeline, conditions, config, seed + index, steps, embeddings)
        sample_dir = output / f'{index:04d}'
        sample_dir.mkdir(parents=True)
        target.save(sample_dir / 'target.png')
        for i, image in enumerate(references, 1):
            image.save(sample_dir / f'condition_{i}.png')
        generated.save(sample_dir / 'generated.png')
        save_comparison(sample_dir / 'comparison.png', target, references, generated)
        manifest.append({'sample_id': row['sample_id'], 'source_id': row['source_id'],
            'name': Path(row['target_path']).name, 'target_path': row['target_path'],
            'condition_paths': list(row['condition_paths']), 'path': sample_dir.relative_to(output).as_posix(),
            'seed': seed + index, 'steps': steps})
    write_json(output / 'manifest.json', {'samples': manifest,
        'comparison_order': ['gray', 'source', 'swatch', 'target', 'generated'],
        'prompt': config.get('prompt'), 'max_area': config['max_area']})


def load_recolor_pipeline(directory, pretrained_model_name_or_path=None, device='cuda',
                          mixed_precision='bf16', prompt=None, guidance_scale=3.5):
    """Load the native pipeline from a local component export."""
    import torch
    from .component_io import load_component_config
    from .modeling import load_components, encode_prompt
    from .text_encoder import TextEncoderModel
    directory = Path(directory)
    config = load_run_config(directory)
    manifest = load_component_config(directory)
    learned = manifest['text_encoder']['type'] == 'learned'
    if learned != bool(config.get('train_text_encoder', False)):
        raise ValueError('text_encoder 组件类型与训练配置不一致')
    model_path, signature = validate_local_model(pretrained_model_name_or_path or manifest['base_model'])
    if signature != config['model_config_signature']:
        raise ValueError('基础模型配置与训练时不一致')
    if prompt is not None:
        config['prompt'] = prompt
    device = torch.device(device)
    if device.type == 'cpu' and mixed_precision != 'no':
        raise ValueError('CPU 推理请使用 --mixed_precision no')
    if not math.isfinite(guidance_scale) or guidance_scale < 1:
        raise ValueError('guidance_scale 必须为有限数且 >=1')
    dtype = {'no': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[mixed_precision]
    components = load_components(model_path, dtype, device)
    def encode(text):
        return encode_prompt(components['text_encoder'], components['tokenizer'], text, device,
                             dtype, config['text_encoder_out_layers'], config['max_sequence_length'])[0]
    positive = None if learned else encode(config['prompt'])
    negative = encode('')
    components.pop('text_encoder').to('cpu')
    components.pop('tokenizer')
    if learned:
        encoder = TextEncoderModel.from_pretrained(directory / manifest['text_encoder']['path'], local_files_only=True)
        encoder.requires_grad_(False).eval().to(device)
        components['text_encoder'] = encoder
        with torch.no_grad():
            positive = encoder().last_hidden_state.detach().to(device=device, dtype=dtype)
        if (positive.shape[-1] != components['transformer'].config.joint_attention_dim
                or positive.shape[1] > config['max_sequence_length']):
            raise ValueError('text encoder shape 与当前模型/文本长度不匹配')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    pipeline = build_pipeline(components, device)
    pipeline.load_lora_weights(str(directory / manifest['transformer']['path']), local_files_only=True)
    pipeline.update_components(guider=pipeline.guider.new(guidance_scale=guidance_scale))
    pipeline.recolor_config = config
    pipeline.recolor_embeddings = (positive, negative)
    return pipeline


def load_validation_sample(root, name, config, direction='b_to_a'):
    target_folder, condition_folders = task_folders(direction)
    return prepare_with_policy(load_rgb(Path(root) / target_folder / name),
        [load_rgb(Path(root) / folder / name) for folder in condition_folders],
        config.get('preprocessing') or {'max_area': config['max_area']})


def main(argv=None):
    parser = argparse.ArgumentParser(description='使用 FLUX.3 换色 LoRA 进行三参考图推理；只加载本地权重')
    parser.add_argument('--pretrained_model_name_or_path', help='不指定则使用model_config.json记录的基础模型路径')
    parser.add_argument('--lora_dir', required=True, help='训练输出目录或 checkpoint 目录')
    parser.add_argument('--output_dir', required=True, help='新的空目录')
    parser.add_argument('--gray_image')
    parser.add_argument('--generated_b_image', '--source_image', dest='generated_b_image')
    parser.add_argument('--color_a_image', '--color_image', dest='color_a_image')
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
    from .component_io import load_component_config
    sources = load_component_config(args.lora_dir)
    model_path, signature = validate_local_model(args.pretrained_model_name_or_path or sources['base_model'])
    if signature != config['model_config_signature']:
        raise ValueError('基础模型配置与训练时不一致')
    if args.prompt is not None:
        config['prompt'] = args.prompt
    protected = [args.lora_dir, model_path]
    if args.dataset_root:
        protected.append(args.dataset_root)
        from datasets import load_from_disk
        if not config.get('dataset_path'):
            raise ValueError('适配器未记录路径Dataset，请使用三图片路径推理')
        data = load_from_disk(config['dataset_path'])['validation']
        if args.split_json:
            selected = set(read_json(args.split_json)['validation'])
            data = data.select([i for i, sample_id in enumerate(data['sample_id']) if sample_id in selected])
        records = [dict(row) for row in select_validation_records(data, args.limit)]
        if not records:
            raise ValueError('验证集为空')
        old_root = Path(config['dataset_root']).resolve()
        new_root = Path(args.dataset_root).resolve()
        if old_root != new_root:
            for row in records:
                row['target_path'] = str(new_root / Path(row['target_path']).relative_to(old_root))
                row['condition_paths'] = [str(new_root / Path(path).relative_to(old_root))
                                          for path in row['condition_paths']]
    else:
        protected.extend(paths)
        _, conditions = prepare_with_policy(None, [load_rgb(p) for p in paths],
            config.get('preprocessing') or {'max_area': config['max_area']})
    output = prepare_output(args.output_dir, protected=protected)
    pipeline = load_recolor_pipeline(args.lora_dir, args.pretrained_model_name_or_path,
        args.device, args.mixed_precision, args.prompt, args.guidance_scale)
    text_embeddings = pipeline.recolor_embeddings
    if args.dataset_root:
        generate_validation(pipeline, args.dataset_root, records, config, output, args.seed,
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
