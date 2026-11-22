#!/usr/bin/env python
"""Standalone original FLUX.3 pipeline test. No training modules or LoRA.

Condition order: masked photograph, source photograph, target color swatch.
Target color comes only from the swatch, never injected into the prompt.
"""
import argparse
import json
import math
from pathlib import Path

# 支持从项目根目录直接执行此子目录脚本。
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageOps

DEFAULT_PROMPT = (
    'Edit image 2. Image 1 marks the editable clothing region. '
    'Recolor only the clothing in image 2 to match the color swatch in image 3. '
    'Use image 2 to preserve the original clothing details. '
    'Preserve the logo design, garment texture, seams, folds, lighting, person, pose and background. '
    'Change clothing color only. Return only the edited photograph.'
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrained_model_name_or_path', default='../../hf_cache/models--black-forest-labs--flux-3-image-klein-base')
    p.add_argument('--dataset_root', default='./test')
    p.add_argument('--sample', default='00019_00.png')
    p.add_argument('--direction', choices=['b_to_a', 'a_to_b'], default='b_to_a')
    p.add_argument('--gray_image')
    p.add_argument('--source_image')
    p.add_argument('--swatch_image')
    p.add_argument('--prompt', default=DEFAULT_PROMPT)
    p.add_argument('--prompt_file', help='UTF-8 prompt file; overrides --prompt')
    p.add_argument('--output_dir', default='./outputs/flux3_pipeline_test')
    p.add_argument('--device', default='cuda')
    p.add_argument('--mixed_precision', choices=['no', 'bf16', 'fp16'], default='bf16')
    p.add_argument('--resolution', type=int, default=1024, help='Scene area budget = resolution squared')
    p.add_argument('--color_swatch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--num_inference_steps', type=int, default=28)
    p.add_argument('--guidance_scale', type=float, default=3.5)
    p.add_argument('--max_sequence_length', type=int, default=512)
    p.add_argument('--text_encoder_out_layers', type=int, nargs='+', default=[4, 8, 12, 16, 20, 24, 28, 32])
    p.add_argument('--offload_text_encoder', action='store_true',
                   help='Precompute text with the official encoder step, then unload Qwen before loading Transformer')
    args = p.parse_args(argv)
    if args.resolution < 16 or not 16 <= args.color_swatch_size <= 1024 or args.color_swatch_size % 16:
        p.error('resolution must be >=16; color_swatch_size must be a multiple of 16 in [16,1024]')
    if args.num_inference_steps < 1:
        p.error('num_inference_steps must be positive')
    explicit = [args.gray_image, args.source_image, args.swatch_image]
    if any(explicit) and not all(explicit):
        p.error('Provide gray_image, source_image and swatch_image together')
    if Path(args.sample).name != args.sample:
        p.error('sample must be a filename, not a path')
    return args


def resolve_model_root(path):
    root = Path(path).expanduser().resolve()
    if (root / 'snapshots').is_dir() and not (root / 'transformer').is_dir():
        revision = (root / 'refs/main').read_text().strip()
        if not revision or any(c not in '0123456789abcdefABCDEF' for c in revision):
            raise ValueError('Invalid HF cache refs/main; supply a snapshot directory')
        root = root / 'snapshots' / revision
    if not (root / 'modular_model_index.json').is_file():
        raise FileNotFoundError(f'Missing modular_model_index.json: {root}')
    return root


def prepare_inputs(args):
    if args.gray_image:
        paths = [args.gray_image, args.source_image, args.swatch_image]
    else:
        folders = (['经过涂灰的原图AB', '生成图B', '主颜色A'] if args.direction == 'b_to_a'
                   else ['经过涂灰的原图AB', '原图A', '主颜色B'])
        paths = [str(Path(args.dataset_root) / folder / args.sample) for folder in folders]
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(ImageOps.exif_transpose(im).convert('RGB'))
    if images[0].size != images[1].size:
        raise ValueError('Gray and source photographs must have the same dimensions')
    w, h = images[1].size
    scale = min(1., args.resolution / math.sqrt(w * h))
    size = (max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16))
    images[:2] = [im.resize(size, Image.Resampling.LANCZOS) for im in images[:2]]
    images[2] = images[2].resize((args.color_swatch_size,) * 2, Image.Resampling.NEAREST)
    return paths, images


def main(argv=None):
    args = parse_args(argv)
    root = resolve_model_root(args.pretrained_model_name_or_path)
    paths, images = prepare_inputs(args)
    prompt = Path(args.prompt_file).read_text(encoding='utf-8').strip() if args.prompt_file else args.prompt
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a new or empty output directory: {output}')
    import torch
    from diffusers import ModularPipeline
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; CPU testing requires --device cpu --mixed_precision no')
    if device.type == 'cpu' and args.mixed_precision != 'no':
        raise ValueError('CPU testing requires --mixed_precision no')
    dtype = {'no': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Official factory and component loader. Do not load the optional prompt upsampler.
    pipe = ModularPipeline.from_pretrained(str(root), local_files_only=True)
    pipe._blocks.sub_blocks.pop('prompt_upsampler', None)
    index = json.loads((root / 'modular_model_index.json').read_text())
    tokenizer_folder = index['tokenizer'][2].get('subfolder', 'text_encoder')
    if Path(tokenizer_folder).is_absolute() or '..' in Path(tokenizer_folder).parts:
        raise ValueError('Unsafe tokenizer subfolder in model index')
    load_kwargs = dict(
        dtype={'vae': torch.float32, 'default': dtype}, local_files_only=True,
        pretrained_model_name_or_path={'tokenizer': str(root / tokenizer_folder), 'default': str(root)},
        subfolder={'tokenizer': ''},
    )
    text_inputs = {'prompt': prompt, 'negative_prompt': '',
                   'max_sequence_length': args.max_sequence_length,
                   'text_encoder_out_layers': tuple(args.text_encoder_out_layers)}
    with torch.inference_mode():
        if args.offload_text_encoder:
            from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
            pipe.load_components(names=['text_encoder', 'tokenizer'], **load_kwargs)
            pipe.to(device)
            embeddings = [Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                pipe.text_encoder, pipe.tokenizer, text, device=device,
                max_sequence_length=args.max_sequence_length,
                hidden_states_layers=tuple(args.text_encoder_out_layers)).to(dtype) for text in (prompt, '')]
            pipe.unload_components(['text_encoder', 'tokenizer'])
            pipe._blocks.sub_blocks.pop('text_encoder')
            text_inputs = {'prompt_embeds': embeddings[0], 'negative_prompt_embeds': embeddings[1]}
            pipe.load_components(names=['transformer', 'vae'], **load_kwargs)
        else:
            pipe.load_components(names=['text_encoder', 'tokenizer', 'transformer', 'vae'], **load_kwargs)
        pipe.to(device)
        pipe.update_components(guider=pipe.guider.new(guidance_scale=args.guidance_scale))
        pipe.transformer.requires_grad_(False).eval()
        pipe.vae.requires_grad_(False).eval()
        if any('lora_' in name for name, _ in pipe.transformer.named_parameters()):
            raise RuntimeError('Expected original weights without LoRA')
        generated = pipe(image=images, height=images[1].height, width=images[1].width,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            num_inference_steps=args.num_inference_steps, output='images', **text_inputs)[0]
    output.mkdir(parents=True, exist_ok=True)
    generated.save(output / 'generated.png')
    for i, im in enumerate(images, 1):
        im.save(output / f'condition_{i}.png')
    metadata = {**vars(args), 'prompt': prompt, 'model_path': str(root), 'input_paths': paths,
                'condition_order': ['gray', 'source', 'swatch'], 'lora_loaded': False,
                'scene_size': list(images[1].size), 'swatch_size': list(images[2].size)}
    (output / 'run_config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {output / "generated.png"}')
    return generated


if __name__ == '__main__':
    main()
