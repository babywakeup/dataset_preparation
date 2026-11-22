"""使用已训练的 FLUX.3 LoRA / text encoder，进行三参考图换色推理。"""
import argparse
from pathlib import Path

# 从项目根目录直接执行子目录 demo。
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flux3_recolor.component_io import load_component_config
from flux3_recolor.config import prepare_output, write_json
from flux3_recolor.data import load_rgb
from flux3_recolor.inference import load_recolor_pipeline, generate_sample, save_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True, help='训练导出目录或 checkpoint 目录')
    parser.add_argument('--pretrained_model_name_or_path', help='可覆盖配置中记录的基础模型路径')
    parser.add_argument('--gray_image', required=True)
    parser.add_argument('--source_image', required=True)
    parser.add_argument('--swatch_image', required=True)
    parser.add_argument('--output_dir', required=True, help='新的空目录')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mixed_precision', choices=['no', 'bf16', 'fp16'], default='bf16')
    args = parser.parse_args(argv)
    if args.seed < 0 or args.num_inference_steps < 1:
        parser.error('seed 必须非负，num_inference_steps 必须为正')

    directory = Path(args.model_dir).expanduser().resolve()
    manifest = load_component_config(directory)
    base = args.pretrained_model_name_or_path or manifest['base_model']
    paths = [args.gray_image, args.source_image, args.swatch_image]
    conditions = [load_rgb(Path(path).expanduser()) for path in paths]
    output = prepare_output(args.output_dir, protected=[directory, base, *paths])
    print(f'[基础权重] VAE / Transformer / Qwen text_encoder: {base}', flush=True)
    pipeline = load_recolor_pipeline(
        directory, pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        device=args.device, mixed_precision=args.mixed_precision, guidance_scale=args.guidance_scale)
    print(f'[已训练权重] 已加载 Transformer LoRA: {directory / manifest["transformer"]["path"]}', flush=True)
    if manifest['text_encoder']['type'] == 'learned':
        print(f'[已训练权重] 已加载 text_encoder: {directory / manifest["text_encoder"]["path"]}; '
              f'shape={tuple(pipeline.recolor_text_encoder.weight.shape)}', flush=True)
    else:
        print('[文本条件] 使用训练配置中保存的 prompt', flush=True)

    # 必须遵循训练配置顺序：涂灰图、源图、色块；噪声由 pipeline 内部创建。
    generated, references = generate_sample(
        pipeline, conditions, pipeline.recolor_config, args.seed, args.num_inference_steps)
    generated.save(output / 'generated.png')
    save_comparison(output / 'comparison.png', None, references, generated)
    for index, image in enumerate(references, 1):
        image.save(output / f'condition_{index}.png')
    write_json(output / 'run_config.json', {
        **vars(args), 'model_dir': str(directory), 'base_model': str(base),
        'condition_order': ['gray', 'source', 'swatch'],
        'text_encoder_type': manifest['text_encoder']['type'],
    })
    print(f'结果: {output / "generated.png"}', flush=True)
    return generated


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
