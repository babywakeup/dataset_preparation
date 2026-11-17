"""No-grad replay of saved identical inputs, with the production precision policy.

Uses only the base transformer (no Qwen/VAE), to keep GPU memory bounded.
Synthetic nonzero LoRA weights test arithmetic consistency, not model quality.
"""
import argparse
from pathlib import Path
import torch
from diffusers import Flux3Transformer2DModel
from flux3_recolor.config import resolve_model_root, write_json
from flux3_recolor.modeling import add_lora
from flux3_recolor.precision import set_transformer_autocast, pipeline_forward
from compare_flux3_training_pipeline import tensor_difference


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained_model_name_or_path', required=True)
    parser.add_argument('--snapshots', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    paths = sorted(Path(args.snapshots).rglob('tensors-seed-*.pt'))
    if not paths:
        raise ValueError('没有输入快照')
    # Limit this diagnostic to ~16GiB on an 80GiB GPU; never consume all free RAM.
    torch.cuda.set_per_process_memory_fraction(.20)
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError('可用显存不足16GiB，稍后再运行')
    model = Flux3Transformer2DModel.from_pretrained(str(resolve_model_root(args.pretrained_model_name_or_path)),
        subfolder='transformer', torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
    model.requires_grad_(False)
    set_transformer_autocast(model, torch.bfloat16)
    results = []
    for path in paths:
        inputs = torch.load(path, map_location='cpu', weights_only=True)['pipeline']['inputs']
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        # Both calls use the same Transformer boundary; VAE is not involved.
        expected = model(**inputs, return_dict=False)[0].cpu()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = model(**inputs, return_dict=False)[0].cpu()
        diff = tensor_difference(expected, actual, 1e-5, 1e-4)
        results.append({'snapshot': path.relative_to(args.snapshots).as_posix(), 'base_outer_amp_vs_pipeline': diff})
    add_lora(model, 16, 16, 0)
    zero = model(**inputs, return_dict=False)[0].cpu()
    zero_difference = tensor_difference(actual, zero, 1e-5, 1e-4)
    for name, p in model.named_parameters():
        if 'lora_B' in name:
            p.fill_(0.001)
    expected = model(**inputs, return_dict=False)[0].cpu()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(**inputs, return_dict=False)[0].cpu()
    # Exercise the real Accelerate forward/output conversion and temporary bypass.
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision='bf16')
    model = accelerator.prepare(model)
    unwrapped = accelerator.unwrap_model(model)
    wrapped_forward = unwrapped.forward
    with pipeline_forward(unwrapped):
        validation = unwrapped(**inputs, return_dict=False)[0].cpu()
    report = {'cases': results, 'zero_lora_vs_base': zero_difference,
              'nonzero_lora_outer_amp_vs_pipeline': tensor_difference(expected, actual, 1e-5, 1e-4),
              'shared_validation_vs_pipeline': tensor_difference(expected, validation, 1e-5, 1e-4),
              'training_forward_restored': unwrapped.forward is wrapped_forward,
              'synthetic_lora_changes_prediction': not torch.equal(zero, expected),
              'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'grad_enabled': torch.is_grad_enabled()}
    write_json(args.output, report)
    print(report)
    return report


if __name__ == '__main__':
    main()
