"""Replay saved no-grad comparisons: repeatability, AMP/conditioning factorial, FP32 arithmetic reference."""
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from compare_flux3_training_pipeline import autocast_context, tensor_difference
from flux3_recolor.config import prepare_output, read_json, validate_local_model, write_json


def mse(prediction, target):
    return (prediction.float().cpu() - target.float().cpu()).square().mean().item()


@torch.no_grad()
def analyze(model, records, device, dtype):
    results = []
    def forward(inputs, use_autocast=False, cast_dtype=None):
        values = {key: value.to(device=device, dtype=cast_dtype or value.dtype) for key, value in inputs.items()}
        context = autocast_context(device, dtype) if use_autocast else nullcontext()
        with context:
            return model(**values, return_dict=False)[0].detach().cpu()
    def delta(reference, other):
        return tensor_difference(reference, other, 1e-5, 1e-4)
    low_precision_predictions = []
    # Keep weights and saved inputs identical; only toggle context or condition tokens.
    for item in records:
        snapshot = item['snapshot']
        training_inputs = snapshot['training']['inputs']
        original_official_inputs = snapshot['pipeline']['inputs']
        for key in training_inputs:
            if key != 'image_cond_hidden_states' and not torch.allclose(
                    training_inputs[key].float(), original_official_inputs[key].float(), atol=1e-6, rtol=1e-6):
                raise ValueError(f'非条件输入未对齐，不能做factorial对照: {key}')
        # Hold non-conditioning fields exactly fixed. A CPU FP32 scheduler may
        # differ by one rounding unit after inverse-shift; don't confound the factorial.
        official_inputs = {**training_inputs,
                           'image_cond_hidden_states': original_official_inputs['image_cond_hidden_states']}
        # CPU smoke fixtures use FP32 throughout. Real GPU uses the original saved BF16 inputs.
        cast_dtype = torch.float32 if dtype == torch.float32 else None
        a = forward(training_inputs, True, cast_dtype)
        b = forward(training_inputs, False, cast_dtype)
        c = forward(official_inputs, True, cast_dtype)
        d = forward(official_inputs, False, cast_dtype)
        repeat_a = forward(training_inputs, True, cast_dtype)
        repeat_d = forward(official_inputs, False, cast_dtype)
        labels = snapshot['velocity_target']
        losses = {'training_condition_amp': mse(a, labels), 'training_condition_no_amp': mse(b, labels),
                  'official_condition_amp': mse(c, labels), 'official_condition_no_amp': mse(d, labels)}
        result = {'direction': item['direction'], 'seed': item['seed'], 'sigma': item['sigma'],
                  'repeated_training_forward': delta(a, repeat_a), 'repeated_official_forward': delta(d, repeat_d),
                  'saved_training_replay': delta(snapshot['training']['prediction'], a),
                  'saved_official_replay': delta(snapshot['pipeline']['prediction'], d),
                  'original_noncondition_inputs_exact': all(torch.equal(training_inputs[key], original_official_inputs[key])
                      for key in training_inputs if key != 'image_cond_hidden_states'),
                  'amp_only': delta(a, b), 'conditioning_only_amp': delta(a, c),
                  'conditioning_only_no_amp': delta(b, d), 'combined': delta(a, d), 'losses': losses,
                  'relative_loss_difference': abs(losses['training_condition_amp'] - losses['official_condition_no_amp'])
                                               / max(abs(losses['training_condition_amp']), 1e-12)}
        results.append(result)
        low_precision_predictions.append((a, b, c, d))
        print(f"BF16/context replay {item['direction']} seed={item['seed']}: "
              f"repeat_max={result['repeated_training_forward']['max_abs']:.8g} "
              f"combined_relative_l2={result['combined']['relative_l2']:.6g}", flush=True)
    # Upcasting preserves the exact original BF16 weight values; this isolates
    # arithmetic precision, rather than claiming recovery of unknown FP32 weights.
    model.to(dtype=torch.float32)
    for item, result, predictions in zip(records, results, low_precision_predictions):
        snapshot = item['snapshot']
        reference_train = forward(snapshot['training']['inputs'], cast_dtype=torch.float32)
        reference_native = forward({**snapshot['training']['inputs'],
                                    'image_cond_hidden_states': snapshot['pipeline']['inputs']['image_cond_hidden_states']},
                                   cast_dtype=torch.float32)
        repeat = forward(snapshot['training']['inputs'], cast_dtype=torch.float32)
        a, b, c, d = predictions
        result['fp32_arithmetic'] = {
            'repeat': delta(reference_train, repeat),
            'conditioning_only': delta(reference_train, reference_native),
            'training_amp_error': delta(reference_train, a), 'training_no_amp_error': delta(reference_train, b),
            'official_amp_error': delta(reference_native, c), 'official_no_amp_error': delta(reference_native, d),
            'final_bf16_rounding_only': delta(reference_train, reference_train.to(torch.bfloat16).float()),
            'training_condition_loss': mse(reference_train, snapshot['velocity_target']),
            'official_condition_loss': mse(reference_native, snapshot['velocity_target'])}
        print(f"FP32 {item['direction']} seed={item['seed']}: "
              f"amp_error={result['fp32_arithmetic']['training_amp_error']['relative_l2']:.6g} "
              f"no_amp_error={result['fp32_arithmetic']['training_no_amp_error']['relative_l2']:.6g} "
              f"output_rounding={result['fp32_arithmetic']['final_bf16_rounding_only']['relative_l2']:.6g}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained_model_name_or_path', required=True)
    parser.add_argument('--report_root', required=True, help='包含report.json或方向子目录的对照目录，须有保存的tensors文件')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compute_dtype', choices=['bf16', 'fp32'], default='bf16')
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == 'cpu' and args.compute_dtype != 'fp32':
        parser.error('CPU smoke请指定 --compute_dtype fp32')
    dtype = torch.bfloat16 if args.compute_dtype == 'bf16' else torch.float32
    root = Path(args.report_root)
    paths = ([root / 'report.json'] if (root / 'report.json').is_file() else sorted(root.glob('*/report.json')))
    if not paths:
        raise ValueError('没有找到对照report.json')
    records = []
    for path in paths:
        report = read_json(path)
        for case in report['cases']:
            snapshot = torch.load(path.parent / f"tensors-seed-{int(case['seed'])}.pt", map_location='cpu', weights_only=True)
            records.append({'direction': report['direction'], 'seed': case['seed'], 'sigma': case['sigma'],
                            'snapshot': snapshot})
    model_path, _ = validate_local_model(args.pretrained_model_name_or_path)
    output = prepare_output(args.output_dir, protected=[root, model_path])
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    from diffusers import Flux3Transformer2DModel
    model = Flux3Transformer2DModel.from_pretrained(model_path, subfolder='transformer', torch_dtype=dtype,
                                                   local_files_only=True).to(device).requires_grad_(False).eval()
    cases = analyze(model, records, device, dtype)
    result = {'compute_dtype': str(dtype), 'tf32_enabled': False,
              'reference': 'FP32 arithmetic using the SAME representable weights and saved inputs; no new VAE/text encoding',
              'no_lora': not any('lora_' in name for name, _ in model.named_parameters()),
              'parameters_with_grad': sum(p.grad is not None for p in model.parameters()),
              'peak_gpu_gib': torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else None,
              'cases': cases}
    write_json(output / 'numeric_report.json', result)
    return result


if __name__ == '__main__':
    main()
