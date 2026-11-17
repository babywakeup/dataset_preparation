"""Locate the FIRST tensor divergence and test scoped normalization interventions; no gradients/LoRA."""
import argparse
import hashlib
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from compare_flux3_training_pipeline import autocast_context, tensor_difference
from flux3_recolor.config import prepare_output, read_json, resolve_model_root, write_json


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        flat = value.detach().flatten()
        count = min(256, flat.numel())
        # FP32 linspace can round N-1 up to N above 2**24; integer indices avoid out-of-bounds.
        indices = torch.arange(count, device=value.device, dtype=torch.long) * max(flat.numel() - 1, 0) // max(count - 1, 1)
        sample = flat[indices].float().cpu()
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'stride': list(value.stride()),
                'sha256': hashlib.sha256(raw).hexdigest(), 'sample': sample.tolist()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    if isinstance(value, dict):
        return {key: fingerprint(item) for key, item in value.items()}
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(type(value))


def same_values(left, right):
    if isinstance(left, dict) and 'sha256' in left:
        return (left['shape'], left['dtype'], left['sha256']) == (right.get('shape'), right.get('dtype'), right.get('sha256'))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_values(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(same_values(a, b) for a, b in zip(left, right))
    return left == right


@torch.no_grad()
def trace_forward(model, forward, max_leaf_events=80):
    events, handles = [], []
    leaf_events = 0
    def record(name, kind, data, coarse=False):
        nonlocal leaf_events
        if not coarse and leaf_events >= max_leaf_events:
            return
        if not coarse:
            leaf_events += 1
        events.append({'module': name, 'event': kind, 'coarse': coarse, 'tensors': fingerprint(data)})
    for name, module in model.named_modules():
        if not name:
            continue
        coarse = bool(re.fullmatch(r'(txt_mode_blocks\.\d+|mode_blocks\.[^.]+\.\d+|single_transformer_blocks\.\d+)', name))
        leaf = not list(module.children())
        if leaf or coarse:
            handles.append(module.register_forward_pre_hook(
                lambda m, a, k, n=name, c=coarse: record(n, 'pre', {'args': a, 'kwargs': k}, c), with_kwargs=True))
            handles.append(module.register_forward_hook(
                lambda m, a, out, n=name, c=coarse: record(n, 'post', out, c)))
    try:
        result = forward()
    finally:
        for handle in handles:
            handle.remove()
    return result, events


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if (a['module'], a['event']) != (b['module'], b['event']) or not same_values(a['tensors'], b['tensors']):
            return {'event_index': index, 'left': a, 'right': b}
    return None if len(left) == len(right) else {'reason': 'event count mismatch'}


@contextmanager
def disable_norm_autocast(model, classes):
    originals = []
    for name, module in model.named_modules():
        if isinstance(module, classes):
            original = module.forward
            def wrapped(*args, _original=original, **kwargs):
                tensor = next((x for x in args if isinstance(x, torch.Tensor)), None)
                context = torch.autocast(tensor.device.type, enabled=False) if tensor is not None else nullcontext()
                with context:
                    return _original(*args, **kwargs)
            originals.append((name, module, original))
            module.forward = wrapped
    try:
        yield [name for name, _, _ in originals]
    finally:
        for _, module, original in originals:
            module.forward = original


@torch.no_grad()
def capture_module_pair(model, module_name, forward_off, forward_on):
    module = dict(model.named_modules())[module_name]
    records = []
    def before(m, args, kwargs):
        records.append({'inputs': [x.detach().cpu().clone() for x in args if isinstance(x, torch.Tensor)]
                        + [x.detach().cpu().clone() for x in kwargs.values() if isinstance(x, torch.Tensor)]})
    def after(m, args, out):
        records[-1]['output'] = out.detach().cpu().clone()
    handles = [module.register_forward_pre_hook(before, with_kwargs=True), module.register_forward_hook(after)]
    try:
        forward_off()
        split = len(records)
        forward_on()
    finally:
        for handle in handles:
            handle.remove()
    left, right = records[0], records[split]
    return {'module': module_name,
            'input_differences': [tensor_difference(a, b, 1e-5, 1e-4) for a, b in zip(left['inputs'], right['inputs'])],
            'output_difference': tensor_difference(left['output'], right['output'], 1e-5, 1e-4)}


@torch.no_grad()
def verify_transformer_controls(model_path, report_root, device, output):
    from diffusers import Flux3Transformer2DModel
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model = Flux3Transformer2DModel.from_pretrained(model_path, subfolder='transformer', torch_dtype=dtype,
                                                   local_files_only=True).to(device).requires_grad_(False).eval()
    root = Path(report_root)
    paths = [root / 'report.json'] if (root / 'report.json').is_file() else sorted(root.glob('*/report.json'))
    results = []
    for path in paths:
        report = read_json(path)
        for case in report['cases']:
            snapshot = torch.load(path.parent / f"tensors-seed-{case['seed']}.pt", map_location='cpu', weights_only=True)
            inputs = {k: v.to(device=device, dtype=torch.float32 if device.type == 'cpu' else v.dtype)
                      for k, v in snapshot['training']['inputs'].items()}
            plain = model(**inputs, return_dict=False)[0]
            with autocast_context(device, dtype):
                amp = model(**inputs, return_dict=False)[0]
            with disable_norm_autocast(model, (torch.nn.LayerNorm,)), autocast_context(device, dtype):
                controlled = model(**inputs, return_dict=False)[0]
            result = {'direction': report['direction'], 'seed': case['seed'], 'sigma': case['sigma'],
                      'baseline': tensor_difference(plain, amp, 1e-5, 1e-4),
                      'layernorm_intervention': tensor_difference(plain, controlled, 1e-5, 1e-4)}
            results.append(result)
            print(report['direction'], case['seed'], 'baseline relative', result['baseline']['relative_l2'],
                  'after LayerNorm intervention', result['layernorm_intervention']['max_abs'], flush=True)
    write_json(Path(output) / 'transformer_six_case_controls.json', results)
    return results


@torch.no_grad()
def run(model_path, snapshot, config, sample, direction, root, device, output):
    from diffusers import Flux3Transformer2DModel, AutoencoderKLFlux2
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from diffusers.modular_pipelines.flux3.encoders import Flux3VaeEncoderStep
    from diffusers.modular_pipelines.flux3.decoders import Flux3DecodeStep
    from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor
    from types import SimpleNamespace
    from flux3_recolor.inference import load_validation_sample
    from flux3_recolor.data import image_tensor
    from flux3_recolor.modeling import encode_latents
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    difference = lambda a, b: tensor_difference(a, b, 1e-5, 1e-4)
    model = Flux3Transformer2DModel.from_pretrained(model_path, subfolder='transformer', torch_dtype=dtype,
                                                   local_files_only=True).to(device).requires_grad_(False).eval()
    inputs = {k: v.to(device) for k, v in snapshot['training']['inputs'].items()}
    if device.type != 'cuda':
        inputs = {k: v.float() for k, v in inputs.items()}
    def transformer_forward(amp):
        with autocast_context(device, dtype) if amp else nullcontext():
            return model(**inputs, return_dict=False)[0]
    plain, off_trace = trace_forward(model, lambda: transformer_forward(False))
    amp, on_trace = trace_forward(model, lambda: transformer_forward(True))
    first = first_difference(off_trace, on_trace)
    print('Transformer first difference:', None if first is None else
          (first['left']['module'], first['left']['event']), flush=True)
    local = None if first is None else capture_module_pair(
        model, first['left']['module'], lambda: transformer_forward(False), lambda: transformer_forward(True))
    with disable_norm_autocast(model, (torch.nn.LayerNorm,)) as patched:
        controlled = transformer_forward(True)
    result = {'transformer': {'same_exact_input': True, 'first_difference': first,
        'local_module_verification': local, 'baseline_difference': difference(plain, amp),
        'layernorm_autocast_disabled_difference': difference(plain, controlled), 'patched_norms': patched,
        'off_trace': off_trace, 'on_trace': on_trace,
        'saved_training_prediction_replay': difference(snapshot['training']['prediction'], amp)}}
    write_json(output / 'transformer_trace.json', result['transformer'])
    print('Transformer intervention relative error:', result['transformer']['layernorm_autocast_disabled_difference']['relative_l2'], flush=True)
    del model, inputs, plain, amp, controlled
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    vaes = {name: AutoencoderKLFlux2.from_pretrained(model_path, subfolder='vae', torch_dtype=precision,
              local_files_only=True).to(device).requires_grad_(False).eval()
            for name, precision in [('low', dtype), ('official', torch.float32)]}
    _, refs = load_validation_sample(root, sample, config, direction)
    pixels = image_tensor(refs[1]).unsqueeze(0).to(device).clone(memory_format=torch.channels_last)
    low, official = vaes['low'], vaes['official']
    def encode(vae, amp):
        with autocast_context(device, dtype) if amp else nullcontext():
            return encode_latents(vae, pixels)
    enc_off, enc_trace_off = trace_forward(low.encoder, lambda: encode(low, False), 20)
    enc_on, enc_trace_on = trace_forward(low.encoder, lambda: encode(low, True), 20)
    enc_first = first_difference(enc_trace_off, enc_trace_on)
    enc_local = None if enc_first is None else capture_module_pair(low.encoder, enc_first['left']['module'],
                                                                  lambda: encode(low, False), lambda: encode(low, True))
    with disable_norm_autocast(low.encoder, (torch.nn.GroupNorm, torch.nn.LayerNorm)) as enc_patched:
        enc_control = encode(low, True)
    components = SimpleNamespace(vae=official, _execution_device=device)
    state = PipelineState()
    state.set('condition_images', [pixels])
    _, state = Flux3VaeEncoderStep()(components, state)
    official_latent = state.get('image_cond_latents')[0]
    helper_latent = encode(official, False)
    # Identify where official FP32 vs current BF16 starts: input and first convolution.
    conv_low_input = pixels.to(low.dtype)
    conv_official_input = pixels.to(official.dtype)
    conv_low = low.encoder.conv_in(conv_low_input)
    conv_official = official.encoder.conv_in(conv_official_input)
    result['vae_encoder'] = {'same_normalized_rgb': True, 'amp_first_difference': enc_first,
        'local_module_verification': enc_local, 'amp_difference': difference(enc_off, enc_on),
        'groupnorm_autocast_disabled_difference': difference(enc_off, enc_control), 'patched_norms': enc_patched,
        'official_dtype_input_difference': difference(conv_official_input, conv_low_input),
        'first_conv_output_difference': difference(conv_official, conv_low),
        'official_helper_vs_native_encoder': difference(official_latent, helper_latent),
        'off_trace': enc_trace_off, 'on_trace': enc_trace_on}
    write_json(output / 'vae_encoder_trace.json', result['vae_encoder'])
    print('VAE encoder first:', None if enc_first is None else enc_first['left']['module'],
          'norm intervention relative error:', result['vae_encoder']['groupnorm_autocast_disabled_difference']['relative_l2'], flush=True)
    # Decoder is isolated with ONE fixed normalized latent; no upstream prediction change.
    saved_inputs = snapshot['training']['inputs']
    noisy = saved_inputs['hidden_states'].transpose(1, 2).reshape_as(snapshot['target'])
    fixed = (noisy.float() - float(saved_inputs['timestep'].item()) *
             snapshot['training']['prediction'].transpose(1, 2).reshape_as(noisy).float()).to(dtype).to(device)
    def raw_decoder_input(vae):
        mean = vae.bn.running_mean[None, :, None, None].to(fixed)
        std = (vae.bn.running_var[None, :, None, None] + vae.config.batch_norm_eps).sqrt().to(fixed)
        return Flux3DecodeStep._unpatchify_latents(fixed * std + mean).to(vae.dtype)
    raw_low, raw_official = [raw_decoder_input(vae) for vae in (low, official)]
    def decode_low(amp):
        with autocast_context(device, dtype) if amp else nullcontext():
            return low.decode(raw_low, return_dict=False)[0]
    dec_off, dec_trace_off = trace_forward(low.decoder, lambda: decode_low(False), 20)
    dec_on, dec_trace_on = trace_forward(low.decoder, lambda: decode_low(True), 20)
    dec_first = first_difference(dec_trace_off, dec_trace_on)
    dec_local = None if dec_first is None else capture_module_pair(low.decoder, dec_first['left']['module'],
                                                                  lambda: decode_low(False), lambda: decode_low(True))
    with disable_norm_autocast(low.decoder, (torch.nn.GroupNorm, torch.nn.LayerNorm)) as dec_patched:
        dec_control = decode_low(True)
    decoded_official = official.decode(raw_official, return_dict=False)[0]
    processor = Flux3ImageProcessor()
    state = PipelineState()
    state.set('latents', fixed)
    state.set('output_type', 'pt')
    state.set('reference_crop', None)
    _, state = Flux3DecodeStep()(SimpleNamespace(vae=official, image_processor=processor), state)
    result['vae_decoder'] = {'fixed_normalized_latent': fingerprint(fixed), 'amp_first_difference': dec_first,
        'local_module_verification': dec_local, 'amp_difference': difference(dec_off, dec_on),
        'groupnorm_autocast_disabled_difference': difference(dec_off, dec_control), 'patched_norms': dec_patched,
        'inverse_bn_and_unpatchify_input_difference': difference(raw_official, raw_low),
        'current_vs_official_output_difference': difference(decoded_official, dec_on),
        'official_manual_vs_native_decoder': difference(state.get('images'), processor.postprocess(decoded_official, output_type='pt')),
        'off_trace': dec_trace_off, 'on_trace': dec_trace_on}
    write_json(output / 'vae_decoder_trace.json', result['vae_decoder'])
    print('VAE decoder first:', None if dec_first is None else dec_first['left']['module'],
          'norm intervention relative error:', result['vae_decoder']['groupnorm_autocast_disabled_difference']['relative_l2'], flush=True)
    write_json(output / 'stage_summary.json', {k: {a: b for a, b in v.items() if not a.endswith('_trace')}
                                              for k, v in result.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained_model_name_or_path', required=True)
    parser.add_argument('--report_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--instance_data_dir', default='./test')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    model_path = str(resolve_model_root(args.pretrained_model_name_or_path))
    report_dir = Path(args.report_dir)
    config = read_json(report_dir / 'report.json')
    snapshot = torch.load(report_dir / f'tensors-seed-{args.seed}.pt', map_location='cpu', weights_only=True)
    output = prepare_output(args.output_dir, protected=[model_path, report_dir, args.instance_data_dir])
    return run(model_path, snapshot, config, config['sample'], config['direction'], args.instance_data_dir,
               torch.device(args.device), output)


if __name__ == '__main__':
    main()
