"""No-grad, no-LoRA differential audit of the real training helper and native FLUX.3 pipeline.

This compares one conditional velocity evaluation at the SAME noisy latent and sigma,
not a training loss to a multi-step, classifier-free-guided final photograph.
No tensor-layout canonicalization or prediction replacement is used in the native comparison.
"""
import argparse
import copy
from contextlib import contextmanager, nullcontext
from pathlib import Path

# 支持从项目根目录直接执行此子目录脚本。
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from flux3_recolor import DEFAULT_PROMPT, task_folders
from flux3_recolor.config import prepare_output, validate_local_model, write_json
from flux3_recolor.inference import load_validation_sample
from flux3_recolor.modeling import (encode_latents, flow_matching_loss, get_qwen3vl_prompt_embeds,
                                   load_components, pack_latents, text_position_ids)


def tensor_summary(value):
    value = value.detach().cpu()
    data = value.float()
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'stride': list(value.stride()),
            'finite': bool(torch.isfinite(data).all()), 'mean': data.mean().item(),
            'std': data.std(unbiased=False).item(), 'min': data.min().item(), 'max': data.max().item()}


def tensor_difference(left, right, atol, rtol):
    left, right = left.detach().cpu(), right.detach().cpu()
    result = {'left': tensor_summary(left), 'right': tensor_summary(right), 'atol': atol, 'rtol': rtol,
              'exact_equal': left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)}
    if left.shape != right.shape:
        return {**result, 'allclose': False, 'reason': 'shape mismatch'}
    delta = left.float() - right.float()
    return {**result, 'allclose': torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol),
            'max_abs': delta.abs().max().item(), 'mean_abs': delta.abs().mean().item(),
            'rmse': delta.square().mean().sqrt().item(),
            'relative_l2': (delta.norm() / left.float().norm().clamp_min(1e-12)).item()}


@contextmanager
def capture_forwards(model):
    calls = []
    def before(module, args, kwargs):
        calls.append({'inputs': {key: value.detach().cpu().clone() for key, value in kwargs.items()
                                 if isinstance(value, torch.Tensor)},
                      'input_strides': {key: list(value.stride()) for key, value in kwargs.items()
                                        if isinstance(value, torch.Tensor)},
                      'training': module.training, 'grad_enabled': torch.is_grad_enabled()})
    def after(module, args, output):
        calls[-1]['prediction'] = output[0].detach().cpu().clone()
    handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after)]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def autocast_context(device, dtype):
    return torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' and dtype != torch.float32 else nullcontext()


def native_pipeline(components, device, model_path, vae_precision):
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines.flux3 import Flux3AutoBlocks
    if (Path(model_path) / 'modular_model_index.json').is_file():
        pipeline = ModularPipeline.from_pretrained(model_path, local_files_only=True)
        # `blocks` returns a deep copy in the installed Diffusers; edit the
        # actual execution tree, otherwise the unrequested upsampler still runs.
        pipeline._blocks.sub_blocks.pop('prompt_upsampler')
    else:  # Offline tiny fixtures and old repositories without pipeline metadata.
        blocks = Flux3AutoBlocks()
        blocks.sub_blocks.pop('prompt_upsampler')
        pipeline = blocks.init_pipeline()
    # Keep the official text encoder, RGB processor, VAE and position-ID blocks.
    supplied = {**components, 'scheduler': copy.deepcopy(components['scheduler'])}
    if vae_precision == 'fp32':
        supplied.pop('vae')
    pipeline.update_components(**supplied)
    if vae_precision == 'fp32':
        # Load ORIGINAL FP32 VAE weights; upcasting the training BF16 VAE would
        # not recover the original weight precision and is not the official recipe.
        pipeline.load_components(names=['vae'], dtype={'vae': torch.float32, 'default': components['transformer'].dtype},
                                 pretrained_model_name_or_path=model_path, subfolder='vae', local_files_only=True)
        if pipeline.vae is None:
            # Older/tiny repos may not declare the VAE load spec; load it explicitly
            # with the same original-weight precision instead of accepting an absent component.
            from diffusers import AutoencoderKLFlux2
            pipeline.update_components(vae=AutoencoderKLFlux2.from_pretrained(
                model_path, subfolder='vae', torch_dtype=torch.float32, local_files_only=True))
        pipeline.vae.requires_grad_(False).eval()
    pipeline.update_components(guider=pipeline.guider.new(enabled=False))
    pipeline.to(device)
    return pipeline


@torch.no_grad()
def compare_sample(components, training_args, sample, device, dtype, seeds, atol, rtol, output, save_tensors,
                   model_path, pipeline_vae_dtype, pipeline_autocast):
    from train_flux3_recolor_lora import DreamBoothDataset, collate_fn, cuda_device_index
    from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
    model, vae, encoder = [components[key] for key in ('transformer', 'vae', 'text_encoder')]
    if any('lora_' in name for name, _ in model.named_parameters()):
        raise ValueError('对照只允许不含LoRA的基础模型')
    for module in (model, vae, encoder):
        module.requires_grad_(False).eval()
    from datasets import Dataset
    from build_flux3_recolor_dataset import generate_samples
    rows = list(generate_samples(training_args.instance_data_dir, training_args.training_directions[0], [sample], 0))
    dataset = DreamBoothDataset(training_args, Dataset.from_list(rows))
    batch = collate_fn([dataset[0]])
    config = {'preprocessing': dataset.preprocessing, 'max_area': training_args.resolution ** 2}
    _, reference_images = load_validation_sample(Path(training_args.instance_data_dir), sample, config, training_args.training_directions[0])
    vae_pixels = []
    def capture_pixels(module, args):
        vae_pixels.append({'tensor': args[0].detach().cpu().clone(), 'stride': list(args[0].stride())})
    handles = [vae.encoder.register_forward_pre_hook(capture_pixels)]
    try:
        # Identical Dataset -> collate -> VAE -> text helper path as training; no autocast around VAE/text.
        target = encode_latents(vae, batch['pixel_values'])
        references = [encode_latents(vae, pixels) for pixels in batch['cond_pixel_values']]
        embeddings = get_qwen3vl_prompt_embeds(encoder, components['tokenizer'], batch['prompts'], device, dtype,
            training_args.text_encoder_out_layers, training_args.max_sequence_length)
        pipeline = native_pipeline(components, device, model_path, pipeline_vae_dtype)
        if pipeline.vae is not vae:
            handles.append(pipeline.vae.encoder.register_forward_pre_hook(capture_pixels))
        rgb_preprocessing = {}
        for i, (pixels, image) in enumerate(zip(batch['cond_pixel_values'], reference_images)):
            native_rgb, _ = pipeline.image_processor.preprocess(image, resize_mode='crop')
            rgb_preprocessing[f'condition_{i + 1}'] = tensor_difference(pixels, native_rgb, atol, rtol)
        cases = []
        devices = [cuda_device_index(device)] if device.type == 'cuda' else []
        for seed in seeds:
            # Replay exactly the real helper's RNG order, without monkey-patching its noise or sampler.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                noise = torch.randn_like(target)
                u = compute_density_for_timestep_sampling(training_args.weighting_scheme, 1,
                    logit_mean=training_args.logit_mean, logit_std=training_args.logit_std,
                    mode_scale=training_args.mode_scale, device=device)
            index = (u * components['scheduler'].config.num_train_timesteps).long().clamp_max(
                components['scheduler'].config.num_train_timesteps - 1)
            sigma = components['scheduler'].sigmas.to(device=device, dtype=target.dtype)[index].reshape(1, 1, 1, 1)
            noisy = (1 - sigma) * target + sigma * noise
            velocity = pack_latents(noise - target).cpu()
            weight = compute_loss_weighting_for_sd3(training_args.weighting_scheme, sigma.reshape(1, 1, 1)).cpu()
            model.train()  # Actual training mode, even though autograd is disabled.
            with torch.random.fork_rng(devices=devices), capture_forwards(model) as training_calls:
                torch.manual_seed(seed)
                with autocast_context(device, dtype):
                    training_loss = flow_matching_loss(model, target, references, embeddings,
                        text_position_ids(embeddings.shape[1], device), components['scheduler'],
                        weighting_scheme=training_args.weighting_scheme, logit_mean=training_args.logit_mean,
                        logit_std=training_args.logit_std, mode_scale=training_args.mode_scale,
                        compute_dtype=dtype)
            training = training_calls[0]
            prediction = training['prediction']
            manual_loss = (weight.float() * (prediction.float() - velocity.float()).square()).mean()
            model.eval()
            # Sigma has ALREADY been shifted by the training scheduler. Supply its inverse-shift
            # to the native pipeline so its normal schedule block shifts it exactly once.
            shift = components['scheduler'].config.shift
            sigma_value = sigma.item()
            raw_sigma = sigma_value / (shift - (shift - 1) * sigma_value)
            pixel_start = len(vae_pixels)
            native_context = autocast_context(device, dtype) if pipeline_autocast == 'training' else nullcontext()
            with capture_forwards(model) as pipeline_calls, native_context:
                state = pipeline(image=reference_images, prompt=batch['prompts'][0],
                    max_sequence_length=training_args.max_sequence_length,
                    text_encoder_out_layers=tuple(training_args.text_encoder_out_layers),
                    height=target.shape[-2] * 16, width=target.shape[-1] * 16,
                    latents=noisy, sigmas=[raw_sigma], num_inference_steps=1, output_type='latent')
            if len(pipeline_calls) != 1:
                raise ValueError(f'应只有一个条件前向（无CFG），实际为{len(pipeline_calls)}')
            native = pipeline_calls[0]
            inputs = {key: tensor_difference(value, native['inputs'][key], atol, rtol)
                      for key, value in training['inputs'].items()}
            for key in inputs:
                inputs[key]['training_input_stride'] = training['input_strides'][key]
                inputs[key]['pipeline_input_stride'] = native['input_strides'][key]
            pipeline_loss = (weight.float() * (native['prediction'].float() - velocity.float()).square()).mean()
            # Extra diagnostic ONLY: replay identical Transformer inputs in eval mode. This does
            # not replace any input or prediction in the unmodified native pipeline comparison above.
            replay_context = autocast_context(device, dtype) if pipeline_autocast == 'training' else nullcontext()
            with replay_context:
                replay = model(**{key: value.to(device) for key, value in training['inputs'].items()}, return_dict=False)[0]
            with autocast_context(device, dtype):
                training_context_replay = model(
                    **{key: value.to(device) for key, value in training['inputs'].items()}, return_dict=False)[0]
            native_pixels = vae_pixels[pixel_start:]
            pixels = {f'condition_{i + 1}': tensor_difference(vae_pixels[i + 1]['tensor'], p['tensor'], atol, rtol)
                      for i, p in enumerate(native_pixels)}
            one_step = noisy.float().cpu() - sigma_value * prediction.float().transpose(1, 2).reshape_as(target.cpu())
            case = {'seed': seed, 'training_timestep_index': index.item(), 'sigma': sigma_value,
                    'pipeline_sigma': pipeline.scheduler.sigmas[0].item(), 'raw_sigma_supplied_to_pipeline': raw_sigma,
                    'pipeline_forward_calls': len(pipeline_calls), 'pixels': pixels, 'inputs': inputs,
                    'text_embeddings': inputs['encoder_hidden_states'],
                    'training_prediction': tensor_summary(prediction),
                    'prediction_difference': tensor_difference(prediction, native['prediction'], atol, rtol),
                    'matched_input_replay': tensor_difference(prediction, replay, atol, rtol),
                    'matched_input_replay_context': pipeline_autocast,
                    'training_context_replay': tensor_difference(prediction, training_context_replay, atol, rtol),
                    'training_loss': training_loss.item(), 'pipeline_prediction_loss': pipeline_loss.item(),
                    'manual_loss_vs_training_loss': tensor_difference(training_loss, manual_loss, atol, rtol),
                    'one_step_latent_difference': tensor_difference(one_step.to(dtype), state.get('latents'), atol, rtol),
                    'all_inputs_close': all(d['allclose'] for d in inputs.values())}
            cases.append(case)
            print(f"seed={seed} sigma={sigma_value:.8g} train_loss={training_loss.item():.8g} "
                  f"pipeline_loss={pipeline_loss.item():.8g} inputs_close={case['all_inputs_close']} "
                  f"prediction_max_abs={case['prediction_difference']['max_abs']:.8g} "
                  f"matched_replay_close={case['matched_input_replay']['allclose']}")
            if save_tensors:
                torch.save({'training': training, 'pipeline': native, 'noise': noise.cpu(),
                            'target': target.cpu(), 'velocity_target': velocity}, output / f'tensors-seed-{seed}.pt')
        # Controlled probes are separate from the ACTUAL path above. They do not replace
        # pipeline inputs; they identify which execution difference explains its reference tokens.
        probes = {}
        start = 0
        for index, (pixels, ref) in enumerate(zip(batch['cond_pixel_values'], references)):
            count = ref.shape[-2] * ref.shape[-1]
            native_tokens = native['inputs']['image_cond_hidden_states'][:, start:start + count]
            probes[f'condition_{index + 1}'] = {}
            for layout in ('contiguous', 'channels_last'):
                layout_pixels = pixels.clone(memory_format=(torch.contiguous_format if layout == 'contiguous'
                                                           else torch.channels_last))
                for use_autocast in (False, True):
                    context = autocast_context(device, dtype) if use_autocast else nullcontext()
                    with context:
                        tokens = pack_latents(encode_latents(vae, layout_pixels))
                    key = f'{layout}_autocast_{use_autocast}'
                    probes[f'condition_{index + 1}'][key] = tensor_difference(tokens, native_tokens, atol, rtol)
            official_pixels = pixels.clone(memory_format=torch.channels_last)
            official_tokens = pack_latents(encode_latents(pipeline.vae, official_pixels)).to(model.dtype)
            probes[f'condition_{index + 1}']['pipeline_vae_no_autocast_channels_last'] = tensor_difference(
                official_tokens, native_tokens, atol, rtol)
            start += count
    finally:
        for handle in handles:
            handle.remove()
        model.eval()
    return {'sample': sample, 'direction': training_args.training_directions[0], 'seeds': seeds,
            'preprocessing': dataset.preprocessing, 'prompt': batch['prompts'][0], 'dtype': str(dtype),
            'rgb_preprocessing': rgb_preprocessing,
            'grad_enabled': torch.is_grad_enabled(), 'lora_parameter_count': 0,
            'training_vae_dtype': str(vae.dtype), 'pipeline_vae_dtype': str(pipeline.vae.dtype),
            'pipeline_autocast': pipeline_autocast, 'prompt_upsampler_enabled': False, 'pipeline_cfg_enabled': False,
            'pipeline_class': pipeline.__class__.__name__,
            'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'parameters_with_grad': sum(p.grad is not None for p in model.parameters()),
            'comparison': 'one conditional velocity evaluation, CFG disabled, same noisy latent and shifted sigma',
            'tolerances': {'atol': atol, 'rtol': rtol}, 'reference_encoding_probes': probes, 'cases': cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained_model_name_or_path', required=True)
    parser.add_argument('--instance_data_dir', default='./test')
    parser.add_argument('--sample', required=True, help='完整文件名，例如00047_00.png')
    parser.add_argument('--direction', choices=['a_to_b', 'b_to_a'], default='a_to_b')
    parser.add_argument('--output_dir', required=True, help='不存在或为空的新目录')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mixed_precision', choices=['no', 'bf16', 'fp16'], default='bf16')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--color_swatch_size', type=int, default=128)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--max_sequence_length', type=int, default=512)
    parser.add_argument('--text_encoder_out_layers', type=int, nargs='+', default=[4, 8, 12, 16, 20, 24, 28, 32])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--save_tensors', action='store_true')
    parser.add_argument('--pipeline_vae_dtype', choices=['fp32', 'same'], default='fp32',
                        help='默认遵循官方示例的FP32 VAE；same复现当前验证的VAE精度')
    parser.add_argument('--pipeline_autocast', choices=['off', 'training'], default='off',
                        help='官方示例默认不包全局autocast；training复现当前训练中验证上下文')
    args = parser.parse_args(argv)
    if Path(args.sample).name != args.sample or args.sample in {'.', '..'}:
        parser.error('sample必须是完整文件名，不能包含目录')
    device = torch.device(args.device)
    if device.type == 'cpu' and args.mixed_precision != 'no':
        parser.error('CPU请使用 --mixed_precision no')
    dtype = {'no': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    from train_flux3_recolor_lora import parse_args
    model_path, _ = validate_local_model(args.pretrained_model_name_or_path)
    training_args = parse_args(['--pretrained_model_name_or_path', model_path,
        '--instance_data_dir', args.instance_data_dir, '--training_directions', args.direction,
        '--resolution', str(args.resolution), '--color_swatch_size', str(args.color_swatch_size),
        '--prompt', args.prompt, '--max_sequence_length', str(args.max_sequence_length),
        '--text_encoder_out_layers', *map(str, args.text_encoder_out_layers), '--report_to', 'none'])
    # Check only the selected sample, before loading any weights.
    target_folder, condition_folders = task_folders(args.direction)
    for folder in (target_folder, *condition_folders):
        if not (Path(args.instance_data_dir) / folder / args.sample).is_file():
            raise FileNotFoundError(f'缺少样本: {folder}/{args.sample}')
    output = prepare_output(args.output_dir, protected=[model_path, args.instance_data_dir])
    components = load_components(model_path, dtype, device)
    report = compare_sample(components, training_args, args.sample, device, dtype,
                            args.seeds, args.atol, args.rtol, output, args.save_tensors,
                            model_path, args.pipeline_vae_dtype, args.pipeline_autocast)
    write_json(output / 'report.json', report)
    print(f'对照报告已保存到 {output / "report.json"}；不是最终多步生成图质量评估。')
    return report


if __name__ == '__main__':
    main()
