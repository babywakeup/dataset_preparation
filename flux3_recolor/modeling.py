# Copyright 2026 HuggingFace Inc. and project contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Latent packing, normalization, timestep recipe and LoRA targets adapted from
# diffusers/examples/dreambooth/train_dreambooth_lora_flux3_klein_img2img.py.
"""FLUX.3 math shared by training and CPU interface tests."""

import torch

FLOW_SHIFT = 6.93
LORA_TARGETS = [
    'to_q', 'to_k', 'to_v', 'to_out.0',
    'ff.linear_in', 'ff.linear_out',
    'early_stream_modulations.image_cond.linear',
    'single_stream_modulations.image_cond.linear',
]

def patchify_latents(latents):
    b, c, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError('VAE latent 宽高必须能被 2 整除')
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def pack_latents(latents):
    return latents.flatten(2).transpose(1, 2)


def image_position_ids(height, width, device, l_coord=0):
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([l_coord]))
    ids = ids.to(device=device, dtype=torch.float32)
    ids[:, 1:3] *= 0.5
    return ids


def text_position_ids(length, device):
    ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, device=device)
    return ids


def reference_tokens(latents):
    if len(latents) != 3:
        raise ValueError('需要三组独立条件 latent')
    tokens, ids = [], []
    for index, latent in enumerate(latents):
        tokens.append(pack_latents(latent))
        ids.append(image_position_ids(*latent.shape[-2:], latent.device, l_coord=-(index + 1)))
    return torch.cat(tokens, dim=1), torch.cat(ids, dim=0)


@torch.no_grad()
def encode_latents(vae, pixels):
    pixels = pixels.to(device=vae.device, dtype=vae.dtype).contiguous(memory_format=torch.channels_last)
    with torch.autocast(pixels.device.type, enabled=False):
        latent = patchify_latents(vae.encode(pixels).latent_dist.mode())
        mean = vae.bn.running_mean.to(latent).view(1, -1, 1, 1)
        std = (vae.bn.running_var.to(latent).view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt()
        return (latent - mean) / std


@torch.no_grad()
def encode_recolor_latents(vae, target, reference_pixels):
    """Encode the aligned scenes together and the independent swatch separately."""
    if len(reference_pixels) != 3:
        raise ValueError('需要三张条件图')
    if target.ndim != 4 or any(pixels.ndim != 4 for pixels in reference_pixels):
        raise ValueError('图片必须为 BCHW tensor')
    batch = target.shape[0]
    if batch < 1 or any(pixels.shape[0] != batch for pixels in reference_pixels):
        raise ValueError('目标与条件 batch 大小必须一致且非空')
    if any(pixels.shape != target.shape for pixels in reference_pixels[:2]):
        raise ValueError('目标与条件场景形状必须一致')
    if reference_pixels[2].shape[1] != target.shape[1]:
        raise ValueError('目标与色块通道数必须一致')
    scenes = encode_latents(vae, torch.cat([target, *reference_pixels[:2]], dim=0))
    target_latent, gray_latent, source_latent = scenes.split(batch, dim=0)
    swatch_latent = encode_latents(vae, reference_pixels[2])
    return target_latent, [gray_latent, source_latent, swatch_latent]


def validate_vae_grid(vae):
    from .data import GRID
    actual = 2 ** (len(vae.config.block_out_channels) - 1) * 2
    if actual != GRID or tuple(vae.config.patch_size) != (2, 2):
        raise ValueError(f'当前预处理要求 token 网格 {GRID}、patch_size=(2,2)，实际网格为 {actual}')


def add_lora(transformer, rank, alpha, dropout, target_modules=None):
    from peft import LoraConfig
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        init_lora_weights='gaussian', target_modules=target_modules or LORA_TARGETS,
    ))
    # Optimizer master weights in FP32, including when the frozen base is FP16/BF16.
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.float()


def flow_matching_loss(model, target, references, embeddings, text_ids, scheduler,
                       weighting_scheme='none', logit_mean=0., logit_std=1., mode_scale=1.29,
                       return_per_sample=False, compute_dtype=None):
    from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
    batch = target.shape[0]
    noise = torch.randn_like(target)
    u = compute_density_for_timestep_sampling(weighting_scheme, batch, logit_mean=logit_mean,
                                              logit_std=logit_std, mode_scale=mode_scale, device=target.device)
    indices = (u * scheduler.config.num_train_timesteps).long().clamp_max(scheduler.config.num_train_timesteps - 1)
    sigmas = scheduler.sigmas.to(device=target.device, dtype=target.dtype)[indices].reshape(batch, 1, 1, 1)
    noisy = (1 - sigmas) * target + sigmas * noise
    cond_tokens, cond_ids = reference_tokens(references)
    forward_dtype = compute_dtype or target.dtype
    pred = model(
        hidden_states=pack_latents(noisy).to(forward_dtype), img_ids=image_position_ids(*target.shape[-2:], target.device),
        timestep=sigmas.reshape(batch).to(forward_dtype), image_cond_hidden_states=cond_tokens.to(forward_dtype), image_cond_ids=cond_ids,
        encoder_hidden_states=embeddings.to(forward_dtype), txt_ids=text_ids, return_dict=False,
    )[0]
    velocity = pack_latents(noise - target)
    if pred.shape != velocity.shape:
        raise ValueError(f'预测与velocity目标形状不一致: prediction={tuple(pred.shape)}, target={tuple(velocity.shape)}')
    weighting = compute_loss_weighting_for_sd3(weighting_scheme, sigmas.reshape(batch, 1, 1))
    per_sample = (weighting.float() * (pred.float() - velocity.float()).square()).mean(dim=(1, 2))
    return per_sample if return_per_sample else per_sample.mean()


@torch.no_grad()
def get_qwen3vl_prompt_embeds(text_encoder, tokenizer, prompt, device, dtype,
                             text_encoder_out_layers, max_sequence_length, pad_to_length=None):
    """Reference batch window and chat template, with explicit padding position IDs."""
    import math
    prompts = [prompt] if isinstance(prompt, str) else prompt
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                           add_generation_prompt=True) for p in prompts]
    lengths = [len(tokenizer.tokenizer(t, truncation=True, max_length=max_sequence_length)['input_ids']) for t in texts]
    window = min(math.ceil(max(lengths) / 80) * 80, max_sequence_length)
    if pad_to_length is not None:
        window = min(max(window, pad_to_length), max_sequence_length)
    inputs = tokenizer.tokenizer(texts, return_tensors='pt', padding='max_length', padding_side='right',
                                  truncation=True, max_length=window)
    ids, attention = inputs['input_ids'].to(device), inputs['attention_mask'].to(device)
    positions = torch.arange(window, device=device)[None].expand_as(ids).masked_fill(attention == 0, 1)
    output = text_encoder(input_ids=ids, attention_mask=attention, position_ids=positions,
                           output_hidden_states=True, use_cache=False)
    hidden = torch.stack([output.hidden_states[i] for i in text_encoder_out_layers], dim=1).to(dtype=dtype)
    b, layers, length, channels = hidden.shape
    return hidden.permute(0, 2, 1, 3).reshape(b, length, layers * channels)


def save_adapter(transformer, output):
    from diffusers.loaders import Flux3LoraLoaderMixin
    from peft.utils import get_peft_model_state_dict
    state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(transformer).items()}
    Flux3LoraLoaderMixin.save_lora_weights(save_directory=str(output), transformer_lora_layers=state,
                                         transformer_lora_adapter_metadata=transformer.peft_config['default'].to_dict(),
                                         safe_serialization=True)


def load_adapter_state(transformer, path):
    from diffusers.loaders import Flux3LoraLoaderMixin
    from peft import set_peft_model_state_dict
    from diffusers.utils import convert_unet_state_dict_to_peft
    state = Flux3LoraLoaderMixin.lora_state_dict(str(path), local_files_only=True)
    state = {k.removeprefix('transformer.'): v for k, v in state.items() if k.startswith('transformer.')}
    if not state:
        raise ValueError('checkpoint 没有 transformer LoRA 权重')
    result = set_peft_model_state_dict(transformer, convert_unet_state_dict_to_peft(state), adapter_name='default')
    missing_lora = [k for k in result.missing_keys if 'lora_' in k]
    if result.unexpected_keys or missing_lora:
        raise ValueError(f'LoRA 权重不匹配: missing={missing_lora}, unexpected={result.unexpected_keys}')


@torch.no_grad()
def encode_prompt(text_encoder, tokenizer, prompt, device, dtype, layers, max_length):
    from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
    embeddings = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
        text_encoder=text_encoder, tokenizer=tokenizer, prompt=prompt, device=device,
        hidden_states_layers=tuple(layers), max_sequence_length=max_length,
    ).to(dtype=dtype)
    return embeddings, text_position_ids(embeddings.shape[1], device)


def load_components(model_path, dtype, device):
    from pathlib import Path
    from .config import resolve_model_root, tokenizer_subfolder
    from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel, FlowMatchEulerDiscreteScheduler
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    kwargs = {'local_files_only': True}
    model_path = str(resolve_model_root(model_path))
    from .precision import set_transformer_autocast
    vae = AutoencoderKLFlux2.from_pretrained(model_path, subfolder='vae', torch_dtype=torch.float32, **kwargs)
    validate_vae_grid(vae)
    transformer = Flux3Transformer2DModel.from_pretrained(model_path, subfolder='transformer', torch_dtype=dtype, **kwargs)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, subfolder='text_encoder', torch_dtype=dtype, **kwargs)
    tokenizer = AutoProcessor.from_pretrained(str(Path(model_path) / tokenizer_subfolder(model_path)), **kwargs)
    set_transformer_autocast(transformer, dtype)
    for model in (vae, transformer, text_encoder):
        model.requires_grad_(False).eval().to(device)
    return {
        'vae': vae, 'transformer': transformer, 'text_encoder': text_encoder, 'tokenizer': tokenizer,
        'scheduler': FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT, use_dynamic_shifting=False),
    }
