#!/usr/bin/env python
# Copyright 2026 HuggingFace Inc. and project contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""FLUX.3 LoRA training adapted from the Diffusers DreamBooth img2img example.

Source: diffusers/examples/dreambooth/train_dreambooth_lora_flux3_klein_img2img.py
Organization: args -> Dataset/collate/sampler -> Accelerator/models/PEFT ->
optimizer/scheduler -> DataLoader/cache -> prepare -> accumulate -> save/validate.
Task differences: local paired folders, three references, identity caches,
deterministic geometry, exact data-position resume and local-only model loading.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from flux3_recolor import DEFAULT_PROMPT, PROMPT_VARIANTS, subdataset_config
from flux3_recolor.config import read_json, write_json, prepare_output, validate_local_model
from flux3_recolor.data import load_rgb, image_tensor
from torch.utils.data import Dataset
from flux3_recolor.samplers import BucketBatchSampler, ScheduledMixBucketBatchSampler

logger = logging.getLogger(__name__)
TRAINER = 'flux3-recolor-reference'
STATE_VERSION = 2


# ------------------------------- Arguments ---------------------------------
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='FLUX.3 三参考图换色 LoRA（参考 DreamBooth 框架）')
    parser.add_argument('--pretrained_model_name_or_path', help='本地基础模型；init/resume时可从组件配置读取')
    parser.add_argument('--instance_data_dir', help='本地四目录根目录，默认 ./test')
    parser.add_argument('--dataset_root', help='instance_data_dir 兼容别名')
    parser.add_argument('--instance_prompt')
    parser.add_argument('--prompt', help='instance_prompt 兼容别名')
    parser.add_argument('--prompt_augmentation', action='store_true', help='训练使用6种固定语义变体；显式prompt/caption优先')
    parser.add_argument('--training_directions', nargs='+', choices=['b_to_a', 'a_to_b'],
                        default=['b_to_a', 'a_to_b'], help='默认合并两个方向；仅旧方向用 b_to_a')
    parser.add_argument('--atob_ratio_start', type=float, default=0.5,
                        help='混合训练起始epoch的AtoB采样占比，默认0.5')
    parser.add_argument('--atob_ratio_end', type=float, default=0.25,
                        help='混合训练末尾epoch的AtoB采样占比，默认0.25')
    parser.add_argument('--mix_schedule_epochs', type=int,
                        help='比例变化覆盖的epoch数；默认等于首次运行计划epoch数')
    parser.add_argument('--caption_json', help='可选：完整文件名到 caption 的 JSON')
    parser.add_argument('--groups_json')
    parser.add_argument('--validation_fraction', type=float, default=0.1)
    parser.add_argument('--output_dir')
    parser.add_argument('--revision')
    parser.add_argument('--variant')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, default=512, help='默认等比面积预算的平方根')
    parser.add_argument('--color_swatch_size', type=int, default=128, help='第三张颜色条件的边长，默认128；不改原文件')
    parser.add_argument('--use_aspect_ratio_buckets', action='store_true')
    parser.add_argument('--aspect_ratio_buckets', help='参考格式 H,W;H,W，或 HxW,HxW；16倍数')
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--random_flip', action='store_true')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--max_train_steps', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--scale_lr', action='store_true')
    parser.add_argument('--lr_scheduler', default='constant', choices=[
        'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'])
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--lr_num_cycles', type=float, default=1)
    parser.add_argument('--lr_power', type=float, default=1)
    parser.add_argument('--optimizer', type=str.lower, choices=['adamw', 'prodigy'], default='adamw')
    parser.add_argument('--use_8bit_adam', action='store_true')
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--prodigy_beta3', type=float)
    parser.add_argument('--prodigy_decouple', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--prodigy_use_bias_correction', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--prodigy_safeguard_warmup', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--lora_alpha', type=float)
    parser.add_argument('--lora_dropout', type=float)
    parser.add_argument('--lora_layers', help='逗号分隔 LoRA 目标模块')
    parser.add_argument('--weighting_scheme', choices=['none', 'sigma_sqrt', 'logit_normal', 'mode', 'cosmap'], default='none')
    parser.add_argument('--logit_mean', type=float, default=0.)
    parser.add_argument('--logit_std', type=float, default=1.)
    parser.add_argument('--mode_scale', type=float, default=1.29)
    parser.add_argument('--train_text_encoder', action='store_true', help='训练自定义text_encoder；独立对照入口自动启用')
    parser.add_argument('--caption_dropout', type=float, default=0.)
    parser.add_argument('--max_sequence_length', type=int, default=512)
    parser.add_argument('--text_encoder_out_layers', type=int, nargs='+', default=[4, 8, 12, 16, 20, 24, 28, 32])
    parser.add_argument('--offload', action='store_true', help='VAE/text_encoder 闲置时回CPU')
    parser.add_argument('--allow_tf32', action='store_true')
    parser.add_argument('--bnb_quantization_config_path')
    parser.add_argument('--do_fp8_training', action='store_true')
    parser.add_argument('--fsdp_text_encoder', action='store_true')
    parser.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], default=None)
    parser.add_argument('--upcast_before_saving', action='store_true')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--checkpointing_steps', type=int, default=100)
    parser.add_argument('--checkpoints_total_limit', type=int)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--resume_from_checkpoint')
    initialization.add_argument('--init_from', help='从训练导出或checkpoint仅继承权重，使用新优化器和step=0')
    parser.add_argument('--logging_steps', type=int, default=10, help='每N个优化器更新步记录区间平均loss，默认10')
    parser.add_argument('--logging_dir', default='logs')
    parser.add_argument('--report_to', default='tensorboard', help='none/tensorboard/wandb/comet_ml/all')
    parser.add_argument('--validation_prompt')
    parser.add_argument('--final_validation_prompt')
    parser.add_argument('--validation_epochs', type=int, default=1)
    parser.add_argument('--validation_steps', type=int, default=100, help='每N优化步更新验证图，0关闭按步验证')
    parser.add_argument('--num_validation_images', type=int, default=4, help='留出验证集总出图数')
    parser.add_argument('--validation_inference_steps', type=int, default=28)
    parser.add_argument('--skip_final_inference', '--skip_final_validation', dest='skip_final_inference', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args(input_args)
    if len(set(args.training_directions)) != len(args.training_directions):
        parser.error('training_directions 不能重复')
    for name in ('atob_ratio_start', 'atob_ratio_end'):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f'{name} 必须在[0,1]范围内')
    custom_prompt = args.instance_prompt is not None or args.prompt is not None
    for canonical, alias, default in [('instance_data_dir', 'dataset_root', './test'),
                                      ('instance_prompt', 'prompt', DEFAULT_PROMPT)]:
        first, second = getattr(args, canonical), getattr(args, alias)
        if first is not None and second is not None and first != second:
            parser.error(f'--{canonical} 与 --{alias} 不能设置不同值')
        setattr(args, canonical, first if first is not None else second if second is not None else default)
        delattr(args, alias)
    source = args.init_from
    if args.resume_from_checkpoint:
        source = resolve_training_checkpoint(args.output_dir or 'outputs/flux3_recolor', args.resume_from_checkpoint)
        args.resume_from_checkpoint = str(source)
    if source:
        from flux3_recolor.initialization import inherit_model_args
        source = inherit_model_args(args, source, resume=bool(args.resume_from_checkpoint))
        if args.init_from:
            args.init_from = source
    for name, default in [('rank', 16), ('lora_alpha', 16.), ('lora_dropout', 0.)]:
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not args.pretrained_model_name_or_path:
        parser.error('请提供 --pretrained_model_name_or_path，或使用 --init_from/--resume_from_checkpoint')
    args.training_prompts = (list(PROMPT_VARIANTS) if args.prompt_augmentation and not custom_prompt
                             and not args.caption_json else [args.instance_prompt])
    for name in ('resolution', 'repeats', 'train_batch_size', 'num_train_epochs', 'gradient_accumulation_steps',
                 'rank', 'max_sequence_length', 'validation_epochs', 'validation_inference_steps', 'logging_steps'):
        if getattr(args, name) < 1:
            parser.error(f'{name} 必须为正整数')
    for name in ('max_train_steps', 'checkpoints_total_limit'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'{name} 必须为正整数')
    for name in ('seed', 'checkpointing_steps', 'validation_steps', 'num_validation_images',
                 'lr_warmup_steps', 'dataloader_num_workers'):
        if getattr(args, name) < 0:
            parser.error(f'{name} 不能为负')
    if args.mix_schedule_epochs is not None and args.mix_schedule_epochs < 1:
        parser.error('mix_schedule_epochs 必须为正整数')
    for name in ('learning_rate', 'lora_alpha', 'adam_epsilon', 'max_grad_norm', 'lr_power', 'logit_std', 'lr_num_cycles'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} 必须为有限正数')
    for name in ('lora_dropout', 'caption_dropout', 'adam_beta1', 'adam_beta2'):
        if not 0 <= getattr(args, name) < 1:
            parser.error(f'{name} 必须在 [0,1)')
    for name in ('logit_mean', 'mode_scale', 'adam_weight_decay'):
        if not math.isfinite(getattr(args, name)):
            parser.error(f'{name} 必须为有限数')
    if args.adam_weight_decay < 0 or not 16 <= args.resolution <= 1024:
        parser.error('weight decay 不能为负；resolution 必须介于16和1024')
    if not 16 <= args.color_swatch_size <= 1024 or args.color_swatch_size % 16:
        parser.error('color_swatch_size 必须为16..1024之间的16倍数')
    if args.aspect_ratio_buckets and not args.use_aspect_ratio_buckets:
        parser.error('--aspect_ratio_buckets 需要 --use_aspect_ratio_buckets')
    if args.use_8bit_adam and args.optimizer != 'adamw':
        parser.error('use_8bit_adam 只适用于 AdamW')
    if args.do_fp8_training and args.bnb_quantization_config_path:
        parser.error('FP8 与 bitsandbytes 量化不能同时启用')
    if args.train_text_encoder and (args.caption_json or args.prompt_augmentation or args.caption_dropout
                                     or args.validation_prompt or args.final_validation_prompt):
        parser.error('可训练text_encoder不支持caption/随机prompt/dropout/验证prompt覆盖')
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    return args


# ------------------------- Dataset and DataLoaders --------------------------
def preprocessing_config(args):
    buckets = None
    if args.aspect_ratio_buckets:
        try:
            if 'x' in args.aspect_ratio_buckets.lower():
                buckets = [tuple(map(int, pair.lower().split('x'))) for pair in args.aspect_ratio_buckets.split(',')]
            else:
                buckets = [tuple(map(int, pair.split(','))) for pair in args.aspect_ratio_buckets.split(';')]
        except ValueError as exc:
            raise ValueError('aspect_ratio_buckets 格式应为 H,W;H,W 或 HxW,HxW') from exc
        if not buckets or any(len(b) != 2 or min(b) < 16 or any(s % 16 for s in b)
                              or b[0] * b[1] > 1024 * 1024 for b in buckets):
            raise ValueError('bucket 宽高必须为正16倍数，面积不超过1MP')
    return {'max_area': args.resolution ** 2, 'color_swatch_size': args.color_swatch_size, 'buckets': buckets,
            'use_aspect_ratio_buckets': args.use_aspect_ratio_buckets}


class DreamBoothDataset(Dataset):
    """Lazy image loading over the already split, unified HF path records."""
    def __init__(self, args, records):
        self.args, self.epoch = args, 0
        self.preprocessing = preprocessing_config(args)
        self.captions = read_json(args.caption_json) if args.caption_json else {}
        self.names = [Path(row['target_path']).name for row in records]
        if args.caption_json and (not isinstance(self.captions, dict)
                or any(not isinstance(value, str) for value in self.captions.values())
                or not set(self.names).issubset(self.captions)):
            raise ValueError('caption_json 必须覆盖所有训练文件名且值为字符串')
        self.buckets, base_buckets, base_directions, valid = [], [], [], []
        bucket_to_idx = {}
        from flux3_recolor.data import SampleDataError
        for index, row in enumerate(records):
            try:
                bucket = self._bucket_for_image(row['height'], row['width'])
            except SampleDataError as exc:
                logger.warning('跳过无效尺寸 sample=%s: %s', row['sample_id'], exc)
                continue
            if bucket not in bucket_to_idx:
                bucket_to_idx[bucket] = len(self.buckets)
                self.buckets.append(bucket)
            base_buckets.append(bucket_to_idx[bucket])
            try:
                subset_id = int(str(row['sample_id']).split(':', 1)[0])
                if not 0 <= subset_id < len(args.training_directions):
                    raise ValueError('子集编号越界')
                direction = args.training_directions[subset_id]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"无法从sample_id解析训练方向: {row['sample_id']}") from exc
            base_directions.append(direction)
            valid.append(index)
        if not valid:
            raise ValueError('Dataset没有可用样本')
        self.records = records.select(valid)
        self.num_instance_images = len(self.records)
        self.bucket_idx = base_buckets * args.repeats
        self.direction_ids = base_directions * args.repeats
        self.sample_ids = [f"{row['sample_id']}::repeat-{repeat}" for repeat in range(args.repeats)
                           for row in self.records]
        self.bucket_indices = [[] for _ in self.buckets]
        for index, bucket in enumerate(self.bucket_idx):
            self.bucket_indices[bucket].append(index)

    def _bucket_for_image(self, height, width):
        from flux3_recolor.data import GRID, SampleDataError, bucket_geometry, grid_geometry
        if min(height, width) < GRID:
            raise SampleDataError(f'宽高至少为{GRID}')
        try:
            if self.preprocessing.get('use_aspect_ratio_buckets'):
                _, (width, height) = bucket_geometry((width, height), self.preprocessing)
            else:
                _, box = grid_geometry((width, height), self.preprocessing['max_area'])
                width, height = box[2] - box[0], box[3] - box[1]
            return height, width
        except ValueError as exc:
            raise SampleDataError(str(exc)) from exc

    def __len__(self):
        return len(self.sample_ids)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def prepare(self, index):
        from flux3_recolor.data import read_path_sample
        row = self.records[index % self.num_instance_images]
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f'{self.args.seed}:{self.epoch}:{index}:{self.sample_ids[index]}'.encode()).digest()[:8], 'big'))
        return read_path_sample(row, self.preprocessing, self.args.center_crop,
                                self.args.random_flip, rng, loader=load_rgb)

    def _read_index(self, index):
        from flux3_recolor.data import SampleDataError
        row = self.records[index % self.num_instance_images]
        target, refs = self.prepare(index)
        height, width = self.buckets[self.bucket_idx[index]]
        if target.size != (width, height) or any(image.size != target.size for image in refs[:2]):
            raise SampleDataError(f"样本 {row['sample_id']} 的实际尺寸与bucket不一致")
        prompt = row['prompt'] or self.args.instance_prompt
        if not self.args.caption_json and len(self.args.training_prompts) > 1:
            offset = int.from_bytes(hashlib.sha256(f'{self.args.seed}:{self.sample_ids[index]}:prompt'.encode()).digest()[:8], 'big')
            prompt = self.args.training_prompts[(offset + self.epoch) % len(self.args.training_prompts)]
        return {'instance_images': image_tensor(target), 'cond_images': [image_tensor(image) for image in refs],
                'instance_prompt': prompt, 'sample_id': self.sample_ids[index],
                'name': Path(row['target_path']).name, 'bucket_idx': self.bucket_idx[index]}

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        from flux3_recolor.data import SampleDataError
        try:
            return self._read_index(index)
        except SampleDataError as exc:
            logger.warning('读取失败 idx=%s sample=%s: %s；尝试其他样本', index, self.sample_ids[index], exc)
        repeat = index // self.num_instance_images
        candidates = list({self.sample_ids[i]: i for i in self.bucket_indices[self.bucket_idx[index]]
                           if i // self.num_instance_images == repeat
                           and self.direction_ids[i] == self.direction_ids[index]
                           and self.sample_ids[i] != self.sample_ids[index]}.values())
        random.Random(self.args.seed + self.epoch + index).shuffle(candidates)
        for actual in candidates[:8]:
            try:
                return self._read_index(actual)
            except SampleDataError as exc:
                logger.warning('读取失败 idx=%s sample=%s: %s', actual, self.sample_ids[actual], exc)
        raise SampleDataError(f'读取失败：样本 {self.sample_ids[index]} 及候选样本均不可用')


def collate_fn(examples):
    import torch
    return {'pixel_values': torch.stack([e['instance_images'] for e in examples]),
            'cond_pixel_values': [torch.stack([e['cond_images'][i] for e in examples]) for i in range(3)],
            'prompts': [e['instance_prompt'] for e in examples],
            'sample_ids': [e['sample_id'] for e in examples], 'names': [e['name'] for e in examples]}


def get_qwen3vl_prompt_embeds(*args, **kwargs):
    from flux3_recolor.modeling import get_qwen3vl_prompt_embeds as encode
    return encode(*args, **kwargs)


# ---------------------------- Checkpoint contracts -------------------------
def validate_training_resume(saved, current, state):
    if state.get('trainer') != TRAINER or state.get('version') != STATE_VERSION:
        raise ValueError('旧训练状态格式不兼容；请使用新输出目录开始训练')
    saved = dict(saved)
    if 'training_directions' in current and 'training_directions' not in saved:
        saved['training_directions'] = ['b_to_a']
        saved['subdatasets'] = subdataset_config(['b_to_a'])
    if len(saved.get('training_directions', [])) == 2:
        # Before the epoch schedule, the builder balanced both directions 1:1.
        saved.setdefault('atob_ratio_start', 0.5)
        saved.setdefault('atob_ratio_end', 0.5)
        saved.setdefault('mix_schedule_epochs', saved.get('num_train_epochs', 1))
    # Older checkpoints used a single prompt and have neither augmentation field.
    if 'prompt_augmentation' not in saved and 'training_prompts' not in saved:
        saved['prompt_augmentation'] = False
        saved['training_prompts'] = [saved.get('instance_prompt', saved.get('prompt', DEFAULT_PROMPT))]
    saved.pop('cache_latents', None)
    if 'train_text_encoder' in current:
        saved.setdefault('train_text_encoder', False)
    mutable = {'checkpointing_steps', 'checkpoints_total_limit', 'validation_steps', 'validation_epochs',
               'num_validation_images', 'validation_inference_steps', 'skip_final_inference', 'report_to',
               'logging_dir', 'logging_steps', 'dataloader_num_workers', 'gradient_checkpointing'}
    if saved.get('lr_scheduler') == current.get('lr_scheduler') == 'constant':
        mutable |= {'max_train_steps', 'num_train_epochs'}
    for key in sorted((set(saved) | set(current)) - mutable):
        if saved.get(key) != current.get(key):
            if key in {'distributed_type', 'world_size'}:
                raise ValueError(
                    f'恢复配置不一致: {key} (checkpoint={saved.get(key)!r}, 当前={current.get(key)!r})；'
                    '请沿用 checkpoint 的分布式模式和进程数'
                )
            raise ValueError(f'恢复配置不一致: {key}')


def owned_checkpoints(output):
    result = []
    for path in Path(output).glob('checkpoint-*'):
        if path.is_symlink() or not path.is_dir() or not re.fullmatch(r'checkpoint-\d+', path.name):
            continue
        try:
            state = read_json(path / 'training_state.json')
        except (OSError, ValueError):
            continue
        if (isinstance(state, dict) and state.get('trainer') == TRAINER and state.get('version') == STATE_VERSION
                and state.get('global_step') == int(path.name.split('-')[-1])):
            result.append(path)
    return sorted(result, key=lambda p: int(p.name.split('-')[-1]))


def checkpoints_to_remove(output, limit, incoming=True):
    paths = owned_checkpoints(output)
    return paths[:max(0, len(paths) - limit + int(incoming))] if limit else []


def resolve_training_checkpoint(output, requested):
    """Explicit resume never silently starts fresh, and missing paths are not old formats."""
    output = Path(output).expanduser().resolve()
    if not output.is_dir():
        raise ValueError(f'恢复输出目录不存在或不是目录: {output}；从头训练请去掉 --resume_from_checkpoint')
    if requested == 'latest':
        paths = owned_checkpoints(output)
        if paths:
            return paths[-1]
        candidates = [p for p in output.glob('checkpoint-*') if p.is_dir()]
        if not candidates:
            raise ValueError(f'输出目录中没有checkpoint-*目录，无法续训: {output}；从头训练请去掉 --resume_from_checkpoint')
        raise ValueError(f'{output} 中没有有效的版本2完成状态；请检查checkpoint的training_state.json。'
                         '缺失、损坏、旧格式或符号链接checkpoint不会自动恢复。')
    checkpoint = Path(requested).expanduser().resolve()
    if checkpoint.parent != output or Path(requested).expanduser().is_symlink():
        raise ValueError('checkpoint 必须为当前输出目录的真实子目录')
    if not checkpoint.is_dir() or not (checkpoint / 'training_state.json').is_file():
        raise ValueError(f'恢复checkpoint不存在或缺少training_state.json: {checkpoint}')
    return checkpoint


# ----------------------- Model/optimizer initialization --------------------
def cuda_device_index(device):
    """Resolve logical CUDA indices, including Accelerate's single-device `cuda`."""
    import torch
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('需要CUDA设备')
    return device.index if device.index is not None else torch.cuda.current_device()


@contextmanager
def offload_models(model, device, offload=False, sharded=False):
    if not sharded:
        model.to(device)
    try:
        yield model
    finally:
        if offload and not sharded:
            model.to('cpu')


def validate_optional_features(args, accelerator):
    import importlib.util
    fsdp = getattr(accelerator.state, 'fsdp_plugin', None)
    if accelerator.distributed_type.name not in {'NO', 'MULTI_CPU', 'MULTI_GPU', 'FSDP'}:
        raise ValueError('当前支持单进程、CPU/GPU DDP 和 FSDP1，不支持其他分布式后端')
    if fsdp:
        if getattr(fsdp, 'fsdp_version', 1) != 1 or not fsdp.use_orig_params:
            raise ValueError('FSDP 需要 fsdp_version=1、use_orig_params=true（冻结基础参数与LoRA混合）')
        if args.bnb_quantization_config_path or args.do_fp8_training or args.use_8bit_adam or args.optimizer == 'prodigy':
            raise ValueError('FSDP 路径目前只支持未量化 Transformer + AdamW，不支持FP8/8bit/Prodigy组合')
    if args.fsdp_text_encoder and not fsdp:
        raise ValueError('--fsdp_text_encoder 需要 Accelerate FSDP1 配置')
    if args.bnb_quantization_config_path and accelerator.num_processes > 1:
        raise ValueError('bitsandbytes量化加载仅支持单GPU；普通LoRA可使用DDP')
    optional = []
    if args.bnb_quantization_config_path or args.use_8bit_adam:
        optional.append(('bitsandbytes', 'bitsandbytes'))
    if args.optimizer == 'prodigy':
        optional.append(('prodigyopt', 'Prodigy'))
    if args.do_fp8_training:
        optional.append(('torchao', 'FP8'))
    for module, feature in optional:
        if importlib.util.find_spec(module) is None:
            raise ValueError(f'{feature} 需要安装可选依赖 {module}；未自动安装')
    if (args.bnb_quantization_config_path or args.do_fp8_training or args.use_8bit_adam or fsdp) and accelerator.device.type != 'cuda':
        raise ValueError('所选量化/FP8/8bit/FSDP高级路径需要CUDA设备')
    return bool(fsdp)


def load_training_components(args, accelerator, dtype, is_fsdp):
    import torch
    from flux3_recolor.config import tokenizer_subfolder
    from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel, FlowMatchEulerDiscreteScheduler, BitsAndBytesConfig
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from flux3_recolor.modeling import FLOW_SHIFT, validate_vae_grid, add_lora
    kwargs = {'local_files_only': True, 'revision': args.revision}
    model_kwargs = {**kwargs, 'variant': args.variant, 'torch_dtype': dtype}
    from flux3_recolor.precision import set_transformer_autocast
    vae_kwargs = {**model_kwargs, 'torch_dtype': torch.float32}
    accelerator.print(f'[基础权重] 加载 VAE: {args.pretrained_model_name_or_path}/vae; dtype=torch.float32')
    vae = AutoencoderKLFlux2.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae', **vae_kwargs)
    validate_vae_grid(vae)
    quant = None
    if args.bnb_quantization_config_path:
        config = read_json(args.bnb_quantization_config_path)
        if config.get('load_in_4bit'):
            config['bnb_4bit_compute_dtype'] = dtype
        if not config.get('load_in_4bit') and not config.get('load_in_8bit'):
            raise ValueError('量化JSON必须启用load_in_4bit或load_in_8bit')
        quant = BitsAndBytesConfig(**config)
    extra = {'quantization_config': quant}
    if quant:
        extra['device_map'] = {'': cuda_device_index(accelerator.device)}
    # FSDP1 flattens parameters before its mixed-precision casts. Frozen base and
    # FP32 LoRA must have uniform master dtype at that point (use_orig_params is insufficient).
    transformer_kwargs = {**model_kwargs, 'torch_dtype': torch.float32} if is_fsdp else model_kwargs
    accelerator.print(f'[基础权重] 加载 Transformer: {args.pretrained_model_name_or_path}/transformer; '
                      f'dtype={transformer_kwargs["torch_dtype"]}')
    transformer = Flux3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='transformer', **transformer_kwargs, **extra)
    set_transformer_autocast(transformer, dtype)
    transformer.requires_grad_(False)
    if quant:
        from peft import prepare_model_for_kbit_training
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        convert_to_float8_training(transformer, config=Float8LinearConfig(pad_inner_dim=True),
            module_filter_fn=lambda module, name: not name.endswith('proj_out') and
                getattr(module, 'in_features', 1) % 16 == 0 and getattr(module, 'out_features', 1) % 16 == 0)
    layers = [layer.strip() for layer in args.lora_layers.split(',')] if args.lora_layers else None
    if layers is not None and any(not layer for layer in layers):
        raise ValueError('lora_layers 含空模块名')
    from peft import LoraConfig
    adapter_config = getattr(args, '_adapter_config', None)
    add_lora(transformer, args.rank, args.lora_alpha, args.lora_dropout, target_modules=layers,
             adapter_config=LoraConfig(**adapter_config) if adapter_config is not None else None)
    if args.init_from:
        from flux3_recolor.modeling import load_adapter_state
        load_adapter_state(transformer, Path(args.init_from) / 'transformer')
        accelerator.print(f'[已训练权重] 已加载 Transformer LoRA: {Path(args.init_from) / "transformer"}')
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if not is_fsdp and not quant:
        transformer.to(accelerator.device)
    accelerator.print(f'[基础权重] 加载 Qwen text_encoder: {args.pretrained_model_name_or_path}/text_encoder; '
                      f'dtype={dtype}（文本编码及 CFG 空文本）')
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='text_encoder', **model_kwargs)
    # Load the component directory directly: tokenizer backend config detection
    # must see Qwen's config, rather than the unrelated native BFL root config.
    processor_dir = Path(args.pretrained_model_name_or_path) / tokenizer_subfolder(args.pretrained_model_name_or_path)
    tokenizer = AutoProcessor.from_pretrained(str(processor_dir), **kwargs)
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    if args.fsdp_text_encoder:
        from diffusers.training_utils import wrap_with_fsdp, get_fsdp_kwargs_from_accelerator
        text_encoder = wrap_with_fsdp(text_encoder, device=accelerator.device, offload=args.offload,
                                     fsdp_kwargs=get_fsdp_kwargs_from_accelerator(accelerator))
    return {'transformer': transformer, 'vae': vae, 'text_encoder': text_encoder, 'tokenizer': tokenizer,
            'scheduler': FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT, use_dynamic_shifting=False)}


def create_optimizer(args, transformer, num_processes, text_encoder=None):
    import torch
    lr = args.learning_rate
    if args.scale_lr:
        lr *= args.gradient_accumulation_steps * args.train_batch_size * num_processes
    parameters = [{'params': [p for p in transformer.parameters() if p.requires_grad], 'lr': lr}]
    if text_encoder is not None:
        parameters.append({'params': text_encoder.parameters(), 'lr': lr, 'weight_decay': 0.})
    kwargs = {'betas': (args.adam_beta1, args.adam_beta2), 'weight_decay': args.adam_weight_decay, 'eps': args.adam_epsilon}
    if args.optimizer == 'prodigy':
        from prodigyopt import Prodigy
        return Prodigy(parameters, beta3=args.prodigy_beta3, decouple=args.prodigy_decouple,
                       use_bias_correction=args.prodigy_use_bias_correction,
                       safeguard_warmup=args.prodigy_safeguard_warmup, **kwargs)
    optimizer = torch.optim.AdamW
    if args.use_8bit_adam:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit
    return optimizer(parameters, lr=lr, **kwargs)


def require_trackers(report_to):
    if report_to == 'none':
        return None
    from accelerate.tracking import get_available_trackers
    available = {tracker.value for tracker in get_available_trackers()}
    if (report_to == 'all' and not available) or (report_to != 'all' and report_to not in available):
        raise ValueError(f'所选tracker {report_to} 不可用，请安装对应依赖或使用 --report_to none')
    return report_to


def main_process_action(accelerator, action):
    """Propagate main-process I/O failures to every rank before the next collective."""
    from accelerate.utils import broadcast_object_list
    payload = [None, None]
    if accelerator.is_main_process:
        try:
            payload[0] = action()
        except Exception as exc:
            payload[1] = f'{type(exc).__name__}: {exc}'
    broadcast_object_list(payload)
    if payload[1]:
        raise ValueError(payload[1])
    return payload[0]


def lora_state_dict(accelerator, transformer, is_fsdp):
    from peft.utils import get_peft_model_state_dict
    full_state = accelerator.get_state_dict(transformer) if is_fsdp else None
    if not accelerator.is_main_process:
        return None
    model = accelerator.unwrap_model(transformer)
    state = get_peft_model_state_dict(model, state_dict=full_state) if is_fsdp else get_peft_model_state_dict(model)
    return {k: value.detach().cpu().contiguous() for k, value in state.items()}


def export_lora(accelerator, transformer, output, args, dtype, is_fsdp, text_encoder=None):
    from flux3_recolor.component_io import save_components
    import torch
    state = lora_state_dict(accelerator, transformer, is_fsdp)
    def save():
        save_components(accelerator.unwrap_model(transformer), output, args.pretrained_model_name_or_path,
            accelerator.unwrap_model(text_encoder) if text_encoder is not None else None,
            state=state, dtype=torch.float32 if args.upcast_before_saving else dtype)
    main_process_action(accelerator, save)


def save_checkpoint(accelerator, transformer, args, state, config, split, dtype, is_fsdp):
    output = Path(args.output_dir)
    destination = output / f"checkpoint-{state['global_step']}"
    def reserve():
        if destination.exists():
            raise ValueError(f'checkpoint 已存在: {destination}')
        return tempfile.mkdtemp(prefix=f".checkpoint-{state['global_step']}-", dir=output)
    pending = Path(main_process_action(accelerator, reserve))
    # DDP hooks save adapters only; FSDP uses Accelerate's native model+optimizer shard format.
    accelerator.save_state(str(pending))
    if is_fsdp:
        export_lora(accelerator, transformer, pending, args, dtype, is_fsdp)
    accelerator.wait_for_everyone()
    def finish():
        write_json(pending / 'recolor_config.json', config)
        write_json(pending / 'split.json', split)
        write_json(pending / 'training_state.json', state)
        pending.rename(destination)
        for path in checkpoints_to_remove(output, args.checkpoints_total_limit, incoming=False):
            shutil.rmtree(path)
    main_process_action(accelerator, finish)


def make_validation_pipeline(args, components, transformer, accelerator, dtype, is_fsdp, state=None):
    from flux3_recolor.inference import build_pipeline
    from flux3_recolor.precision import set_transformer_autocast
    if not is_fsdp and '_validation_pipeline' in components:
        pipeline = components['_validation_pipeline']
        if pipeline.transformer is not accelerator.unwrap_model(transformer):
            raise RuntimeError('验证 pipeline 与训练模型未共享对象')
        pipeline.to(accelerator.device)
        return pipeline
    if is_fsdp:
        from diffusers import Flux3Transformer2DModel
        from peft import LoraConfig, set_peft_model_state_dict
        model = Flux3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,
            subfolder='transformer', torch_dtype=dtype, local_files_only=True, variant=args.variant)
        from flux3_recolor.modeling import add_lora
        layers = [x.strip() for x in args.lora_layers.split(',')] if args.lora_layers else None
        adapter_config = getattr(args, '_adapter_config', None)
        add_lora(model, args.rank, args.lora_alpha, args.lora_dropout, target_modules=layers,
                 adapter_config=LoraConfig(**adapter_config) if adapter_config is not None else None)
        set_peft_model_state_dict(model, state, adapter_name='default')
        set_transformer_autocast(model, dtype)
        model.requires_grad_(False).eval()
    else:
        model = accelerator.unwrap_model(transformer)
    pipeline_components = {**components, 'transformer': model}
    if components.get('text_encoder') is not None:
        pipeline_components['text_encoder'] = accelerator.unwrap_model(components['text_encoder'])
    pipeline = build_pipeline(pipeline_components, accelerator.device)
    if not is_fsdp:
        components['_validation_pipeline'] = pipeline
    return pipeline


def log_validation(args, accelerator, components, transformer, dtype, is_fsdp,
                   config, validation_records, embeddings, label, global_step):
    from flux3_recolor.inference import select_validation_records
    from flux3_recolor.validation import ValidationRuns
    records = select_validation_records(validation_records, args.num_validation_images)
    runs = components.setdefault('_validation_runs', ValidationRuns())
    runs.run(global_step, config, [row['sample_id'] for row in records],
             args.seed, args.validation_inference_steps,
             lambda: _log_validation(args, accelerator, components, transformer, dtype, is_fsdp,
                                     config, records, embeddings, label, global_step))


def _log_validation(args, accelerator, components, transformer, dtype, is_fsdp,
                    config, validation_records, embeddings, label, global_step):
    import torch
    from flux3_recolor.precision import pipeline_forward
    from flux3_recolor.inference import generate_validation
    from flux3_recolor.validation import validation_output, pipeline_text_embeddings, log_validation_images
    adapter_state = lora_state_dict(accelerator, transformer, is_fsdp) if is_fsdp else None
    def generate():
        devices = [cuda_device_index(accelerator.device)] if accelerator.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            model = accelerator.unwrap_model(transformer)
            was_training = model.training
            model.eval()
            pipeline = None
            try:
                pipeline = make_validation_pipeline(args, components, transformer, accelerator, dtype, is_fsdp, adapter_state)
                active_embeddings = pipeline_text_embeddings(pipeline, embeddings)
                destination = Path(args.output_dir) / f'validation-{label}'
                with validation_output(destination, protected=[args.instance_data_dir]) as staging:
                    with pipeline_forward(pipeline.transformer):
                        generate_validation(pipeline, args.instance_data_dir, validation_records,
                            config, staging, args.seed, args.validation_inference_steps, active_embeddings)
                log_validation_images(accelerator.trackers, destination, label, global_step)
            finally:
                del pipeline
                model.train(was_training)
                if args.offload:
                    components['vae'].to('cpu')
                if accelerator.device.type == 'cuda':
                    torch.cuda.empty_cache()
    main_process_action(accelerator, generate)
    accelerator.wait_for_everyone()


def prepare_path_dataset(args, sample_names=None):
    """Reuse metadata across training runs; output/model/LR do not affect it."""
    from flux3_recolor.dataset_cache import prepare_cached_dataset
    config = {'root': str(Path(args.instance_data_dir).expanduser().resolve()),
        'directions': args.training_directions, 'validation_fraction': args.validation_fraction,
        'seed': args.seed, 'prompt': args.instance_prompt,
        'captions': read_json(args.caption_json) if args.caption_json else None,
        'source_groups': read_json(args.groups_json) if args.groups_json else None,
        'sample_names': sample_names}
    return prepare_cached_dataset(config, num_proc=max(1, args.dataloader_num_workers))


# --------------------------------- Training --------------------------------
def main(input_args=None):
    args = input_args if isinstance(input_args, argparse.Namespace) else parse_args(input_args)
    policy = preprocessing_config(args)
    args.output_dir = str(Path(args.output_dir or 'outputs/flux3_recolor').expanduser().resolve())
    checkpoint = (resolve_training_checkpoint(args.output_dir, args.resume_from_checkpoint)
                  if args.resume_from_checkpoint else None)
    if checkpoint is not None and args.mix_schedule_epochs is None:
        saved_config = read_json(checkpoint / 'recolor_config.json')
        args.mix_schedule_epochs = saved_config.get('mix_schedule_epochs', saved_config.get('num_train_epochs', 1))

    import torch
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
    from diffusers.optimization import get_scheduler
    from flux3_recolor.modeling import encode_recolor_latents, flow_matching_loss, text_position_ids, load_adapter_state

    args.pretrained_model_name_or_path, model_signature = validate_local_model(args.pretrained_model_name_or_path)
    expected_signature = getattr(args, '_source_signature', None)
    if expected_signature is not None and expected_signature != model_signature:
        raise ValueError('基础模型配置与初始化权重不一致')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with=require_trackers(args.report_to),
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(Path(args.output_dir) / args.logging_dir)),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=DataLoaderConfiguration(even_batches=False, split_batches=False, dispatch_batches=False),
        step_scheduler_with_optimizer=False,
    )
    args.mixed_precision = accelerator.mixed_precision
    if args.mixed_precision != 'no' and accelerator.device.type != 'cuda':
        raise ValueError('CPU训练请显式使用 --mixed_precision no')
    is_fsdp = validate_optional_features(args, accelerator)
    if args.train_text_encoder and is_fsdp:
        raise ValueError('可训练text_encoder暂支持单卡/DDP，不支持FSDP')
    set_seed(args.seed)
    if args.allow_tf32 and accelerator.device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = {'no': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.mixed_precision]
    main_process_action(accelerator, lambda: str(prepare_output(args.output_dir,
        protected=[args.instance_data_dir, args.pretrained_model_name_or_path] +
                  ([args.init_from] if args.init_from else []), resume=checkpoint is not None)))
    path_dataset = main_process_action(accelerator, lambda: prepare_path_dataset(args))
    accelerator.print(f'路径Dataset: {path_dataset}')
    from datasets import load_from_disk
    prepared = load_from_disk(path_dataset)
    split = {name: list(rows['sample_id']) for name, rows in prepared.items()}
    report = {'path_dataset': path_dataset, 'dataset_format': 'recolor_paths_v1',
              'train_records': len(prepared['train']), 'validation_records': len(prepared['validation'])}
    if not args.groups_json:
        accelerator.print('拆分按完整文件名主干；不同文件名的同源衍生图请提供 --groups_json。')
    dataset = DreamBoothDataset(args, prepared['train'])
    if len(args.training_directions) == 2:
        batch_sampler = ScheduledMixBucketBatchSampler(
            dataset, args.train_batch_size, args.seed, accelerator.num_processes,
            args.atob_ratio_start, args.atob_ratio_end,
            total_epochs=args.mix_schedule_epochs or args.num_train_epochs)
    else:
        batch_sampler = BucketBatchSampler(dataset, args.train_batch_size, args.seed, accelerator.num_processes)
    generator = torch.Generator().manual_seed(args.seed)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers, generator=generator, pin_memory=accelerator.device.type == 'cuda')
    local_batches = len(batch_sampler) // accelerator.num_processes
    updates_per_epoch = math.ceil(local_batches / args.gradient_accumulation_steps)
    args.max_train_steps = args.max_train_steps or args.num_train_epochs * updates_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / updates_per_epoch)
    if isinstance(batch_sampler, ScheduledMixBucketBatchSampler):
        if args.mix_schedule_epochs is None:
            args.mix_schedule_epochs = args.num_train_epochs
        batch_sampler.set_total_epochs(args.mix_schedule_epochs)

    config = {key: value for key, value in vars(args).items() if not key.startswith('_') and key not in {
        'instance_data_dir', 'output_dir', 'pretrained_model_name_or_path', 'groups_json',
        'caption_json', 'resume_from_checkpoint', 'local_rank', 'bnb_quantization_config_path'}}
    if len(args.training_directions) < 2:
        config.pop('atob_ratio_start', None)
        config.pop('atob_ratio_end', None)
        config.pop('mix_schedule_epochs', None)
    config.update(format_version=1, trainer=TRAINER, training_state_version=STATE_VERSION,
        model_path=args.pretrained_model_name_or_path, model_config_signature=model_signature,
        split=split, world_size=accelerator.num_processes,
        dataset_format='recolor_paths_v1', dataset_path=path_dataset,
        weights_layout='components_v1', condition_order=['gray', 'source', 'swatch'],
        dataset_root=str(Path(args.instance_data_dir).resolve()), grid=16,
        max_area=args.resolution ** 2, preprocessing=policy, prompt=args.instance_prompt,
        captions_digest=hashlib.sha256(json.dumps(dataset.captions, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        quantization_config=read_json(args.bnb_quantization_config_path) if args.bnb_quantization_config_path else None,
        distributed_type=accelerator.distributed_type.name)
    if args.train_text_encoder:
        config['text_encoder_format'] = 'transformers_v1'
    config = json.loads(json.dumps(config))  # Normalize tuples to the persisted JSON form.
    state = {'trainer': TRAINER, 'version': STATE_VERSION, 'global_step': 0, 'epoch': 0, 'batch_offset': 0}
    if checkpoint is not None:
        state = read_json(checkpoint / 'training_state.json')
        saved_config = read_json(checkpoint / 'recolor_config.json')
        config['init_from'] = saved_config.get('init_from')
        validate_training_resume(saved_config, config, state)
        if read_json(checkpoint / 'split.json') != split:
            raise ValueError('恢复时split发生变化')
        if state['global_step'] >= args.max_train_steps:
            raise ValueError('max_train_steps 必须大于checkpoint已完成步数')
        if any(int(p.name.split('-')[-1]) > state['global_step'] for p in owned_checkpoints(args.output_dir)):
            raise ValueError('已有更晚的完成checkpoint，请从latest恢复，避免覆盖后续产物')
    components = load_training_components(args, accelerator, dtype, is_fsdp)
    transformer, vae = components['transformer'], components['vae']

    def compute_text_embeddings(prompts, pad_to_length=None):
        encoder = components['text_encoder']
        with offload_models(encoder, accelerator.device, args.offload, sharded=args.fsdp_text_encoder):
            embeds = get_qwen3vl_prompt_embeds(encoder, components['tokenizer'], prompts, accelerator.device, dtype,
                args.text_encoder_out_layers, args.max_sequence_length, pad_to_length=pad_to_length)
        if embeds.shape[-1] != accelerator.unwrap_model(transformer).config.joint_attention_dim:
            raise ValueError('所选文本隐藏层维度与Transformer joint_attention_dim不一致')
        return embeds

    # Fixed prompt is encoded once. Custom captions remain online so each shuffled
    # batch gets exactly its largest window, rather than a stale batch-index cache.
    text_encoder = None
    source = Path(args.init_from) if args.init_from else checkpoint
    if source and args.train_text_encoder and (source / 'text_encoder').is_dir():
        from flux3_recolor.text_encoder import TextEncoderModel, current_text_embeddings
        text_encoder = TextEncoderModel.from_pretrained(source / 'text_encoder',
            local_files_only=True, torch_dtype=torch.float32).to(accelerator.device).train()
        if text_encoder.config.hidden_size != transformer.config.joint_attention_dim:
            raise ValueError('已有text_encoder hidden_size与Transformer不一致')
        fixed_embeddings = current_text_embeddings(text_encoder, dtype)
        accelerator.print(f'[已训练权重] 已加载 text_encoder: {source / "text_encoder"}; shape={tuple(text_encoder.weight.shape)}')
    else:
        fixed_embeddings = compute_text_embeddings(args.instance_prompt) if not args.caption_json else None
        if args.train_text_encoder:
            from flux3_recolor.text_encoder import TextEncoderModel
            text_encoder = TextEncoderModel.from_features(fixed_embeddings).to(accelerator.device)
            accelerator.print('[新初始化] text_encoder 从当前 prompt 初始化')
    if args.init_from:
        accelerator.print('[新训练] init_from 仅继承权重：新建 optimizer/scheduler，global_step 从 0 开始')
    optimizer = create_optimizer(args, transformer, accelerator.num_processes, text_encoder)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps, num_cycles=args.lr_num_cycles, power=args.lr_power)
    prompt_embeddings = {}
    if len(args.training_prompts) > 1:
        # Encode together for a common token window; preserve per-prompt attention masks.
        encoded_prompts = compute_text_embeddings(args.training_prompts)
        prompt_embeddings = {prompt: encoded_prompts[i:i + 1] for i, prompt in enumerate(args.training_prompts)}
        fixed_embeddings = None
    fixed_empty = compute_text_embeddings('', fixed_embeddings.shape[1]) if fixed_embeddings is not None and args.caption_dropout else None
    if prompt_embeddings and args.caption_dropout:
        fixed_empty = compute_text_embeddings('', encoded_prompts.shape[1])
    validation_embeddings = {}
    if args.num_validation_images:
        negative = compute_text_embeddings('')
        for prompt in {args.validation_prompt or args.instance_prompt,
                       args.validation_prompt or args.final_validation_prompt or args.instance_prompt}:
            positive = (fixed_embeddings if prompt == args.instance_prompt and fixed_embeddings is not None
                        else compute_text_embeddings(prompt))
            validation_embeddings[prompt] = (positive, negative)
    if not args.caption_json:
        encoder = components.pop('text_encoder')
        if not args.fsdp_text_encoder:
            encoder.to('cpu')
        del encoder
        components.pop('tokenizer')
    if accelerator.device.type == 'cuda':
        torch.cuda.empty_cache()

    # Hooks mirror the reference example. FSDP uses native Accelerator checkpointing
    # (full/sharded base + optimizer) and exports a standard adapter alongside it.
    handles = []
    if not is_fsdp:
        def save_model_hook(models, weights, directory):
            from flux3_recolor.component_io import save_components
            main_process_action(accelerator, lambda: save_components(
                accelerator.unwrap_model(transformer), directory, args.pretrained_model_name_or_path,
                accelerator.unwrap_model(text_encoder) if text_encoder is not None else None))
            weights.clear()
        def load_model_hook(models, directory):
            load_adapter_state(accelerator.unwrap_model(transformer), Path(directory) / 'transformer')
            accelerator.print(f'[已训练权重] 恢复 Transformer LoRA: {Path(directory) / "transformer"}')
            if args.train_text_encoder:
                from flux3_recolor.text_encoder import load_text_encoder_into
                load_text_encoder_into(accelerator.unwrap_model(text_encoder), directory)
                accelerator.print(f'[已训练权重] 恢复 text_encoder: {Path(directory) / "text_encoder"}')
            models.clear()
        handles = [accelerator.register_save_state_pre_hook(save_model_hook),
                   accelerator.register_load_state_pre_hook(load_model_hook)]

    if text_encoder is not None:
        transformer, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, text_encoder, optimizer, train_dataloader, lr_scheduler)
        components['text_encoder'] = text_encoder
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler)
    training_models = [transformer] + ([text_encoder] if text_encoder is not None else [])
    if len(train_dataloader) != local_batches:
        raise ValueError('分片后的DataLoader长度与分桶计划不符')
    def persist_config():
        write_json(Path(args.output_dir) / 'recolor_config.json', config)
        write_json(Path(args.output_dir) / 'dataset_report.json', report)
        write_json(Path(args.output_dir) / 'split.json', split)
    main_process_action(accelerator, persist_config)
    if accelerator.is_main_process:
        tracker_config = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in config.items() if v is not None}
        accelerator.init_trackers('flux3-recolor-lora', config=tracker_config)
    if checkpoint:
        accelerator.load_state(str(checkpoint))
    accelerator.print(f"Training: {len(dataset)} samples; {local_batches} batches/rank/epoch; "
                      f"{accelerator.num_processes} processes; {args.max_train_steps} optimizer steps")
    from flux3_recolor.training_logging import TrainingLogger
    training_logger = TrainingLogger(args, accelerator, state['global_step'])
    epoch = state['epoch']
    optimizer.zero_grad(set_to_none=True)
    try:
        while state['global_step'] < args.max_train_steps:
            dataset.set_epoch(epoch)
            batch_sampler.set_epoch(epoch)
            train_dataloader.set_epoch(epoch)
            generator.manual_seed(args.seed + epoch)
            plan = batch_sampler.batches()
            if isinstance(batch_sampler, ScheduledMixBucketBatchSampler):
                stats = batch_sampler.last_mix
                accelerator.print(f"epoch={epoch + 1}/{args.num_train_epochs} "
                                  f"AtoB={stats['atob_samples']}/{stats['total_samples']} "
                                  f"({stats['achieved_atob_ratio']:.1%}; target="
                                  f"{stats['requested_atob_ratio']:.1%})")
            start_offset = state['batch_offset'] if state['epoch'] == epoch else 0
            epoch_dataloader = train_dataloader
            if start_offset:
                accelerator.print(f'恢复 epoch {epoch + 1}：每个进程跳过已训练的 {start_offset} 个 batch')
                epoch_dataloader = accelerator.skip_first_batches(train_dataloader, start_offset)
            group_loss = torch.zeros((), device=accelerator.device)
            group_count = 0
            transformer.train()
            for batch_index, batch in enumerate(epoch_dataloader, start=start_offset):
                group_start = batch_index // args.gradient_accumulation_steps * args.gradient_accumulation_steps
                group_end = min(group_start + args.gradient_accumulation_steps, local_batches)
                group_samples = sum(len(indices) for indices in
                    plan[group_start * accelerator.num_processes:group_end * accelerator.num_processes])
                batch_size = len(batch['sample_ids'])
                with accelerator.accumulate(*training_models):
                    with offload_models(vae, accelerator.device, args.offload):
                        target, references = encode_recolor_latents(vae, batch['pixel_values'], batch['cond_pixel_values'])
                    embeds = (text_encoder(batch_size=batch_size).last_hidden_state.to(dtype=dtype) if text_encoder is not None
                              else torch.cat([prompt_embeddings[p] for p in batch['prompts']]) if prompt_embeddings
                              else fixed_embeddings.repeat(batch_size, 1, 1) if fixed_embeddings is not None
                              else compute_text_embeddings(batch['prompts']))
                    if args.caption_dropout:
                        empty = fixed_empty if fixed_empty is not None else compute_text_embeddings('', embeds.shape[1])
                        drop = torch.rand(batch_size, device=accelerator.device) < args.caption_dropout
                        embeds = embeds.clone()
                        embeds[drop] = empty
                    with accelerator.autocast():
                        per_sample_loss = flow_matching_loss(transformer, target, references, embeds,
                            text_position_ids(embeds.shape[1], accelerator.device), components['scheduler'],
                            weighting_scheme=args.weighting_scheme, logit_mean=args.logit_mean,
                            logit_std=args.logit_std, mode_scale=args.mode_scale, return_per_sample=True,
                            compute_dtype=dtype)
                        loss = per_sample_loss.mean()
                    finite = accelerator.reduce(torch.isfinite(loss).to(torch.int), reduction='sum')
                    if finite.item() != accelerator.num_processes:
                        raise FloatingPointError(f'非有限loss；当前rank样本: {batch["names"]}')
                    group_loss += loss.detach().float() * batch_size
                    group_count += batch_size
                    # Accelerator divides by accumulation_steps; DDP averages ranks.
                    # Correct both factors, including undersized final accumulation groups.
                    factor = args.gradient_accumulation_steps * accelerator.num_processes * batch_size / group_samples
                    accelerator.backward(loss * factor)
                    if accelerator.sync_gradients:
                        # Accelerate recognizes the full list to dispatch FSDP's global norm.
                        # Frozen parameters have grad=None and are safely ignored.
                        accelerator.clip_grad_norm_((p for model in training_models for p in model.parameters()), args.max_grad_norm)
                    optimizer.step()
                    successful = accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
                    if successful:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                end_epoch = batch_index + 1 == local_batches
                state.update(epoch=epoch + 1 if end_epoch else epoch, batch_offset=0 if end_epoch else batch_index + 1)
                if not accelerator.sync_gradients:
                    continue
                if not successful:
                    group_loss.zero_()
                    group_count = 0
                    accelerator.print('FP16 optimizer step skipped; 未推进LR或global_step')
                    continue
                state['global_step'] += 1
                global_step = state['global_step']
                training_logger.step(group_loss, group_count, global_step, lr_scheduler.get_last_lr()[0])
                group_loss.zero_()
                group_count = 0
                # Commit recoverable training state before fallible validation/rendering.
                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(accelerator, transformer, args, state, config, split, dtype, is_fsdp)
                if args.num_validation_images and args.validation_steps and global_step % args.validation_steps == 0:
                    prompt = args.validation_prompt or args.instance_prompt
                    log_validation(args, accelerator, components, transformer, dtype, is_fsdp,
                        {**config, 'prompt': prompt}, prepared['validation'], validation_embeddings[prompt], f'step-{global_step}', global_step)
                if state['global_step'] >= args.max_train_steps:
                    break
            if state['epoch'] == epoch + 1 and args.num_validation_images and (epoch + 1) % args.validation_epochs == 0:
                prompt = args.validation_prompt or args.instance_prompt
                log_validation(args, accelerator, components, transformer, dtype, is_fsdp,
                    {**config, 'prompt': prompt}, prepared['validation'], validation_embeddings[prompt],
                    f'epoch-{epoch + 1}-step-{state["global_step"]}', state['global_step'])
            epoch += 1
        export_lora(accelerator, transformer, args.output_dir, args, dtype, is_fsdp, text_encoder)
        main_process_action(accelerator, lambda: write_json(Path(args.output_dir) / 'training_state.json', state))
        if args.num_validation_images and not args.skip_final_inference:
            prompt = args.validation_prompt or args.final_validation_prompt or args.instance_prompt
            log_validation(args, accelerator, components, transformer, dtype, is_fsdp,
                {**config, 'prompt': prompt}, prepared['validation'], validation_embeddings[prompt], f'final-{state["global_step"]}', state['global_step'])
        accelerator.wait_for_everyone()
        accelerator.print(f'LoRA saved: {args.output_dir}')
        # Return the trained model for programmatic use; CLI callers simply ignore it.
        return {'state': state, 'model': None if is_fsdp else accelerator.unwrap_model(transformer),
                'text_encoder': accelerator.unwrap_model(text_encoder) if text_encoder is not None else None}
    finally:
        training_logger.close()
        for handle in handles:
            handle.remove()
        accelerator.end_training()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
