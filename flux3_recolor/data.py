"""Strict filename pairing, deterministic splits, and shared image geometry."""

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from . import CONDITION_FOLDERS, TARGET_FOLDER

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
GRID = 16  # FLUX VAE downsampling (8) followed by latent patchifying (2).
MAX_REFERENCE_AREA = 1024 * 1024


def load_rgb(path):
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert('RGB')
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f'无法解码图片 {path}: {exc}') from exc


def inspect_dataset(root):
    """Decode every quartet, reject incomplete pairs, fingerprint normalized pixels."""
    root = Path(root)
    folders = (TARGET_FOLDER, *CONDITION_FOLDERS)
    file_sets = {}
    for folder in folders:
        path = root / folder
        if not path.is_dir():
            raise ValueError(f'缺少数据目录: {path}')
        file_sets[folder] = {
            p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        }
    names = sorted(set.union(*file_sets.values()))
    if not names:
        raise ValueError('数据集没有图片')
    for folder, files in file_sets.items():
        missing = sorted(set(names) - files)
        if missing:
            raise ValueError(f'{folder} 缺少对应图片: {", ".join(missing[:10])}')
    sizes = {folder: Counter() for folder in folders}
    digest = hashlib.sha256()
    for name in names:
        scene_size = None
        for folder in folders:
            image = load_rgb(root / folder / name)
            w, h = image.size
            if min(w, h) < GRID:
                raise ValueError(f'{folder}/{name} 太小，宽高至少为 {GRID}')
            if folder != CONDITION_FOLDERS[2]:
                if scene_size is not None and image.size != scene_size:
                    raise ValueError(f'{folder}/{name} 场景尺寸 {image.size} 与目标 {scene_size} 不一致')
                scene_size = image.size
            sizes[folder][f'{w}x{h}'] += 1
            digest.update(json.dumps([folder, name, w, h], ensure_ascii=False).encode())
            digest.update(image.tobytes())
    return {
        'count': len(names), 'names': names, 'fingerprint': digest.hexdigest(),
        'sizes': {k: dict(sorted(v.items())) for k, v in sizes.items()},
    }


def split_samples(names, fraction, seed, groups=None):
    """Keep groups on the same side; default grouping is the complete filename stem."""
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction 必须介于 0 和 1 之间')
    names = sorted(names)
    if groups is not None and set(groups) != set(names):
        raise ValueError('分组映射必须恰好覆盖全部图片文件名')
    grouped = {}
    for name in names:
        group = groups[name] if groups is not None else Path(name).stem
        if not isinstance(group, str) or not group:
            raise ValueError(f'无效来源分组: {name}')
        grouped.setdefault(group, []).append(name)
    keys = sorted(grouped)
    if len(keys) < 2:
        raise ValueError('至少需要两个独立来源组，才能拆分非空训练集和验证集')
    random.Random(seed).shuffle(keys)
    count = min(len(keys) - 1, max(1, round(len(keys) * fraction)))
    validation_groups = set(keys[:count])
    return {
        'train': sorted(n for k, group in grouped.items() if k not in validation_groups for n in group),
        'validation': sorted(n for k, group in grouped.items() if k in validation_groups for n in group),
    }


def grid_image(image, max_area):
    """Downscale only, preserving aspect ratio; center-crop at most 15px per axis."""
    if max_area < GRID * GRID or max_area > MAX_REFERENCE_AREA:
        raise ValueError('像素预算必须介于 256 和 1048576 之间')
    image = ImageOps.exif_transpose(image).convert('RGB')
    w, h = image.size
    scale = min(1.0, math.sqrt(max_area / (w * h)))
    nw, nh = int(w * scale), int(h * scale)
    if min(nw, nh) < GRID:
        raise ValueError(f'缩放后的图片 {(nw, nh)} 小于 {GRID} 像素网格')
    if (nw, nh) != image.size:
        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
    width, height = nw // GRID * GRID, nh // GRID * GRID
    left, top = (nw - width) // 2, (nh - height) // 2
    return image.crop((left, top, left + width, top + height))


def prepare_images(target, conditions, max_area):
    if len(conditions) != 3:
        raise ValueError('必须提供顺序固定的三张条件图')
    conditions = [ImageOps.exif_transpose(im).convert('RGB') for im in conditions]
    if conditions[0].size != conditions[1].size:
        raise ValueError('涂灰图与生成图 B 的场景尺寸不一致')
    if target is not None:
        target = ImageOps.exif_transpose(target).convert('RGB')
        if target.size != conditions[0].size:
            raise ValueError('目标与条件场景尺寸不一致')
        target = grid_image(target, max_area)
    # A native 32x32 swatch stays 32x32, independent of the scene pixel budget.
    return target, [grid_image(im, max_area) for im in conditions]


def reference_bucket(height, width, max_area):
    """Diffusers' discrete aspect ladder and nearest-bucket metric, using Pillow-only math."""
    resolution = min(round(math.sqrt(max_area)), round(math.sqrt(height * width)))
    pixels = resolution * resolution
    options = []
    for rw, rh in [(1, 1), (4, 5), (5, 4), (2, 3), (3, 2), (3, 4), (4, 3), (9, 16), (16, 9), (1, 2), (2, 1)]:
        h = max(16, round(math.sqrt(pixels / (rw / rh)) / 16) * 16)
        w = max(16, round(math.sqrt(pixels / (rw / rh)) * (rw / rh) / 16) * 16)
        # Reference rounding can exceed the multi-reference pipeline's absolute 1MP limit.
        while h * w > MAX_REFERENCE_AREA:
            if h >= w:
                h -= 16
            else:
                w -= 16
        if (h, w) not in options:
            options.append((h, w))
    return options[min(range(len(options)), key=lambda i: (abs(height * options[i][1] - width * options[i][0]), -i))]


def prepare_with_policy(target, conditions, policy, center_crop=True, random_flip=False, rng=None):
    """Bucket and geometry policy shared with inference (which always center-crops)."""
    rng = rng or random.Random(0)
    if len(conditions) != 3:
        raise ValueError('需要三张条件图')
    if not policy.get('use_aspect_ratio_buckets'):
        target, refs = prepare_images(target, conditions, policy['max_area'])
    else:
        scenes = ([target] if target is not None else []) + list(conditions[:2])
        scenes = [ImageOps.exif_transpose(im).convert('RGB') for im in scenes]
        if any(im.size != scenes[0].size for im in scenes):
            raise ValueError('分桶需要相同场景尺寸')
        width, height = scenes[0].size
        buckets = policy.get('buckets')
        if buckets:
            th, tw = buckets[min(range(len(buckets)), key=lambda i: (abs(height * buckets[i][1] - width * buckets[i][0]), -i))]
        else:
            th, tw = reference_bucket(height, width, policy['max_area'])
        scale = max(tw / width, th / height)
        nw, nh = max(tw, round(width * scale)), max(th, round(height * scale))
        left = (nw - tw) // 2 if center_crop else rng.randrange(nw - tw + 1)
        top = (nh - th) // 2 if center_crop else rng.randrange(nh - th + 1)
        scenes = [im.resize((nw, nh), Image.Resampling.LANCZOS).crop((left, top, left + tw, top + th)) for im in scenes]
        target = scenes.pop(0) if target is not None else None
        refs = [*scenes, grid_image(conditions[2], policy['max_area'])]
    if random_flip and rng.random() < 0.5:
        target = ImageOps.mirror(target) if target is not None else None
        refs = [ImageOps.mirror(im) if i < 2 else im for i, im in enumerate(refs)]
    return target, refs


def load_sample(root, name, max_area):
    root = Path(root)
    try:
        return prepare_images(
            load_rgb(root / TARGET_FOLDER / name),
            [load_rgb(root / folder / name) for folder in CONDITION_FOLDERS],
            max_area,
        )
    except ValueError as exc:
        raise ValueError(f'样本 {name}: {exc}') from exc


def image_tensor(image):
    # Keep dataset inspection usable with Pillow alone.
    import numpy as np
    import torch
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1.0


class RecolorDataset:
    def __init__(self, root, names, max_area):
        self.root, self.names, self.max_area = Path(root), list(names), max_area

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        target, conditions = load_sample(self.root, self.names[index], self.max_area)
        return {
            'name': self.names[index], 'target': image_tensor(target),
            'conditions': [image_tensor(im) for im in conditions],
        }
