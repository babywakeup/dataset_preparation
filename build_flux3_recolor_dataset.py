#!/usr/bin/env python
"""Prepare local recolor path datasets with image dimensions; no model weights are loaded."""
import argparse
import gc
import json
import logging
import math
import random
import tempfile
import struct
from pathlib import Path

from PIL import Image
from datasets import Dataset, DatasetDict, Features, List, Value, concatenate_datasets

from flux3_recolor import DEFAULT_PROMPT, task_folders
from flux3_recolor.data import SampleDataError, index_dataset, split_samples

logger = logging.getLogger(__name__)


def _png_exif(path):
    # Pillow's PNG getexif() can decompress pixels. Read only the optional eXIf
    # chunk, including when it follows IDAT, and seek over compressed payloads.
    with open(path, 'rb') as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(8)
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError('PNG chunk头不完整')
            length, kind = struct.unpack('>I4s', header)
            if stream.tell() + length + 4 > end:
                raise ValueError('PNG chunk不完整')
            if kind == b'eXIf':
                return stream.read(length)
            if kind == b'IEND':
                return None
            stream.seek(length + 4, 1)


def read_image_size(path):
    """Get displayed dimensions without RGB conversion or pixel decompression."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            if image.format == 'PNG':
                raw = image.info.get('exif') or _png_exif(path)
                exif = Image.Exif()
                if raw:
                    exif.load(raw)
            else:
                exif = image.getexif()
            if exif.get(274, 1) in (5, 6, 7, 8):
                height, width = width, height
        return {'height': height, 'width': width}
    except (OSError, ValueError, SyntaxError, struct.error, Image.DecompressionBombError) as exc:
        raise SampleDataError(f'无法读取图片尺寸 {path}: {exc}') from exc


def generate_samples(root, direction, names, subset_id, prompt=DEFAULT_PROMPT, captions=None, source_groups=None):
    target_folder, folders = task_folders(direction)
    for name in names:
        target = str(Path(root) / target_folder / name)
        conditions = [str(Path(root) / folder / name) for folder in folders]
        try:
            size = read_image_size(target)
            condition_sizes = [read_image_size(path) for path in conditions]
        except SampleDataError as exc:
            logger.warning('跳过子集 %s 的无效图片头 sample=%s: %s', subset_id, name, exc)
            continue
        source_id = source_groups[name] if source_groups is not None else Path(name).stem
        yield {'sample_id': f'{subset_id}:{name}', 'source_id': source_id,
               'target_path': target, 'condition_paths': conditions,
               'condition_sizes': condition_sizes, 'prompt': (captions or {}).get(name, prompt), **size}


def concatenate_subsets(subsets, ratios, seed):
    """Apply the mixing ratio here only; preserve each subset's original rows."""
    scale = max(len(dataset) / weight for dataset, weight in zip(subsets, ratios))
    parts = []
    for index, (dataset, weight) in enumerate(zip(subsets, ratios)):
        count = max(len(dataset), round(scale * weight))
        order = list(range(len(dataset)))
        random.Random(seed + index).shuffle(order)
        # Whole copies avoid losing originals; only a partial copy needs indices.
        copies, tail = divmod(count, len(dataset))
        parts.extend([dataset] * copies)
        if tail:
            parts.append(dataset.select(order[:tail]))
    return concatenate_datasets(parts)


def build_dataset(root, output, directions=('b_to_a', 'a_to_b'), mix_ratio=None,
                  validation_fraction=0.1, seed=42, num_proc=1, *, prompt=DEFAULT_PROMPT, captions=None, source_groups=None, sample_names=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    directions = list(directions)
    ratios = list(mix_ratio) if mix_ratio is not None else [1.0] * len(directions)
    if not directions or len(set(directions)) != len(directions):
        raise ValueError('子集不能为空或重复')
    for direction in directions:
        task_folders(direction)
    if (len(ratios) != len(directions)
            or any(not math.isfinite(weight) or weight <= 0 for weight in ratios)):
        raise ValueError('mix_ratio 必须为每个子集指定一个正数')
    if not 0 < validation_fraction < 1 or num_proc < 1:
        raise ValueError('validation_fraction 必须在0和1之间，num_proc必须为正整数')
    if captions is not None and (not isinstance(captions, dict) or any(not isinstance(v, str) for v in captions.values())):
        raise ValueError('captions必须是文件名到字符串的映射')
    if source_groups is not None and (not isinstance(source_groups, dict)
            or any(not isinstance(v, str) or not v for v in source_groups.values())):
        raise ValueError('source_groups必须是文件名到非空来源组字符串的映射')
    if output.exists():
        raise ValueError(f'输出目录已存在，拒绝覆盖: {output}')
    if output == root or output in root.parents:
        raise ValueError('输出目录不能包含输入数据目录')
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_names = set(sample_names) if sample_names is not None else None
    size_features = {'height': Value('int32'), 'width': Value('int32')}
    features = Features({'sample_id': Value('string'), 'source_id': Value('string'),
        'target_path': Value('string'), 'condition_paths': List(Value('string'), length=3),
        'condition_sizes': List(size_features, length=3), 'prompt': Value('string'), **size_features})
    # Build each subset once in a temporary Arrow cache, then persist only the
    # final train/validation DatasetDict. Only paths and small metadata are serialized.
    with tempfile.TemporaryDirectory(prefix='.recolor-build-', dir=output.parent) as cache:
        subsets = []
        counts = []
        for subset_id, direction in enumerate(directions):
            names = index_dataset(root, [direction])['names']
            if selected_names is not None:
                names = [name for name in names if name in selected_names]
            if not names:
                raise ValueError(f'子集 {direction} 没有选中的完整样本')
            if source_groups is not None and not set(names).issubset(source_groups):
                raise ValueError('source_groups必须覆盖所有构建样本')
            dataset = Dataset.from_generator(generate_samples, features=features,
                cache_dir=cache, num_proc=min(num_proc, len(names)), writer_batch_size=1000,
                gen_kwargs={'root': str(root), 'direction': direction, 'names': names,
                            'subset_id': subset_id, 'prompt': prompt, 'captions': captions,
                            'source_groups': source_groups})
            if not len(dataset):
                raise ValueError(f'子集 {direction} 没有可用图片')
            subsets.append(dataset)
            counts.append(len(dataset))
        merged = concatenate_subsets(subsets, ratios, seed)
        # Split only after concatenation; all copies and both views of one source
        # stay on the same side. Downstream code has no direction column/branch.
        sources = sorted(set(merged['source_id']))
        split = split_samples(sources, validation_fraction, seed, groups={s: s for s in sources})
        validation_sources = set(split['validation'])
        indices = {'train': [], 'validation': []}
        for index, source_id in enumerate(merged['source_id']):
            indices['validation' if source_id in validation_sources else 'train'].append(index)
        result = DatasetDict({key: merged.select(rows)
                              for key, rows in indices.items()})
        result.save_to_disk(str(output), max_shard_size='512MB')
        # NFS cannot unlink Arrow cache files while their memory maps are alive.
        del result, merged, subsets, dataset
        gc.collect()
    # Loading the saved data makes the returned object independent of temp cache.
    from datasets import load_from_disk
    result = load_from_disk(str(output))
    summary = {'input': str(root), 'subsets': dict(zip(directions, counts)),
               'mix_ratio': ratios, 'seed': seed, 'validation_fraction': validation_fraction,
               'rows': {key: len(value) for key, value in result.items()},
               'source_groups': {key: len(set(value['source_id'])) for key, value in result.items()},
               'storage': 'absolute_paths', 'dimensions': 'EXIF-oriented height/width; no embedded image bytes'}
    (output / 'build_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', default='./test')
    parser.add_argument('--output_dir', default='./datasets/recolor_paths')
    parser.add_argument('--training_directions', nargs='+', choices=['b_to_a', 'a_to_b'],
                        default=['b_to_a', 'a_to_b'], help='只用于构建各子集')
    parser.add_argument('--mix_ratio', nargs='+', type=float, help='与各子集对应，默认1:1；通过重复补足比例')
    parser.add_argument('--validation_fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_proc', type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    build_dataset(args.dataset_root, args.output_dir, args.training_directions, args.mix_ratio,
                  args.validation_fraction, args.seed, args.num_proc)


if __name__ == '__main__':
    main()
