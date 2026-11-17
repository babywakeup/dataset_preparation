"""Shared, metadata-only cache for prepared recolor path datasets."""

import gc
import hashlib
import json
import tempfile
from pathlib import Path

from filelock import FileLock

from . import DEFAULT_PROMPT, task_folders, subdataset_config
from .data import index_dataset


def dataset_cache_key(config):
    """Fingerprint effective build inputs without opening image contents."""
    root = Path(config['root']).resolve()
    directions = list(config.get('directions', ('b_to_a', 'a_to_b')))
    selected = config.get('sample_names')
    selected = set(selected) if selected is not None else None
    files = {}
    for direction in directions:
        target, conditions = task_folders(direction)
        folders = [target, *conditions]
        names = index_dataset(root, [direction])['names']
        if selected is not None:
            names = [name for name in names if name in selected]
        if not names:
            raise ValueError(f'子集 {direction} 没有选中的完整样本')
        for folder in folders:
            for name in names:
                stat = (root / folder / name).stat()
                files[(folder, name)] = [folder, name, stat.st_size, stat.st_mtime_ns]
    effective = {
        'root': str(root), 'directions': directions,
        'mix_ratio': list(config['mix_ratio']) if config.get('mix_ratio') is not None else [1.0] * len(directions),
        'validation_fraction': config.get('validation_fraction', 0.1),
        'seed': config.get('seed', 42), 'prompt': config.get('prompt', DEFAULT_PROMPT),
        'captions': config.get('captions'), 'source_groups': config.get('source_groups'),
        'sample_names': sorted(selected) if selected is not None else None,
    }
    manifest = {'format_version': 2, 'config': effective,
                'subdatasets': subdataset_config(directions), 'files': [files[key] for key in sorted(files)]}
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest(), manifest


def prepare_cached_dataset(config, cache_root=None, num_proc=1):
    """Build a missing key under a lock, then atomically publish its directory."""
    key, manifest = dataset_cache_key(config)
    cache_root = (Path(cache_root) if cache_root is not None else
                  Path(manifest['config']['root']).parent / 'datasets' / 'recolor_cache').resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / key
    with FileLock(str(cache_root / f'{key}.lock')):
        if (destination / 'cache_config.json').is_file() and (destination / 'dataset_dict.json').is_file():
            saved = json.loads((destination / 'cache_config.json').read_text(encoding='utf-8'))
            if saved != manifest:
                raise ValueError(f'缓存配置不匹配: {destination}')
            from datasets import load_from_disk
            cached = load_from_disk(str(destination))
            del cached
            gc.collect()
            return str(destination)
        if destination.exists():
            raise ValueError(f'缓存目录不完整，拒绝覆盖: {destination}')
        from build_flux3_recolor_dataset import build_dataset
        with tempfile.TemporaryDirectory(prefix=f'.{key}-', dir=cache_root) as staging:
            output = Path(staging) / 'dataset'
            settings = dict(manifest['config'])
            settings['num_proc'] = num_proc
            dataset = build_dataset(output=output, **settings)
            del dataset
            gc.collect()
            (output / 'cache_config.json').write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
            output.rename(destination)
    return str(destination)
