"""Serializable run contracts and local-only filesystem checks."""

import hashlib
import json
import re
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def resolve_model_root(path):
    root = Path(path).expanduser().resolve()
    if (root / 'snapshots').is_dir() and not (root / 'transformer').is_dir():
        ref = root / 'refs/main'
        if not ref.is_file():
            raise ValueError('HF缓存没有refs/main，请直接提供snapshots中的具体目录')
        revision = ref.read_text().strip()
        if not re.fullmatch(r'[a-fA-F0-9]{7,64}', revision):
            raise ValueError('HF缓存refs/main不是有效的snapshot revision')
        root = root / 'snapshots' / revision
        if not root.is_dir():
            raise ValueError(f'HF缓存snapshot缺失: {revision}')
    return root


def tokenizer_subfolder(path):
    root = resolve_model_root(path)
    index = root / 'modular_model_index.json'
    if index.is_file():
        entry = read_json(index).get('tokenizer')
        if isinstance(entry, list) and len(entry) > 2 and isinstance(entry[2], dict):
            folder = entry[2].get('subfolder') or ''
            if not isinstance(folder, str) or Path(folder).is_absolute() or '..' in Path(folder).parts:
                raise ValueError('modular_model_index的tokenizer子目录必须为安全的本地相对路径')
            if not (root / folder).is_dir():
                raise ValueError(f'缺少modular_model_index指定的tokenizer子目录: {folder}')
            return folder
    if (root / 'tokenizer').is_dir():
        return 'tokenizer'
    if (root / 'text_encoder/tokenizer_config.json').is_file():
        return 'text_encoder'
    raise ValueError('缺少本地tokenizer/processor：需要tokenizer目录或text_encoder中的tokenizer配置')


def validate_local_model(path):
    root = resolve_model_root(path)
    for component in ('transformer', 'vae', 'text_encoder'):
        if not (root / component).is_dir():
            raise ValueError(f'缺少本地模型组件 {component}: {root}；请先手动下载完整 Diffusers 权重')
    transformer_config = root / 'transformer/config.json'
    if not transformer_config.is_file():
        raise ValueError(f'缺少 {transformer_config}')
    config = read_json(transformer_config)
    if config.get('flow_map', False):
        raise ValueError('本训练入口需要非蒸馏 FLUX.3（flow_map=False），不支持 flow_map 蒸馏权重')
    if config.get('_class_name') != 'Flux3Transformer2DModel':
        raise ValueError('需要 Flux3Transformer2DModel 的 Diffusers 格式本地权重')
    tokenizer_subfolder(root)
    digest = hashlib.sha256()
    for component in ('transformer', 'vae', 'text_encoder'):
        file = root / component / 'config.json'
        if not file.is_file():
            raise ValueError(f'缺少 {file}')
        digest.update(file.read_bytes())
        if not any((root / component).glob('*.safetensors')) and not any((root / component).glob('*.bin')):
            raise ValueError(f'缺少 {component} 的本地权重文件')
        for index in (root / component).glob('*.index.json'):
            shards = read_json(index).get('weight_map', {})
            if not isinstance(shards, dict):
                raise ValueError(f'无效权重索引: {index}')
            for shard in set(shards.values()):
                if not isinstance(shard, str) or Path(shard).name != shard or not (index.parent / shard).is_file():
                    raise ValueError(f'权重索引缺少本地分片或路径无效: {component}/{shard}')
    return str(root), digest.hexdigest()


def prepare_output(path, protected=(), resume=False):
    path = Path(path).expanduser().resolve()
    for item in protected:
        item = Path(item).expanduser().resolve()
        if path == item or path.is_relative_to(item) or item.is_relative_to(path):
            raise ValueError(f'输出目录与输入目录重叠: {path}')
    if path.exists() and (not path.is_dir() or (any(path.iterdir()) and not resume)):
        raise ValueError(f'输出目录非空，拒绝覆盖: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_checkpoint(output, requested):
    output = Path(output).resolve()
    if requested == 'latest':
        candidates = [p for p in output.glob('checkpoint-*')
                      if p.is_dir() and p.name.removeprefix('checkpoint-').isdigit()
                      and (p / 'training_state.json').is_file()]
        if not candidates:
            raise ValueError('输出目录没有可恢复的 checkpoint')
        path = max(candidates, key=lambda p: int(p.name.split('-')[-1]))
    else:
        path = Path(requested).expanduser().resolve()
    if path.parent != output or not (path / 'training_state.json').is_file():
        raise ValueError('只能恢复当前 output_dir 中包含 training_state.json 的 checkpoint')
    return path
