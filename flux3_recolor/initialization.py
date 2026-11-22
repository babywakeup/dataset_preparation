"""Read trained component structure without restoring optimizer or data state."""
import json
from pathlib import Path

from .component_io import load_component_config
from .config import read_json


def inherit_model_args(args, directory, resume=False):
    """Inherit adapter structure from its actual metadata, not today's defaults."""
    from safetensors import safe_open
    directory = Path(directory).expanduser().resolve()
    manifest = load_component_config(directory)
    with safe_open(directory / manifest['transformer']['path'] / 'pytorch_lora_weights.safetensors',
                   framework='pt', device='cpu') as weights:
        metadata = json.loads((weights.metadata() or {}).get('lora_adapter_metadata', '{}'))
    adapter = {key.removeprefix('transformer.'): value for key, value in metadata.items()
               if key.startswith('transformer.')}
    if not {'r', 'lora_alpha', 'target_modules'}.issubset(adapter):
        raise ValueError('初始化LoRA缺少rank/alpha/target_modules元数据')
    saved = read_json(directory / 'recolor_config.json') if (directory / 'recolor_config.json').is_file() else {}
    for name, value in [('rank', adapter['r']), ('lora_alpha', adapter['lora_alpha']),
                        ('lora_dropout', adapter.get('lora_dropout', 0.))]:
        explicit = getattr(args, name)
        if explicit is not None and explicit != value:
            raise ValueError(f'{name} 与已有LoRA不一致: {explicit} != {value}')
        setattr(args, name, value)
    targets = adapter['target_modules']
    layers = targets if isinstance(targets, str) else ','.join(sorted(targets))
    if args.lora_layers is not None:
        requested = args.lora_layers if isinstance(targets, str) else set(x.strip() for x in args.lora_layers.split(','))
        expected = targets if isinstance(targets, str) else set(targets)
        if requested != expected:
            source = 'resume_from_checkpoint' if resume else 'init_from'
            saved_layers = targets if isinstance(targets, str) else ', '.join(sorted(targets))
            details = (f'lora_layers 与已有LoRA的target_modules不一致\n'
                       f'  {source} ({directory}): {saved_layers}\n'
                       f'  lora_layers: {requested if isinstance(requested, str) else ", ".join(sorted(requested))}')
            if not isinstance(targets, str):
                details += (f'\n  新增模块: {", ".join(sorted(requested - expected)) or "无"}'
                            f'\n  缺少模块: {", ".join(sorted(expected - requested)) or "无"}')
            raise ValueError(details)
    else:
        args.lora_layers = saved.get('lora_layers') if resume and 'lora_layers' in saved else layers
    adapter['inference_mode'] = False
    args._adapter_config = adapter
    args._source_signature = saved.get('model_config_signature')
    args.pretrained_model_name_or_path = args.pretrained_model_name_or_path or manifest['base_model']
    if manifest['text_encoder']['type'] == 'learned':
        args.train_text_encoder = True
        encoder = read_json(directory / manifest['text_encoder']['path'] / 'config.json')
        if encoder['length'] > args.max_sequence_length:
            raise ValueError('max_sequence_length 小于已有text_encoder的length')
    return str(directory)
