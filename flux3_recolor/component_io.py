"""Component-oriented export of trained weights and their base-model reference."""
from pathlib import Path

from .config import read_json, write_json


def save_components(transformer, output, base_model, text_encoder=None, state=None, dtype=None):
    from diffusers.loaders import Flux3LoraLoaderMixin
    from peft.utils import get_peft_model_state_dict
    output = Path(output)
    state = get_peft_model_state_dict(transformer) if state is None else state
    state = {key: value.detach().to(device='cpu', dtype=dtype or value.dtype).contiguous()
             for key, value in state.items()}
    Flux3LoraLoaderMixin.save_lora_weights(str(output / 'transformer'),
        transformer_lora_layers=state,
        transformer_lora_adapter_metadata=transformer.peft_config['default'].to_dict())
    if text_encoder is not None:
        text_encoder.save_pretrained(output / 'text_encoder')
    write_json(output / 'model_config.json', {
        'format_version': 1, 'base_model': str(base_model),
        'transformer': {'type': 'lora', 'path': 'transformer'},
        'text_encoder': ({'type': 'learned', 'path': 'text_encoder'}
                         if text_encoder is not None else {'type': 'base'}),
        'vae': {'type': 'base'},
    })


def load_component_config(directory):
    config = read_json(Path(directory) / 'model_config.json')
    if (config.get('format_version') != 1 or not config.get('base_model')
            or config.get('transformer') != {'type': 'lora', 'path': 'transformer'}
            or config.get('vae') != {'type': 'base'}
            or config.get('text_encoder') not in (
                {'type': 'base'}, {'type': 'learned', 'path': 'text_encoder'})):
        raise ValueError('模型组件配置不兼容')
    return config
