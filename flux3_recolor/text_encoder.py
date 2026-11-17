"""Independent Transformers model for a shared trainable text condition."""
from pathlib import Path

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

SUBFOLDER = 'text_encoder'
DEFAULT_INITIAL_PATH = 'outputs/flux3_task_embeddingnew/checkpoint-9900/task_embedding.safetensors'


class TextEncoderConfig(PretrainedConfig):
    model_type = 'recolor_text_encoder'

    def __init__(self, length=240, hidden_size=20480, initial_path=DEFAULT_INITIAL_PATH, **kwargs):
        super().__init__(**kwargs)
        if type(length) is not int or length < 1 or type(hidden_size) is not int or hidden_size < 1:
            raise ValueError('text encoder length and hidden_size must be positive integers')
        self.length = length
        self.hidden_size = hidden_size
        self.initial_path = kwargs.pop('initial_tensor_path', initial_path)
        self.initial_tensor_path = self.initial_path


class TextEncoderModel(PreTrainedModel):
    config_class = TextEncoderConfig
    main_input_name = 'batch_size'

    def __init__(self, config):
        super().__init__(config)
        self.weight = torch.nn.Parameter(torch.zeros(1, config.length, config.hidden_size))
        self._load_initial_tensor(getattr(config, 'initial_path', None) or getattr(config, 'initial_tensor_path', None))
        self.post_init()

    def _load_initial_tensor(self, path):
        if not path:
            return
        file_path = Path(path)
        if not file_path.is_file():
            repo_file = Path(__file__).resolve().parent.parent / path
            if repo_file.is_file():
                file_path = repo_file
            else:
                return
        try:
            from safetensors.torch import load_file
            data = load_file(str(file_path))
            tensor = data.get('embedding', data.get('weight', data.get('model.weight')))
            if tensor is None and len(data) == 1:
                tensor = next(iter(data.values()))
            if tensor is None:
                return
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                if tensor.shape[-1] == self.config.hidden_size:
                    copy_len = min(self.config.length, tensor.shape[1])
                    with torch.no_grad():
                        self.weight.data[:, :copy_len, :].copy_(tensor[:, :copy_len, :].float())
        except Exception:
            return

    @classmethod
    def from_features(cls, features, initial_path=DEFAULT_INITIAL_PATH):
        if features.ndim != 3 or features.shape[0] != 1 or min(features.shape) < 1:
            raise ValueError('text encoder shape must be [1, length, hidden_size]')
        model = cls(TextEncoderConfig(length=features.shape[1], hidden_size=features.shape[2],
                                      initial_path=initial_path))
        with torch.no_grad():
            model.weight.copy_(features.detach().float())
            model._load_initial_tensor(initial_path)
        return model

    def forward(self, batch_size=1):
        return BaseModelOutput(last_hidden_state=self.weight.expand(batch_size, -1, -1))


def current_text_embeddings(model, dtype=None):
    return model.weight.detach().to(dtype=dtype or model.weight.dtype)


def save_text_encoder(model, directory):
    model.save_pretrained(Path(directory) / SUBFOLDER)


def load_text_embeddings(directory, device='cpu', dtype=None):
    model = TextEncoderModel.from_pretrained(Path(directory) / SUBFOLDER, local_files_only=True)
    value = current_text_embeddings(model)
    if (value.ndim != 3 or value.shape[0] != 1 or min(value.shape) < 1
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('invalid text encoder shape or values')
    return value.to(device=device, dtype=dtype or value.dtype)


def load_text_encoder_into(model, directory):
    value = load_text_embeddings(directory)
    if value.shape != model.weight.shape:
        raise ValueError(f'text encoder shape mismatch: {tuple(value.shape)} != {tuple(model.weight.shape)}')
    with torch.no_grad():
        model.weight.copy_(value.to(model.weight))
