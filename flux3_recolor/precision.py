"""Fixed Transformer autocast boundary; VAE modules and weights stay untouched."""
from contextlib import contextmanager
from functools import wraps
from types import MethodType

import torch


def set_transformer_autocast(model, dtype):
    """Use the same compute dtype in training and the native inference pipeline."""
    if not hasattr(model, '_recolor_original_forward'):
        model._recolor_original_forward = model.forward
        @wraps(model.forward.__func__)
        def forward(self, *args, **kwargs):
            device = next(self.parameters()).device
            dtype = self._recolor_compute_dtype
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                return self._recolor_original_forward(*args, **kwargs)
        model.forward = MethodType(forward, model)
    model._recolor_compute_dtype = dtype
    return model


@contextmanager
def pipeline_forward(model):
    """Temporarily bypass Accelerate's output-FP32 conversion for native sampling.

    The model's own Transformer autocast remains active. Use on an unwrapped
    model during synchronized validation, then restore its training forward.
    """
    wrapped = model.forward
    model.forward = getattr(model, '_original_forward', wrapped)
    try:
        yield model
    finally:
        model.forward = wrapped
