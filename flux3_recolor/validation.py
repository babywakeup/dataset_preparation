"""Training validation deduplication, text conditions and image output."""
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import prepare_output


class ValidationRuns:
    """Remember successful renders only within this training process."""
    def __init__(self):
        self.completed = set()

    def run(self, step, config, sample_ids, seed, steps, action):
        key = (step, json.dumps(config, sort_keys=True), tuple(sample_ids), seed, steps)
        if key in self.completed:
            return
        action()
        self.completed.add(key)


@contextmanager
def validation_output(destination, protected):
    """Render in a sibling directory; replace an older run after success."""
    destination = prepare_output(destination, protected=protected, resume=True)
    with TemporaryDirectory(prefix=f'.{destination.name}-', dir=destination.parent) as tmp:
        staging = Path(tmp)
        yield staging
        shutil.rmtree(destination)
        staging.rename(destination)


def pipeline_text_embeddings(pipeline, embeddings):
    """Use the shared, current text encoder; preserve the frozen CFG negative."""
    import torch
    text_encoder = getattr(pipeline, 'recolor_text_encoder', None)
    if text_encoder is None:
        return embeddings
    with torch.no_grad():
        positive = text_encoder(batch_size=1).last_hidden_state.detach().to(embeddings[0])
    return positive, embeddings[1]


def log_validation_images(trackers, destination, label, step):
    from .data import load_rgb
    samples = sorted(p.parent for p in destination.rglob('generated.png') if p.is_file())
    phase = 'test' if label.startswith('final-') else 'validation'
    for tracker in trackers:
        if tracker.name == 'tensorboard':
            import numpy as np
            for sample in samples:
                tag = sample.relative_to(destination).as_posix()
                for kind in ('generated', 'comparison'):
                    tracker.writer.add_image(f'{phase}/{kind}/{tag}',
                        np.asarray(load_rgb(sample / f'{kind}.png')), global_step=step, dataformats='HWC')
            tracker.writer.flush()
        elif tracker.name == 'wandb':
            import wandb
            tracker.log({f'{phase}/{kind}': [wandb.Image(load_rgb(sample / f'{kind}.png'),
                caption=sample.relative_to(destination).as_posix()) for sample in samples]
                for kind in ('generated', 'comparison')}, step=step)
