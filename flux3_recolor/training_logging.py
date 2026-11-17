"""Interval loss reporting and optimizer-step progress, separate from training math."""
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm


class TrainingLogger:
    def __init__(self, args, accelerator, initial_step):
        self.args, self.accelerator = args, accelerator
        self.loss_sum = torch.zeros((), device=accelerator.device)
        self.samples = 0
        self.progress = tqdm(total=args.max_train_steps, initial=initial_step, desc='Training',
                             unit='step', disable=not accelerator.is_main_process)

    def step(self, loss_sum, samples, step, lr):
        self.progress.update(1)
        self.loss_sum.add_(loss_sum)
        self.samples += samples
        if step % self.args.logging_steps and step != self.args.max_train_steps:
            return
        stats = torch.stack((self.loss_sum, self.loss_sum.new_tensor(self.samples)))
        loss_sum, samples = self.accelerator.reduce(stats, reduction='sum').tolist()
        logs = {'loss': loss_sum / samples, 'lr': lr}
        self.accelerator.log(logs, step=step)
        if self.accelerator.is_main_process:
            self.progress.set_postfix(loss=f'{logs["loss"]:.6f}', lr=lr)
            self.progress.write(f'step={step}/{self.args.max_train_steps} loss={logs["loss"]:.6f} lr={lr:.8g}')
            with (Path(self.args.output_dir) / 'train_history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'step': step, **logs}) + '\n')
        self.loss_sum.zero_()
        self.samples = 0

    def close(self):
        self.progress.close()
