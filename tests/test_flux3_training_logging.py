import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class TrainingLoggingTests(unittest.TestCase):
    def test_reduce_and_write_only_at_interval_with_sample_weighted_loss(self):
        from flux3_recolor.training_logging import TrainingLogger
        with tempfile.TemporaryDirectory() as tmp, patch('flux3_recolor.training_logging.tqdm') as progress:
            args = SimpleNamespace(output_dir=tmp, max_train_steps=3, logging_steps=2)
            accelerator = SimpleNamespace(device=torch.device('cpu'), is_main_process=True,
                reduce=Mock(side_effect=lambda value, **kw: value), log=Mock())
            logger = TrainingLogger(args, accelerator, 0)
            logger.step(torch.tensor(10.), 2, 1, 0.01)
            accelerator.reduce.assert_not_called()
            self.assertFalse((Path(tmp)/'train_history.jsonl').exists())
            logger.step(torch.tensor(30.), 3, 2, 0.005)
            self.assertEqual(accelerator.reduce.call_count, 1)
            accelerator.log.assert_called_with({'loss': 8., 'lr': 0.005}, step=2)
            logger.step(torch.tensor(7.), 1, 3, 0.)
            logger.close()
            history = [json.loads(line) for line in (Path(tmp)/'train_history.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in history], [2,3])
            self.assertEqual(history[-1]['loss'], 7.)
            self.assertEqual(progress.return_value.update.call_count, 3)
            progress.return_value.close.assert_called_once()
