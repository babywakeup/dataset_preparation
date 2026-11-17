"""Opt-in CUDA acceptance with random tiny weights; no real user training or downloads."""
import json
import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('RUN_FLUX3_CUDA_TESTS') == '1', '显式开启RUN_FLUX3_CUDA_TESTS=1运行单卡CUDA验证')
class CudaValidationTests(unittest.TestCase):
    def test_unindexed_cuda_rng_and_training_validation_continuation(self):
        import torch
        from train_flux3_recolor_lora import cuda_device_index, main
        from flux3_recolor_test_utils import make_tiny_repository, make_dataset
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 1, '验收仅应暴露一张授权GPU')
        index = cuda_device_index(torch.device('cuda'))
        self.assertEqual(index, torch.cuda.current_device())
        before = torch.cuda.get_rng_state(index).clone()
        with torch.random.fork_rng(devices=[index]):
            torch.rand(8, device='cuda')
        torch.testing.assert_close(before, torch.cuda.get_rng_state(index), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data = make_tiny_repository(root / 'model'), make_dataset(root / 'data', 4)
            output = root / 'trained'
            main(['--pretrained_model_name_or_path', str(model), '--instance_data_dir', str(data),
                  '--output_dir', str(output), '--max_train_steps', '2', '--mixed_precision', 'bf16',
                  '--training_directions', 'b_to_a',
                  '--rank', '2', '--lora_alpha', '2', '--max_sequence_length', '8',
                  '--text_encoder_out_layers', '1', '--report_to', 'tensorboard',
                  '--checkpointing_steps', '1', '--validation_steps', '1',
                  '--num_validation_images', '1', '--validation_inference_steps', '1'])
            self.assertEqual(json.loads((output / 'training_state.json').read_text())['global_step'], 2)
            for step in (1, 2):
                self.assertTrue((output / f'checkpoint-{step}/training_state.json').is_file())
                self.assertTrue((output / f'validation-step-{step}/manifest.json').is_file())
            events = EventAccumulator(str(output / 'logs/flux3-recolor-lora')).Reload()
            tags = [tag for tag in events.Tags()['images'] if tag.startswith('validation/generated/')]
            self.assertEqual(len(tags), 1)
            self.assertEqual({event.step for event in events.Images(tags[0])}, {1, 2})


if __name__ == '__main__':
    unittest.main()
