import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('RUN_FLUX3_DDP_TESTS') == '1', '显式设置RUN_FLUX3_DDP_TESTS=1运行两进程CPU验收')
class DistributedReferenceTrainerTests(unittest.TestCase):
    def test_two_cpu_ranks_sync_parameters_resume_and_survive_validation(self):
        from flux3_recolor_test_utils import make_tiny_repository, make_dataset
        from safetensors.torch import load_file
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data = make_tiny_repository(root / 'model'), make_dataset(root / 'data', 13)
            captions = root / 'captions.json'
            captions.write_text(json.dumps({f'{i}.png': 'Recolor using the swatch. ' * (i + 1) for i in range(13)}))
            base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2',
                    'tests/flux3_reference_ddp_worker.py', '--pretrained_model_name_or_path', str(model),
                    '--instance_data_dir', str(data), '--mixed_precision', 'no', '--report_to', 'none',
                    '--rank', '2', '--lora_alpha', '6', '--lora_dropout', '0.1', '--resolution', '32',
                    '--max_sequence_length', '8', '--text_encoder_out_layers', '1',
                    '--train_batch_size', '2', '--gradient_accumulation_steps', '2',
                    '--training_directions', 'b_to_a', 'a_to_b', '--checkpointing_steps', '1', '--validation_steps', '1',
                    '--num_validation_images', '1', '--validation_inference_steps', '1',
                    '--caption_json', str(captions), '--caption_dropout', '0.2']
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'ACCELERATE_USE_CPU': 'true', 'OMP_NUM_THREADS': '1',
                   'PYTHONPATH': str(Path.cwd()) + ':' + str(Path.cwd() / 'tests'), 'PYTHONDONTWRITEBYTECODE': '1'}
            def run(output, steps, resume=False):
                command = [*base, '--output_dir', str(output), '--max_train_steps', str(steps)]
                if resume:
                    command.extend(['--resume_from_checkpoint', 'latest'])
                log = root / f'{output.name}-{steps}.log'
                with log.open('w') as stream:
                    proc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
                self.assertEqual(proc.returncode, 0, log.read_text()[-14000:])
            full, resumed = root / 'full', root / 'resumed'
            run(full, 3)
            run(resumed, 1)
            mid_epoch = json.loads((resumed / 'checkpoint-1/training_state.json').read_text())
            self.assertEqual((mid_epoch['epoch'], mid_epoch['batch_offset']), (0, 2))
            run(resumed, 3, resume=True)
            for output in (full, resumed):
                a, b = [load_file(str(output / f'rank-artifacts/rank-{rank}.safetensors')) for rank in (0, 1)]
                for name in a:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
                shards = [json.loads((output / f'rank-artifacts/rank-{rank}.json').read_text()) for rank in (0, 1)]
                self.assertTrue(set(shards[0]).isdisjoint(shards[1]))
                self.assertTrue((output / 'validation-step-3/manifest.json').is_file())
            a, b = [load_file(str(output / 'transformer/pytorch_lora_weights.safetensors')) for output in (full, resumed)]
            for name in a:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
