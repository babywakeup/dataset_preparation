import json
import tempfile
import unittest
from pathlib import Path

import torch


class ReferenceTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flux3_recolor_test_utils import make_tiny_repository
        cls.shared = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.shared.cleanup)
        cls.model = make_tiny_repository(Path(cls.shared.name) / 'model')

    def setUp(self):
        from flux3_recolor_test_utils import make_dataset
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = make_dataset(self.root / 'data')

    def train(self, output, *extra):
        from train_flux3_recolor_lora import main
        return main(['--pretrained_model_name_or_path', str(self.model), '--instance_data_dir', str(self.data),
                     '--output_dir', str(output), '--mixed_precision', 'no', '--rank', '2', '--lora_alpha', '6',
                     '--max_sequence_length', '160', '--text_encoder_out_layers', '1', '--train_batch_size', '2',
                     '--gradient_accumulation_steps', '2', '--checkpointing_steps', '1', '--report_to', 'none',
                     '--num_validation_images', '0', '--resolution', '32', *extra])

    def test_cached_training_resume_matches_uninterrupted_parameters_rng_and_lr(self):
        from safetensors.torch import load_file
        full, resumed = self.root / 'full', self.root / 'resumed'
        self.train(full, '--max_train_steps', '4', '--cache_latents', '--lora_dropout', '0.1')
        full_rng = torch.rand(3)
        self.train(resumed, '--max_train_steps', '2', '--cache_latents', '--lora_dropout', '0.1')
        self.train(resumed, '--max_train_steps', '4', '--cache_latents', '--lora_dropout', '0.1',
                   '--resume_from_checkpoint', 'latest')
        torch.testing.assert_close(torch.rand(3), full_rng, rtol=0, atol=0)
        a, b = [load_file(str(p / 'pytorch_lora_weights.safetensors')) for p in (full, resumed)]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        history = [json.loads(line) for line in (full / 'train_history.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in history], [1, 2, 3, 4])

    def test_cosine_caption_dropout_validation_and_inference(self):
        from flux3_recolor.inference import main as infer
        captions = {f'{i}.png': 'recolor red ' * (i * 18 + 1) for i in range(7)}
        caption_path = self.root / 'captions.json'
        caption_path.write_text(json.dumps(captions))
        output = self.root / 'cosine'
        self.train(output, '--max_train_steps', '2', '--lr_scheduler', 'cosine', '--lr_warmup_steps', '0',
                   '--caption_json', str(caption_path), '--caption_dropout', '0.5',
                   '--num_validation_images', '1', '--validation_steps', '1', '--validation_inference_steps', '1',
                   '--lora_layers', 'to_q,to_k,to_v,to_out.0', '--offload')
        self.assertTrue((output / 'validation-final-2/manifest.json').is_file())
        history = [json.loads(line) for line in (output / 'train_history.jsonl').read_text().splitlines()]
        self.assertGreater(history[0]['lr'], history[1]['lr'])
        self.assertEqual(history[1]['lr'], 0.)
        infer(['--lora_dir', str(output), '--dataset_root', str(self.data), '--limit', '1',
               '--output_dir', str(self.root / 'inferred'), '--device', 'cpu', '--mixed_precision', 'no',
               '--num_inference_steps', '1'])
        with self.assertRaisesRegex(ValueError, 'max_train_steps'):
            self.train(output, '--max_train_steps', '3', '--lr_scheduler', 'cosine',
                       '--caption_json', str(caption_path), '--caption_dropout', '0.5',
                       '--lora_layers', 'to_q,to_k,to_v,to_out.0', '--offload',
                       '--resume_from_checkpoint', 'latest')

    def test_epoch_horizon_warmup_worker_and_checkpoint_limit(self):
        output = self.root / 'epochs'
        self.train(output, '--num_train_epochs', '2', '--lr_scheduler', 'constant_with_warmup',
                   '--lr_warmup_steps', '2', '--dataloader_num_workers', '1', '--checkpoints_total_limit', '2',
                   '--report_to', 'tensorboard')
        history = [json.loads(line) for line in (output / 'train_history.jsonl').read_text().splitlines()]
        self.assertEqual([row['lr'] for row in history], [5e-5, 1e-4, 1e-4, 1e-4])
        self.assertEqual(sorted(p.name for p in output.glob('checkpoint-*')), ['checkpoint-3', 'checkpoint-4'])
        state = json.loads((output / 'training_state.json').read_text())
        self.assertEqual((state['global_step'], state['epoch'], state['batch_offset']), (4, 2, 0))
        self.assertTrue(list(output.rglob('events.out.tfevents.*')))

    def test_fsdp_loading_uses_uniform_fp32_master_parameters(self):
        from train_flux3_recolor_lora import parse_args, load_training_components
        from accelerate import Accelerator
        args = parse_args(['--pretrained_model_name_or_path', str(self.model)])
        components = load_training_components(args, Accelerator(cpu=True), torch.bfloat16, is_fsdp=True)
        self.assertEqual({p.dtype for p in components['transformer'].parameters()}, {torch.float32})
        self.assertEqual(components['vae'].dtype, torch.bfloat16)

    def test_clipping_passes_the_full_model_for_fsdp_global_norm_dispatch(self):
        from unittest.mock import patch
        from accelerate import Accelerator
        original = Accelerator.clip_grad_norm_
        def check(accelerator, parameters, *args, **kwargs):
            parameters = list(parameters)
            expected = list(accelerator._models[0].parameters())
            self.assertEqual([id(p) for p in parameters], [id(p) for p in expected])
            return original(accelerator, parameters, *args, **kwargs)
        with patch.object(Accelerator, 'clip_grad_norm_', new=check):
            self.train(self.root / 'clipping', '--max_train_steps', '1')

    def test_prodigy_optimizer_checkpoint_resume(self):
        import importlib.util
        if importlib.util.find_spec('prodigyopt') is None:
            self.skipTest('可选依赖prodigyopt未安装')
        output = self.root / 'prodigy'
        self.train(output, '--max_train_steps', '1', '--optimizer', 'prodigy', '--learning_rate', '1')
        self.train(output, '--max_train_steps', '2', '--optimizer', 'prodigy', '--learning_rate', '1',
                   '--resume_from_checkpoint', 'latest')
        state = torch.load(output / 'checkpoint-2/optimizer.bin', weights_only=False)
        self.assertTrue(any(group.get('d', 0) > 0 for group in state['param_groups']))

    def test_tensorboard_validation_images_update_during_training_at_real_steps(self):
        from unittest.mock import patch
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import train_flux3_recolor_lora as trainer
        output = self.root / 'live_images'
        original = trainer.log_validation
        observed = []

        def observe(*args, **kwargs):
            original(*args, **kwargs)
            # Read the actual event file immediately after a training-time validation,
            # before end_training closes/flushed the tracker at the end of the run.
            reader = EventAccumulator(str(output / 'logs/flux3-recolor-lora')).Reload()
            tags = reader.Tags()['images']
            generated = [tag for tag in tags if tag.startswith('validation/generated/')]
            comparisons = [tag for tag in tags if tag.startswith('validation/comparison/')]
            self.assertEqual(len(generated), 1)
            self.assertEqual(len(comparisons), 1)
            events = reader.Images(generated[0])
            self.assertTrue(all((event.width, event.height) == (32, 32) for event in events))
            self.assertTrue(all((event.width, event.height) == (160, 32) for event in reader.Images(comparisons[0])))
            observed.append({event.step for event in events})

        with patch.object(trainer, 'log_validation', new=observe):
            self.train(output, '--max_train_steps', '2', '--num_validation_images', '1',
                       '--validation_steps', '1', '--validation_inference_steps', '1', '--report_to', 'tensorboard')
        self.assertEqual(observed[0], {1})
        self.assertEqual(observed[1], {1, 2})

    def test_checkpoint_is_committed_before_validation_failure_and_can_resume(self):
        from unittest.mock import patch
        import train_flux3_recolor_lora as trainer
        output = self.root / 'validation_failure'
        def fail(*args, **kwargs):
            checkpoint = output / 'checkpoint-1'
            self.assertTrue((checkpoint / 'training_state.json').is_file())
            self.assertTrue((checkpoint / 'pytorch_lora_weights.safetensors').is_file())
            self.assertTrue((checkpoint / 'optimizer.bin').is_file())
            raise RuntimeError('intentional validation failure')
        with patch.object(trainer, 'log_validation', new=fail):
            with self.assertRaisesRegex(RuntimeError, 'intentional validation failure'):
                self.train(output, '--max_train_steps', '2', '--num_validation_images', '1',
                           '--validation_steps', '1', '--validation_inference_steps', '1')
        self.train(output, '--max_train_steps', '2', '--resume_from_checkpoint', 'latest')
        self.assertEqual(json.loads((output / 'training_state.json').read_text())['global_step'], 2)


if __name__ == '__main__':
    unittest.main()
