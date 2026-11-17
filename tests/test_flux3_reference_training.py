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
                     '--num_validation_images', '0', '--resolution', '32', '--training_directions', 'b_to_a',
                     '--logging_steps', '1', *extra])

    def test_bidirectional_training_resume_validation_and_directional_inference(self):
        from safetensors.torch import load_file
        from flux3_recolor.inference import main as infer
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from flux3_recolor.data import split_samples
        # A legal filename can collide with the generated artifact basename.
        for folder in self.data.iterdir():
            (folder / '6.png').rename(folder / 'generated.png')
        names = [f'{i}.png' for i in range(6)] + ['generated.png']
        seed = next(seed for seed in range(100) if split_samples(names, 0.1, seed)['validation'] == ['generated.png'])
        options = ['--training_directions', 'b_to_a', 'a_to_b', '--caption_dropout', '0.2',
                   '--num_validation_images', '1', '--validation_steps', '1', '--validation_inference_steps', '1',
                   '--report_to', 'tensorboard', '--seed', str(seed)]
        full, resumed = self.root / 'both-full', self.root / 'both-resumed'
        self.train(full, '--max_train_steps', '3', *options)
        self.train(resumed, '--max_train_steps', '1', *options)
        stale = resumed / 'validation-step-2'
        stale.mkdir()
        (stale / 'old.png').write_bytes(b'previous interrupted run')
        self.train(resumed, '--max_train_steps', '3', '--resume_from_checkpoint', 'latest', *options)
        self.assertFalse((stale / 'old.png').exists())
        self.assertTrue((stale / 'manifest.json').is_file())
        self.assertFalse((full / 'validation-final-3').exists())
        for key, value in load_file(str(full / 'transformer/pytorch_lora_weights.safetensors')).items():
            torch.testing.assert_close(value, load_file(str(resumed / 'transformer/pytorch_lora_weights.safetensors'))[key], rtol=0, atol=0)
        config = json.loads((full / 'recolor_config.json').read_text())
        self.assertEqual(config['training_directions'], ['b_to_a', 'a_to_b'])
        self.assertEqual(config['dataset_format'], 'recolor_paths_v1')
        split = json.loads((full / 'split.json').read_text())
        manifest = json.loads((full / 'validation-step-1/manifest.json').read_text())
        self.assertEqual(len(manifest['samples']), 1)
        self.assertEqual(manifest['samples'][0]['sample_id'], split['validation'][0])
        self.assertNotIn('direction', manifest['samples'][0])
        events = EventAccumulator(str(full / 'logs/flux3-recolor-lora'), size_guidance={'images': 0}).Reload()
        self.assertTrue((full / 'validation-step-1/0000/comparison.png').is_file())
        self.assertEqual(sorted({event.step for event in events.Images('validation/generated/0000')}), [1, 2, 3])
        history = [json.loads(line) for line in (full / 'train_history.jsonl').read_text().splitlines()]
        self.assertTrue(all('loss' in row for row in history))
        self.assertFalse(any(key.startswith('loss_') for row in history for key in row))
        self.assertTrue((full / 'model_config.json').is_file())
        self.assertFalse((full / 'vae').exists())
        self.assertFalse((full / 'text_encoder').exists())
        # Component manifest, rather than an old training-only path, locates the base.
        config['model_path'] = 'unused-training-path'
        (full / 'recolor_config.json').write_text(json.dumps(config))
        inferred = self.root / 'unified-inference'
        infer(['--lora_dir', str(full), '--dataset_root', str(self.data), '--limit', '1',
               '--output_dir', str(inferred), '--device', 'cpu', '--mixed_precision', 'no', '--num_inference_steps', '1'])
        result = json.loads((inferred / 'manifest.json').read_text())
        self.assertEqual(result['samples'][0]['sample_id'], split['validation'][0])
        self.assertNotIn('direction', result['samples'][0])

    def test_logging_interval_averages_loss_and_tqdm_tracks_optimizer_steps(self):
        from unittest.mock import patch
        from safetensors.torch import load_file
        baseline, interval = self.root / 'every_step', self.root / 'interval'
        options = ['--train_batch_size', '1', '--gradient_accumulation_steps', '1', '--max_train_steps', '5']
        self.train(baseline, *options, '--logging_steps', '1')
        with patch('flux3_recolor.training_logging.tqdm') as progress:
            self.train(interval, *options, '--logging_steps', '2')
        read = lambda output: [json.loads(line) for line in (output/'train_history.jsonl').read_text().splitlines()]
        every, sparse = read(baseline), read(interval)
        self.assertEqual([row['step'] for row in sparse], [2,4,5])
        for row, entries in zip(sparse, [every[:2], every[2:4], every[4:]]):
            self.assertAlmostEqual(row['loss'], sum(x['loss'] for x in entries)/len(entries), places=6)
        self.assertEqual(sum(call.args[0] for call in progress.return_value.update.call_args_list), 5)
        progress.return_value.close.assert_called_once()
        for key, value in load_file(str(baseline/'transformer/pytorch_lora_weights.safetensors')).items():
            torch.testing.assert_close(value, load_file(str(interval/'transformer/pytorch_lora_weights.safetensors'))[key], rtol=0, atol=0)
        with patch('flux3_recolor.training_logging.tqdm') as progress:
            self.train(interval, '--train_batch_size', '1', '--gradient_accumulation_steps', '1',
                       '--max_train_steps', '8', '--logging_steps', '3', '--resume_from_checkpoint', 'latest')
        self.assertEqual(progress.call_args.kwargs['initial'], 5)
        self.assertEqual(sum(call.args[0] for call in progress.return_value.update.call_args_list), 3)
        self.assertEqual([row['step'] for row in read(interval)], [2,4,5,6,8])

    def test_online_training_resume_matches_uninterrupted_parameters_rng_and_lr(self):
        from safetensors.torch import load_file
        full, resumed = self.root / 'full', self.root / 'resumed'
        self.train(full, '--max_train_steps', '4', '--lora_dropout', '0.1')
        full_rng = torch.rand(3)
        self.train(resumed, '--max_train_steps', '2', '--lora_dropout', '0.1')
        self.train(resumed, '--max_train_steps', '4', '--lora_dropout', '0.1',
                   '--resume_from_checkpoint', 'latest')
        torch.testing.assert_close(torch.rand(3), full_rng, rtol=0, atol=0)
        a, b = [load_file(str(p / 'transformer/pytorch_lora_weights.safetensors')) for p in (full, resumed)]
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
                   '--lora_layers', 'to_q,to_k,to_v,to_out.0', '--offload',
                   '--final_validation_prompt', 'Recolor using only the provided swatch.')
        self.assertTrue((output / 'validation-final-2/manifest.json').is_file())
        from PIL import Image
        saved = json.loads((output / 'recolor_config.json').read_text())
        self.assertEqual(saved['preprocessing']['color_swatch_size'], 128)
        swatches = list((output / 'validation-final-2').glob('*/condition_3.png'))
        self.assertTrue(swatches)
        with Image.open(swatches[0]) as image:
            self.assertEqual(image.size, (128, 128))
        history = [json.loads(line) for line in (output / 'train_history.jsonl').read_text().splitlines()]
        self.assertGreater(history[0]['lr'], history[1]['lr'])
        self.assertEqual(history[1]['lr'], 0.)
        infer(['--lora_dir', str(output), '--dataset_root', str(self.data), '--limit', '1',
               '--output_dir', str(self.root / 'inferred'), '--device', 'cpu', '--mixed_precision', 'no',
               '--num_inference_steps', '1'])
        with Image.open(next((self.root / 'inferred').glob('*/condition_3.png'))) as image:
            self.assertEqual(image.size, (128, 128))
        with self.assertRaisesRegex(ValueError, 'max_train_steps'):
            self.train(output, '--max_train_steps', '3', '--lr_scheduler', 'cosine',
                       '--caption_json', str(caption_path), '--caption_dropout', '0.5',
                       '--lora_layers', 'to_q,to_k,to_v,to_out.0', '--offload',
                       '--final_validation_prompt', 'Recolor using only the provided swatch.',
                       '--resume_from_checkpoint', 'latest')

    def test_prompt_augmentation_embeddings_dropout_and_validation(self):
        output = self.root / 'prompt_variants'
        self.train(output, '--max_train_steps', '2', '--prompt_augmentation', '--caption_dropout', '0.5',
                   '--num_validation_images', '1', '--validation_steps', '1',
                   '--validation_inference_steps', '1', '--offload')
        saved = json.loads((output / 'recolor_config.json').read_text())
        self.assertEqual(len(saved['training_prompts']), 6)
        self.assertEqual(saved['prompt'], saved['training_prompts'][0])
        self.assertTrue((output / 'validation-step-2/manifest.json').is_file())
        self.train(output, '--max_train_steps', '3', '--prompt_augmentation', '--caption_dropout', '0.5',
                   '--num_validation_images', '1', '--validation_steps', '1',
                   '--validation_inference_steps', '1', '--offload', '--resume_from_checkpoint', 'latest')

    def test_accumulation_matches_full_batch_gradient_including_short_tail(self):
        from unittest.mock import patch
        from accelerate import Accelerator
        captured = []
        original_clip = Accelerator.clip_grad_norm_
        def capture(accelerator, parameters, *args, **kwargs):
            params = list(parameters)
            captured.append(torch.cat([p.grad.detach().flatten().clone() for p in params if p.grad is not None]))
            return original_clip(accelerator, params, *args, **kwargs)
        def objective(model, target, references, *args, **kwargs):
            # Independent deterministic per-sample objective isolates accumulation
            # scaling from random noise/timestep draws and batch ordering.
            model = model.module if hasattr(model, 'module') else model
            parameter = next(p for n, p in model.named_parameters() if 'lora_B' in n)
            losses = (parameter.mean() - target.float().flatten(1).mean(1)).square()
            return losses if kwargs.get('return_per_sample') else losses.mean()
        with patch('flux3_recolor.modeling.flow_matching_loss', side_effect=objective), patch.object(
                Accelerator, 'clip_grad_norm_', new=capture):
            # Six training examples: 4+2 exercises an undersized last microbatch;
            # 2+2+2 with accumulation=4 also exercises an incomplete accumulation group.
            for batch_size, accumulation in ((6, 1), (4, 2), (2, 4)):
                output = self.root / f'gradient-{batch_size}-{accumulation}'
                self.train(output, '--max_train_steps', '1', '--train_batch_size', str(batch_size),
                           '--gradient_accumulation_steps', str(accumulation))
        self.assertEqual(len(captured), 3)
        for gradient in captured[1:]:
            torch.testing.assert_close(gradient, captured[0], rtol=2e-5, atol=1e-7)

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
        self.assertEqual(components['vae'].dtype, torch.float32)

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
            self.assertTrue((checkpoint / 'transformer/pytorch_lora_weights.safetensors').is_file())
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
