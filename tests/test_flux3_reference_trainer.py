from flux3_recolor_test_utils import make_training_dataset
"""Reference-style trainer contracts; all model fixtures are generated locally."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image


class ReferenceTrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        from flux3_recolor import CONDITION_FOLDERS, TARGET_FOLDER
        for folder in (TARGET_FOLDER, *CONDITION_FOLDERS):
            (self.root / folder).mkdir()
            for i in range(5):
                size = (32, 32) if folder == CONDITION_FOLDERS[2] else ((64, 96) if i < 4 else (96, 64))
                Image.linear_gradient('L').resize(size).convert('RGB').save(self.root / folder / f'{i}.png')

    def args(self, *extra):
        from train_flux3_recolor_lora import parse_args
        return parse_args(['--pretrained_model_name_or_path', 'unused-local-model', '--instance_data_dir', str(self.root), '--report_to', 'none',
                           '--training_directions', 'b_to_a', *extra])

    def test_prompt_augmentation_is_reproducible_and_custom_prompt_wins(self):
        from flux3_recolor import DEFAULT_PROMPT
        dataset = make_training_dataset(self.args('--prompt_augmentation'), ['0.png'])
        prompts = []
        for epoch in range(18):
            dataset.set_epoch(epoch)
            prompts.append(dataset[0]['instance_prompt'])
        self.assertGreater(len(set(prompts)), 1)
        dataset.set_epoch(0)
        self.assertEqual(dataset[0]['instance_prompt'], prompts[0])
        self.assertEqual(len(dataset.args.training_prompts), 6)
        self.assertEqual(self.args().training_prompts, [DEFAULT_PROMPT])
        custom = self.args('--prompt_augmentation', '--prompt', 'custom')
        self.assertEqual(custom.training_prompts, ['custom'])
        import json
        captions = self.root / 'captions.json'
        captions.write_text(json.dumps({'0.png': 'per-image instruction'}))
        captioned = make_training_dataset(self.args('--prompt_augmentation', '--caption_json', str(captions)), ['0.png'])
        self.assertEqual(captioned[0]['instance_prompt'], 'per-image instruction')

    def test_legacy_single_prompt_resume_defaults_and_augmentation_mismatch(self):
        from train_flux3_recolor_lora import validate_training_resume, TRAINER, STATE_VERSION
        saved = {'instance_prompt': 'same prompt'}
        current = dict(saved, prompt_augmentation=False, training_prompts=['same prompt'])
        state = {'trainer': TRAINER, 'version': STATE_VERSION}
        validate_training_resume(saved, current, state)
        with self.assertRaisesRegex(ValueError, '恢复配置不一致'):
            validate_training_resume(saved, dict(current, prompt_augmentation=True), state)

    def test_legacy_resume_ignores_removed_cache_latents_field(self):
        from train_flux3_recolor_lora import validate_training_resume, TRAINER, STATE_VERSION
        saved = {'instance_prompt': 'same prompt', 'cache_latents': True}
        current = {'instance_prompt': 'same prompt', 'prompt_augmentation': False, 'training_prompts': ['same prompt']}
        state = {'trainer': TRAINER, 'version': STATE_VERSION}
        validate_training_resume(saved, current, state)

    def test_reference_parameters_and_old_alias_conflicts(self):
        args = self.args('--train_batch_size', '2', '--lr_scheduler', 'cosine', '--lr_warmup_steps', '3',
                         '--lora_layers', 'to_q,to_v')
        self.assertEqual(args.train_batch_size, 2)
        self.assertEqual(args.lr_scheduler, 'cosine')
        self.assertEqual(args.instance_data_dir, str(self.root))
        with self.assertRaises(SystemExit):
            self.args('--dataset_root', 'different')
        from train_flux3_recolor_lora import preprocessing_config
        policy = preprocessing_config(self.args('--use_aspect_ratio_buckets', '--aspect_ratio_buckets', '32,48;48,32'))
        self.assertEqual(policy['buckets'], [(32, 48), (48, 32)])

    def test_bucket_batches_keep_small_bucket_and_all_reference_shapes(self):
        from train_flux3_recolor_lora import BucketBatchSampler, collate_fn
        dataset = make_training_dataset(self.args('--resolution', '64'), [f'{i}.png' for i in range(5)])
        sampler = BucketBatchSampler(dataset, 2, seed=42)
        batches = list(sampler)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(5)))
        self.assertEqual(sorted(len(batch) for batch in batches), [1, 2, 2])
        for indices in batches:
            batch = collate_fn([dataset[i] for i in indices])
            self.assertEqual(batch['pixel_values'].shape[0], len(indices))
            self.assertEqual(len(batch['cond_pixel_values']), 3)

    def test_dataset_default_swatch_resize_is_recorded_in_preprocessing(self):
        from train_flux3_recolor_lora import preprocessing_config
        args = self.args('--resolution', '32')
        self.assertEqual(preprocessing_config(args)['color_swatch_size'], 128)
        dataset = make_training_dataset(args, ['0.png'])
        self.assertEqual(tuple(dataset[0]['cond_images'][2].shape), (3, 128, 128))
        smaller = self.args('--color_swatch_size', '32')
        self.assertEqual(preprocessing_config(smaller)['color_swatch_size'], 32)

    def test_deterministic_epoch_augmentation(self):
        args = self.args('--use_aspect_ratio_buckets', '--aspect_ratio_buckets', '32x48,48x32', '--random_flip')
        dataset = make_training_dataset(args, ['0.png', '1.png'])
        import torch
        torch.testing.assert_close(dataset[0]['instance_images'], dataset[0]['cond_images'][0], rtol=0, atol=0)
        first = dataset[0]['instance_images'].clone()
        dataset.set_epoch(3)
        dataset.set_epoch(0)
        torch.testing.assert_close(first, dataset[0]['instance_images'], rtol=0, atol=0)
        self.assertTrue(self.args('--random_flip').random_flip)

    def test_checkpoint_retention_only_selects_owned_completed_directories(self):
        from train_flux3_recolor_lora import checkpoints_to_remove
        from flux3_recolor.config import write_json
        for name in ('checkpoint-1', 'checkpoint-2', 'checkpoint-999', 'checkpoint-misc'):
            (self.root / name).mkdir()
        for i in (1, 2):
            write_json(self.root / f'checkpoint-{i}/training_state.json',
                       {'trainer': 'flux3-recolor-reference', 'version': 2, 'global_step': i})
        selected = checkpoints_to_remove(self.root, limit=2)
        self.assertEqual([p.name for p in selected], ['checkpoint-1'])
        (self.root / 'checkpoint-998').mkdir()
        write_json(self.root / 'checkpoint-998/training_state.json', ['unrelated'])
        self.assertEqual([p.name for p in checkpoints_to_remove(self.root, limit=2)], ['checkpoint-1'])

    def test_auto_buckets_match_reference_ladder(self):
        from diffusers.training_utils import generate_aspect_ratio_buckets, find_nearest_bucket
        dataset = make_training_dataset(self.args('--resolution', '64', '--use_aspect_ratio_buckets', '--center_crop'), ['0.png'])
        options = generate_aspect_ratio_buckets(64)
        height, width = options[find_nearest_bucket(96, 64, options)]
        self.assertEqual(dataset.buckets[dataset.bucket_idx[0]], (height, width))

    def test_explicit_missing_tracker_is_not_silently_ignored(self):
        from train_flux3_recolor_lora import require_trackers
        self.assertIsNone(require_trackers('none'))
        with self.assertRaises(ValueError):
            require_trackers('not-a-tracker')

    def test_nonconstant_resume_rejects_horizon_change_and_old_state(self):
        from train_flux3_recolor_lora import validate_training_resume
        saved = {'lr_scheduler': 'cosine', 'max_train_steps': 10, 'world_size': 1}
        with self.assertRaisesRegex(ValueError, 'max_train_steps'):
            validate_training_resume(saved, {**saved, 'max_train_steps': 20},
                                     {'trainer': 'flux3-recolor-reference', 'version': 2})
        with self.assertRaisesRegex(ValueError, '旧|version|格式'):
            validate_training_resume(saved, saved, {'global_step': 1})

    def test_latest_missing_directory_is_distinct_from_incompatible_state(self):
        from train_flux3_recolor_lora import resolve_training_checkpoint
        with self.assertRaisesRegex(ValueError, '目录不存在') as caught:
            resolve_training_checkpoint(self.root / 'missing', 'latest')
        self.assertNotIn('旧状态', str(caught.exception))
        with self.assertRaisesRegex(ValueError, '没有checkpoint'):
            resolve_training_checkpoint(self.root, 'latest')
        (self.root / 'checkpoint-1').mkdir()
        with self.assertRaisesRegex(ValueError, 'training_state.json|完成状态'):
            resolve_training_checkpoint(self.root, 'latest')

    def test_latest_valid_state_and_missing_resume_fail_before_loading_inputs(self):
        from train_flux3_recolor_lora import resolve_training_checkpoint
        from flux3_recolor.config import write_json
        import subprocess
        import sys
        checkpoint = self.root / 'checkpoint-2'
        checkpoint.mkdir()
        write_json(checkpoint / 'training_state.json',
                   {'trainer': 'flux3-recolor-reference', 'version': 2, 'global_step': 2})
        self.assertEqual(resolve_training_checkpoint(self.root, 'latest'), checkpoint.resolve())
        missing = self.root / 'never-created'
        result = subprocess.run([sys.executable, 'train_flux3_recolor_lora.py',
            '--pretrained_model_name_or_path', 'unused-model', '--instance_data_dir', 'unused-data',
            '--output_dir', str(missing), '--resume_from_checkpoint', 'latest'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('恢复输出目录不存在', result.stderr)
        self.assertNotIn('缺少本地模型组件', result.stderr)

    def test_single_cuda_without_explicit_index_uses_current_logical_device(self):
        import torch
        from unittest.mock import patch
        from train_flux3_recolor_lora import cuda_device_index
        with patch.object(torch.cuda, 'current_device', return_value=0):
            self.assertEqual(cuda_device_index(torch.device('cuda')), 0)
        self.assertEqual(cuda_device_index(torch.device('cuda:3')), 3)

    def test_custom_lora_targets_and_time_weighting_change_real_loss(self):
        import torch
        from diffusers import Flux3Transformer2DModel, FlowMatchEulerDiscreteScheduler
        from flux3_recolor.modeling import add_lora, flow_matching_loss, text_position_ids
        torch.set_num_threads(1)
        model = Flux3Transformer2DModel(in_channels=8, cond_in_channels=8, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=1, joint_attention_dim=12,
            timestep_channels=8, axes_dims_rope=(4, 4, 4, 4), mlp_ratio=2)
        add_lora(model, 2, 2, 0, target_modules=['to_q'])
        self.assertTrue(all('to_q' in name for name, p in model.named_parameters() if p.requires_grad))
        target, text = torch.randn(2, 8, 2, 2), torch.randn(2, 3, 12)
        refs = [target, target, torch.randn(2, 8, 1, 1)]
        scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False)
        torch.manual_seed(9)
        plain = flow_matching_loss(model, target, refs, text, text_position_ids(3, 'cpu'), scheduler)
        torch.manual_seed(9)
        weighted = flow_matching_loss(model, target, refs, text, text_position_ids(3, 'cpu'), scheduler,
                                      weighting_scheme='sigma_sqrt')
        self.assertGreater(weighted.item(), plain.item())


if __name__ == '__main__':
    unittest.main()
