from flux3_recolor_test_utils import make_training_dataset
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class BidirectionalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        colors = {'原图A': (200, 40, 30), '生成图B': (20, 90, 220),
                  '经过涂灰的原图AB': (128, 128, 128), '主颜色A': (200, 40, 30), '主颜色B': (20, 90, 220)}
        for folder, color in colors.items():
            (self.root / folder).mkdir()
            for i in range(7):
                Image.new('RGB', (32, 32), color).save(self.root / folder / f'{i}.png')

    def args(self, *extra):
        from train_flux3_recolor_lora import parse_args
        return parse_args(['--pretrained_model_name_or_path', 'unused-local-model', '--instance_data_dir', str(self.root), '--report_to', 'none',
                           '--resolution', '32', *extra])

    def test_default_dataset_swaps_source_target_and_swatch_without_identity_collisions(self):
        import torch
        from train_flux3_recolor_lora import collate_fn
        dataset = make_training_dataset(self.args('--repeats', '2'), ['0.png', '1.png'])
        self.assertEqual(len(dataset), 8)
        samples = [dataset[i] for i in range(len(dataset))]
        self.assertEqual(len({sample['sample_id'] for sample in samples}), 8)
        for sample in samples:
            a, b = [torch.tensor(rgb).float() / 127.5 - 1 for rgb in ((200, 40, 30), (20, 90, 220))]
            target, source, swatch = ((a, b, a) if sample['sample_id'].startswith('0:') else (b, a, b))
            torch.testing.assert_close(sample['instance_images'][:, 0, 0], target)
            torch.testing.assert_close(sample['cond_images'][1][:, 0, 0], source)
            torch.testing.assert_close(sample['cond_images'][2][:, 0, 0], swatch)
            self.assertEqual(tuple(sample['cond_images'][2].shape), (3, 128, 128))
            self.assertNotIn('direction', sample)
        self.assertEqual(sum(s['sample_id'].startswith('1:') for s in samples), 4)
        self.assertNotIn('directions', collate_fn(samples[:2]))
        single = make_training_dataset(self.args('--training_directions', 'b_to_a'), ['0.png'])
        self.assertEqual(single.sample_ids, ['0:0.png::repeat-0'])

    def test_reverse_swatch_pairing_purity_and_fingerprint_are_checked(self):
        from flux3_recolor.data import inspect_dataset
        report = inspect_dataset(self.root, directions=['b_to_a', 'a_to_b'])
        self.assertEqual(report['count'], 7)
        self.assertEqual(report['training_record_count'], 14)
        Image.new('RGB', (32, 32), 'green').save(self.root / '主颜色B/0.png')
        changed = inspect_dataset(self.root, directions=['b_to_a', 'a_to_b'])
        self.assertNotEqual(changed['fingerprint'], report['fingerprint'])
        Image.linear_gradient('L').resize((32, 32)).convert('RGB').save(self.root / '主颜色B/0.png')
        with self.assertRaisesRegex(ValueError, '主颜色B.*纯色'):
            inspect_dataset(self.root, directions=['b_to_a', 'a_to_b'])
        (self.root / '主颜色B/0.png').unlink()
        with self.assertRaisesRegex(ValueError, '主颜色B.*0.png'):
            inspect_dataset(self.root, directions=['b_to_a', 'a_to_b'])
        # Legacy forward-only data inspection does not require B swatches.
        self.assertEqual(inspect_dataset(self.root)['count'], 7)

    def test_source_split_happens_before_direction_expansion(self):
        from flux3_recolor.data import split_samples
        names = [f'{i}.png' for i in range(7)]
        split = split_samples(names, 0.3, 42, {name: str(i // 2) for i, name in enumerate(names)})
        training = make_training_dataset(self.args(), split['train'])
        validation = make_training_dataset(self.args(), split['validation'])
        train_names = {training[i]['name'] for i in range(len(training))}
        val_names = {validation[i]['name'] for i in range(len(validation))}
        self.assertFalse(train_names & val_names)
        self.assertEqual(len(training), len(split['train']) * 2)
        self.assertEqual(len(validation), len(split['validation']) * 2)

    def test_legacy_resume_requires_single_direction_and_new_specs_are_strict(self):
        from train_flux3_recolor_lora import validate_training_resume, TRAINER, STATE_VERSION
        from flux3_recolor import subdataset_config
        state = {'trainer': TRAINER, 'version': STATE_VERSION}
        saved = {'instance_prompt': 'same', 'prompt_augmentation': False, 'training_prompts': ['same']}
        current = {**saved, 'training_directions': ['b_to_a'], 'subdatasets': subdataset_config(['b_to_a'])}
        validate_training_resume(saved, current, state)
        with self.assertRaisesRegex(ValueError, '恢复配置不一致'):
            validate_training_resume(saved, {**current, 'training_directions': ['b_to_a', 'a_to_b'],
                                             'subdatasets': subdataset_config(['b_to_a', 'a_to_b'])}, state)

    def test_per_sample_loss_preserves_distinct_direction_errors(self):
        import torch
        from unittest.mock import patch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from flux3_recolor.modeling import flow_matching_loss, text_position_ids
        target = torch.zeros(2, 2, 2, 2)
        noise = torch.ones_like(target)
        def model(**kwargs):
            return ((noise - target).flatten(2).transpose(1, 2) + torch.tensor([1., 3.])[:, None, None],)
        with patch('torch.randn_like', return_value=noise):
            losses = flow_matching_loss(model, target, [target, target, target], torch.zeros(2, 3, 4),
                                        text_position_ids(3, 'cpu'),
                                        FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False),
                                        return_per_sample=True)
        torch.testing.assert_close(losses, torch.tensor([1., 9.]))


if __name__ == '__main__':
    unittest.main()
