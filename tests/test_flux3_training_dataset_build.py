import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TrainingDatasetBuildTests(unittest.TestCase):
    def setUp(self):
        from flux3_recolor_test_utils import make_dataset
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = make_dataset(self.root / 'images')

    def args(self, *extra):
        from train_flux3_recolor_lora import parse_args
        return parse_args(['--pretrained_model_name_or_path', 'unused-local-model', '--instance_data_dir', str(self.images), '--output_dir', str(self.root / 'run'),
            '--mixed_precision', 'no', '--report_to', 'none', '--resolution', '32', '--max_train_steps', '1',
            '--num_validation_images', '0', *extra])

    def test_constructor_is_called_by_training_before_loading_models(self):
        from train_flux3_recolor_lora import main, prepare_path_dataset
        from datasets import load_from_disk
        from build_flux3_recolor_dataset import build_dataset
        model = self.root / 'model'; model.mkdir()
        with patch('train_flux3_recolor_lora.validate_local_model', return_value=(str(model), 'test-signature')), \
             patch('train_flux3_recolor_lora.load_training_components', side_effect=RuntimeError('stop before models')), \
             patch('build_flux3_recolor_dataset.build_dataset', wraps=build_dataset) as build:
            with self.assertRaisesRegex(RuntimeError, 'stop before models'):
                main(self.args('--pretrained_model_name_or_path', str(model)))
        self.assertEqual(build.call_count, 1)
        path = prepare_path_dataset(self.args())
        dataset = load_from_disk(path)
        self.assertRegex(Path(path).name, r'^[0-9a-f]{64}$')
        self.assertEqual(sum(len(rows) for rows in dataset.values()), 14)
        self.assertNotIn('direction', dataset['train'].column_names)

    def test_saved_dataset_is_reused_and_preserves_training_prompt_captions_and_groups(self):
        from train_flux3_recolor_lora import prepare_path_dataset
        from datasets import load_from_disk
        groups = {f'{i}.png': ('shared' if i < 2 else str(i)) for i in range(7)}
        captions = {f'{i}.png': f'caption {i}' for i in range(7)}
        (self.root / 'groups.json').write_text(json.dumps(groups))
        (self.root / 'captions.json').write_text(json.dumps(captions))
        args = self.args('--groups_json', str(self.root / 'groups.json'), '--caption_json', str(self.root / 'captions.json'))
        path = prepare_path_dataset(args)
        with patch('build_flux3_recolor_dataset.build_dataset', side_effect=AssertionError('must reuse')):
            self.assertEqual(prepare_path_dataset(args), path)
        dataset = load_from_disk(path)
        for rows in dataset.values():
            for row in rows:
                name = Path(row['target_path']).name
                self.assertEqual(row['prompt'], captions[name])
                self.assertEqual(row['source_id'], groups[name])
        self.assertTrue(set(dataset['train']['source_id']).isdisjoint(dataset['validation']['source_id']))

    def test_independently_valid_records_are_kept_with_group_mapping(self):
        from train_flux3_recolor_lora import prepare_path_dataset
        from datasets import load_from_disk
        (self.images / '主颜色B/0.png').unlink()
        groups = {f'{i}.png': str(i) for i in range(7)}
        group_file = self.root / 'groups.json'
        group_file.write_text(json.dumps(groups))
        args = self.args('--groups_json', str(group_file))
        with self.assertLogs(level='WARNING'):
            path = prepare_path_dataset(args)
        dataset = load_from_disk(path)
        self.assertEqual(sum(len(rows) for rows in dataset.values()), 14)
        self.assertIn('0', {source for rows in dataset.values() for source in rows['source_id']})
        ids = {sample for rows in dataset.values() for sample in rows['sample_id']}
        self.assertIn('0:0.png', ids)
        self.assertNotIn('1:0.png', ids)


if __name__ == '__main__':
    unittest.main()
