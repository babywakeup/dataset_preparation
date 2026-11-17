import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


class CachedConsumptionTests(unittest.TestCase):
    def setUp(self):
        from flux3_recolor_test_utils import make_dataset
        from train_flux3_recolor_lora import parse_args
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = make_dataset(self.root / 'images')
        self.args = parse_args(['--pretrained_model_name_or_path', 'unused-model',
            '--instance_data_dir', str(self.images), '--output_dir', str(self.root / 'run'),
            '--resolution', '32', '--mixed_precision', 'no', '--report_to', 'none'])

    def prepared(self):
        from build_flux3_recolor_dataset import build_dataset
        return build_dataset(self.images, self.root / 'prepared')

    def test_dataset_and_sampler_initialize_from_metadata_without_opening_images(self):
        from train_flux3_recolor_lora import DreamBoothDataset, BucketBatchSampler, collate_fn
        prepared = self.prepared()
        with patch('PIL.Image.open', side_effect=AssertionError('must use cached dimensions')):
            dataset = DreamBoothDataset(self.args, prepared['train'])
            sampler = BucketBatchSampler(dataset, 2)
            batches = sampler.batches()
        self.assertEqual(len(dataset), len(prepared['train']))
        self.assertTrue(dataset.buckets)
        for batch in batches:
            self.assertEqual(len({dataset.bucket_idx[i] for i in batch}), 1)
        sample = dataset[0]
        self.assertIn('bucket_idx', sample)
        self.assertNotIn('direction', sample)
        self.assertNotIn('directions', collate_fn([sample]))
        self.assertEqual(tuple(sample['cond_images'][2].shape), (3, 128, 128))

    def test_empty_caption_falls_back_to_instance_prompt(self):
        import json
        from datasets import Dataset
        from train_flux3_recolor_lora import DreamBoothDataset
        row = self.prepared()['train'][0]
        row['prompt'] = ''
        caption_file = self.root / 'captions.json'
        caption_file.write_text(json.dumps({Path(row['target_path']).name: ''}))
        self.args.caption_json = str(caption_file)
        dataset = DreamBoothDataset(self.args, Dataset.from_list([row]))
        self.assertEqual(dataset[0]['instance_prompt'], self.args.instance_prompt)

    def test_main_only_consumes_cached_records_before_model_loading(self):
        from train_flux3_recolor_lora import main, prepare_path_dataset
        path = prepare_path_dataset(self.args)
        model = self.root / 'model'
        model.mkdir()
        with patch('train_flux3_recolor_lora.prepare_path_dataset', return_value=path), \
             patch('flux3_recolor.data.index_dataset', side_effect=AssertionError('no scan')), \
             patch('flux3_recolor.data.split_samples', side_effect=AssertionError('no split')), \
             patch('PIL.Image.open', side_effect=AssertionError('no image open')), \
             patch('train_flux3_recolor_lora.validate_local_model', return_value=(str(model), 'signature')), \
             patch('train_flux3_recolor_lora.load_training_components', side_effect=RuntimeError('stop before models')):
            with self.assertRaisesRegex(RuntimeError, 'stop before models'):
                main(self.args)

    def test_shared_random_geometry_keeps_all_scenes_aligned(self):
        import torch
        from datasets import Dataset
        from build_flux3_recolor_dataset import generate_samples
        from train_flux3_recolor_lora import DreamBoothDataset
        for folder in ['原图A', '生成图B', '经过涂灰的原图AB']:
            Image.linear_gradient('L').resize((64, 80)).convert('RGB').save(self.images / folder / '0.png')
        rows = list(generate_samples(self.images, 'b_to_a', ['0.png'], 0))
        self.args.use_aspect_ratio_buckets = True
        self.args.aspect_ratio_buckets = '32,32'
        self.args.random_flip = True
        dataset = DreamBoothDataset(self.args, Dataset.from_list(rows))
        for epoch in range(4):
            dataset.set_epoch(epoch)
            row = dataset[0]
            torch.testing.assert_close(row['instance_images'], row['cond_images'][0], rtol=0, atol=0)
            torch.testing.assert_close(row['instance_images'], row['cond_images'][1], rtol=0, atol=0)

    def test_validation_uses_saved_paths_and_has_no_direction_expansion(self):
        from flux3_recolor.inference import generate_validation
        from train_flux3_recolor_lora import preprocessing_config
        prepared = self.prepared()
        selected = prepared['validation'].select([0])
        output = self.root / 'validation'
        config = {'max_area': 1024, 'preprocessing': preprocessing_config(self.args), 'prompt': 'recolor'}
        with patch('flux3_recolor.inference.generate_sample', side_effect=lambda pipe, refs, *args: (refs[0], refs)) as generate:
            generate_validation(None, self.images, selected, config, output, 42, 1, None)
        self.assertEqual(generate.call_count, 1)
        import json
        samples = json.loads((output / 'manifest.json').read_text())['samples']
        self.assertEqual(samples[0]['sample_id'], selected[0]['sample_id'])
        self.assertNotIn('direction', samples[0])
        self.assertTrue((output / samples[0]['path'] / 'condition_3.png').is_file())

    def test_validation_path_reader_defaults_to_center_crop(self):
        from flux3_recolor.data import read_path_sample
        row = self.prepared()['validation'][0]
        for path in [row['target_path'], *row['condition_paths'][:2]]:
            Image.linear_gradient('L').resize((64, 96)).convert('RGB').save(path)
        policy = {'max_area': 1024, 'color_swatch_size': 128,
                  'use_aspect_ratio_buckets': True, 'buckets': [(32, 32)]}
        actual = read_path_sample(row, policy)
        expected = read_path_sample(row, policy, center_crop=True)
        for a, b in zip([actual[0], *actual[1]], [expected[0], *expected[1]]):
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_validation_selection_spans_unified_rows(self):
        from datasets import Dataset
        from flux3_recolor.inference import select_validation_records
        records = Dataset.from_dict({'sample_id': [str(i) for i in range(20)]})
        selected = select_validation_records(records, 4)
        self.assertEqual(list(selected['sample_id']), ['0', '6', '12', '19'])
        self.assertEqual(len(select_validation_records(records, 100)), 20)

    def test_independent_subsets_are_not_intersected_again_by_trainer(self):
        from datasets import load_from_disk
        from train_flux3_recolor_lora import prepare_path_dataset
        (self.images / '主颜色B/0.png').unlink()
        with self.assertLogs(level='WARNING'):
            path = prepare_path_dataset(self.args)
        data = load_from_disk(path)
        ids = {sid for rows in data.values() for sid in rows['sample_id']}
        self.assertIn('0:0.png', ids)
        self.assertNotIn('1:0.png', ids)


if __name__ == '__main__':
    unittest.main()
