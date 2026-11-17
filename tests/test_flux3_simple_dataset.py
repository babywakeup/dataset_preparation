from flux3_recolor_test_utils import make_training_dataset
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SimpleDatasetTests(unittest.TestCase):
    def setUp(self):
        from flux3_recolor_test_utils import make_dataset
        from train_flux3_recolor_lora import parse_args
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = make_dataset(Path(self.tmp.name) / 'data')
        self.args = parse_args(['--pretrained_model_name_or_path', 'unused-local-model', '--instance_data_dir', str(self.root),
                                '--resolution', '32', '--report_to', 'none'])

    def test_no_obsolete_dataset_parameters(self):
        self.assertFalse(hasattr(self.args, 'check_dataset'))
        self.assertFalse(hasattr(self.args, 'dataset_validation'))
        self.assertFalse(hasattr(self.args, 'sample_retries'))

    def test_index_only_lists_complete_pairs_without_opening_images(self):
        from flux3_recolor.data import index_dataset
        (self.root / '主颜色B/0.png').unlink()
        with patch('PIL.Image.open', side_effect=AssertionError('must not read images')):
            report = index_dataset(self.root, self.args.training_directions)
        self.assertEqual(report['names'], [f'{i}.png' for i in range(1, 7)])
        with patch('PIL.PngImagePlugin.PngImageFile.load', side_effect=AssertionError('pixel decode')):
            dataset = make_training_dataset(self.args, report['names'])
        self.assertEqual(len(dataset), 12)

    def test_dataset_retries_data_errors_and_returns_actual_identity(self):
        from flux3_recolor.data import SampleDataError, load_rgb
        dataset = make_training_dataset(self.args, ['0.png', '1.png', '2.png'])
        def load(path):
            if Path(path).name == '0.png':
                raise SampleDataError('bad image')
            return load_rgb(path)
        with patch('train_flux3_recolor_lora.load_rgb', side_effect=load), self.assertLogs(level='WARNING'):
            sample = dataset[0]
        self.assertNotEqual(sample['name'], '0.png')
        self.assertNotIn('direction', sample)
        self.assertIn(sample['sample_id'], dataset.sample_ids)
        with patch.object(dataset, 'prepare', side_effect=RuntimeError('bug')):
            with self.assertRaisesRegex(RuntimeError, 'bug'):
                dataset[0]
        with patch.object(dataset, 'prepare', side_effect=SampleDataError('all bad')):
            with self.assertLogs(level='WARNING'), self.assertRaisesRegex(SampleDataError, '读取失败'):
                dataset[0]

    def test_invalid_geometry_is_skipped_and_repeats_do_not_waste_retries(self):
        from PIL import Image
        from flux3_recolor.data import SampleDataError
        Image.new('RGB', (8, 8)).save(self.root / '原图A/0.png')
        with self.assertLogs(level='WARNING'):
            dataset = make_training_dataset(self.args, ['0.png', '1.png', '2.png'])
        self.assertNotIn('0:0.png', list(dataset.records['sample_id']))
        self.args.repeats = 20
        dataset = make_training_dataset(self.args, ['1.png', '2.png', '3.png'])
        with patch.object(dataset, 'prepare', side_effect=SampleDataError('bad')) as reader:
            with self.assertLogs(level='WARNING'), self.assertRaises(SampleDataError):
                dataset[0]
        names = [dataset.records[call.args[0] % dataset.num_instance_images]['sample_id'] for call in reader.call_args_list]
        self.assertEqual(len(names), len(set(names)))


if __name__ == '__main__':
    unittest.main()
