import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


class DatasetCacheTests(unittest.TestCase):
    def setUp(self):
        from flux3_recolor_test_utils import make_dataset
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = make_dataset(self.root / 'images')
        self.cache = self.root / 'cache'
        from flux3_recolor import DEFAULT_PROMPT
        self.config = {'root': str(self.images), 'directions': ['b_to_a', 'a_to_b'],
            'validation_fraction': 0.1, 'seed': 42, 'prompt': DEFAULT_PROMPT,
            'captions': None, 'source_groups': None, 'sample_names': [f'{i}.png' for i in range(7)]}

    def test_key_preserves_original_manifest_schema(self):
        from flux3_recolor.dataset_cache import dataset_cache_key
        from flux3_recolor import subdataset_config
        _, manifest = dataset_cache_key(self.config)
        self.assertEqual(set(manifest), {'format_version', 'config', 'subdatasets', 'files'})
        self.assertEqual(manifest['format_version'], 2)
        self.assertEqual(manifest['subdatasets'], subdataset_config(self.config['directions']))
        self.assertTrue(all(isinstance(row, list) and len(row) == 4 for row in manifest['files']))
        self.assertEqual(manifest['files'], sorted(manifest['files']))

    def test_hit_checks_manifest_and_dataset_readability(self):
        import json
        from flux3_recolor.dataset_cache import prepare_cached_dataset
        path = Path(prepare_cached_dataset(self.config, self.cache))
        manifest_file = path / 'cache_config.json'
        original = manifest_file.read_text()
        manifest_file.write_text(json.dumps({'wrong': True}))
        with self.assertRaisesRegex(ValueError, '缓存配置不匹配'):
            prepare_cached_dataset(self.config, self.cache)
        manifest_file.write_text(original)
        with patch('datasets.load_from_disk', side_effect=RuntimeError('broken saved dataset')):
            with self.assertRaisesRegex(RuntimeError, 'broken saved dataset'):
                prepare_cached_dataset(self.config, self.cache)

    def test_hit_has_hash_name_and_never_builds_or_opens_images(self):
        from flux3_recolor.dataset_cache import prepare_cached_dataset
        path = prepare_cached_dataset(self.config, self.cache)
        self.assertRegex(Path(path).name, r'^[0-9a-f]{64}$')
        with patch('PIL.Image.open', side_effect=AssertionError('cache hit must not open images')), \
             patch('build_flux3_recolor_dataset.build_dataset', side_effect=AssertionError('cache hit must not build')):
            self.assertEqual(prepare_cached_dataset(self.config, self.cache, num_proc=4), path)
        self.assertTrue((Path(path) / 'cache_config.json').is_file())

    def test_key_tracks_content_inputs_and_source_stat(self):
        from flux3_recolor.dataset_cache import dataset_cache_key
        key = dataset_cache_key(self.config)[0]
        self.assertEqual(dataset_cache_key(dict(reversed(list(self.config.items()))))[0], key)
        for change in [{'seed': 8}, {'validation_fraction': 0.3}, {'prompt': 'different'},
                       {'directions': ['a_to_b']}, {'mix_ratio': [2, 1]},
                       {'captions': {'0.png': 'new caption'}},
                       {'source_groups': {f'{i}.png': str(i // 2) for i in range(7)}},
                       {'sample_names': ['0.png', '1.png', '2.png']}]:
            self.assertNotEqual(dataset_cache_key({**self.config, **change})[0], key)
        path = self.images / '原图A/0.png'
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        self.assertNotEqual(dataset_cache_key(self.config)[0], key)

    def test_cross_training_runs_ignore_training_only_arguments(self):
        from train_flux3_recolor_lora import parse_args, prepare_path_dataset
        args = parse_args(['--pretrained_model_name_or_path', 'unused-model',
                           '--instance_data_dir', str(self.images), '--output_dir', str(self.root / 'run1')])
        path = prepare_path_dataset(args)
        args.output_dir = str(self.root / 'run2')
        args.learning_rate = 0.002
        args.rank = 32
        args.resolution = 1024
        args.dataloader_num_workers = 4
        with patch('build_flux3_recolor_dataset.build_dataset', side_effect=AssertionError('should share cache')):
            self.assertEqual(prepare_path_dataset(args), path)
        self.assertNotIn('run1', path)
        self.assertNotIn('run2', path)

    def test_concurrent_callers_build_once(self):
        from flux3_recolor.dataset_cache import prepare_cached_dataset
        from build_flux3_recolor_dataset import build_dataset
        with patch('build_flux3_recolor_dataset.build_dataset', wraps=build_dataset) as build:
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(lambda _: prepare_cached_dataset(self.config, self.cache), range(2)))
        self.assertEqual(build.call_count, 1)
        self.assertEqual(paths[0], paths[1])

    def test_failed_build_does_not_publish_and_can_retry(self):
        from flux3_recolor.dataset_cache import prepare_cached_dataset, dataset_cache_key
        def fail(output, **kwargs):
            Path(output).mkdir()
            (Path(output) / 'partial').write_text('incomplete')
            raise RuntimeError('interrupted build')
        with patch('build_flux3_recolor_dataset.build_dataset', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                prepare_cached_dataset(self.config, self.cache)
        key = dataset_cache_key(self.config)[0]
        self.assertFalse((self.cache / key).exists())
        path = prepare_cached_dataset(self.config, self.cache)
        self.assertTrue((Path(path) / 'dataset_dict.json').is_file())


if __name__ == '__main__':
    unittest.main()
