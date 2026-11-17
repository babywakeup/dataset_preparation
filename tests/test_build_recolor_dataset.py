import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


class BuildRecolorDatasetTests(unittest.TestCase):
    def setUp(self):
        from flux3_recolor_test_utils import make_dataset
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = make_dataset(self.root / 'images')

    def test_only_paths_and_independent_sizes_are_saved_without_pixel_decoding(self):
        from build_flux3_recolor_dataset import build_dataset
        from datasets import load_from_disk
        Image.new('RGB', (48, 64), (255, 128, 0)).save(self.images / '原图A/0.png')
        Image.new('RGB', (64, 48), (7, 200, 255)).save(self.images / '生成图B/0.png')
        output = self.root / 'prepared'
        with patch('PIL.PngImagePlugin.PngImageFile.load', side_effect=AssertionError('pixel decode')):
            build_dataset(self.images, output, seed=1)
        with patch('PIL.Image.open', side_effect=AssertionError('should only read metadata')):
            dataset = load_from_disk(str(output))
            rows = [row for split in dataset.values() for row in split if row['source_id'] == '0']
        self.assertEqual(len(rows), 2)
        self.assertEqual({(r['height'], r['width']) for r in rows}, {(64, 48), (48, 64)})
        self.assertEqual(dataset['train'].features['target_path'].dtype, 'string')
        for row in rows:
            self.assertTrue(Path(row['target_path']).is_file())
            self.assertEqual(len(row['condition_paths']), 3)
            self.assertEqual(len(row['condition_sizes']), 3)
            for path, size in zip(row['condition_paths'], row['condition_sizes']):
                self.assertTrue(Path(path).is_file())
                with Image.open(path) as image:
                    self.assertEqual(size, {'height': image.height, 'width': image.width})
            self.assertNotIn('direction', row)
            self.assertNotIn('target', row)
            self.assertNotIn('conditions', row)
        a = next(r for r in rows if r['height'] == 64)
        self.assertEqual(Path(a['condition_paths'][1]), self.images / '生成图B/0.png')

    def test_late_png_exif_orientation_is_reflected_without_decoding(self):
        import struct
        from build_flux3_recolor_dataset import build_dataset
        image = Image.new('RGB', (48, 64))
        exif = Image.Exif()
        exif[274] = 6
        path = self.images / '原图A/0.png'
        image.save(path, exif=exif)
        data = path.read_bytes()
        chunks, offset = [], 8
        while offset < len(data):
            length = struct.unpack('>I', data[offset:offset+4])[0]
            chunks.append(data[offset:offset+length+12])
            offset += length + 12
        exif_chunk = next(chunk for chunk in chunks if chunk[4:8] == b'eXIf')
        chunks = [chunk for chunk in chunks if chunk[4:8] != b'eXIf']
        path.write_bytes(data[:8] + b''.join(chunks[:-1] + [exif_chunk, chunks[-1]]))
        with patch('PIL.PngImagePlugin.PngImageFile.load', side_effect=AssertionError('pixel decode')):
            dataset = build_dataset(self.images, self.root / 'oriented')
        row = next(row for split in dataset.values() for row in split if row['sample_id'] == '0:0.png')
        self.assertEqual((row['height'], row['width']), (48, 64))
        # Original A also appears as source condition in the other subset.
        reverse = next(row for split in dataset.values() for row in split if row['sample_id'] == '1:0.png')
        self.assertEqual(reverse['condition_sizes'][1], {'height': 48, 'width': 64})

    def test_concat_ratio_and_group_split(self):
        from build_flux3_recolor_dataset import build_dataset
        dataset = build_dataset(self.images, self.root / 'weighted', mix_ratio=(2, 1), seed=7)
        counts = {}
        for split in dataset.values():
            for sample in split['sample_id']:
                subset = sample.split(':')[0]
                counts[subset] = counts.get(subset, 0) + 1
        self.assertEqual(counts, {'0': 14, '1': 7})
        self.assertTrue(set(dataset['train']['source_id']).isdisjoint(dataset['validation']['source_id']))
        self.assertEqual(set(dataset['train']['source_id']) | set(dataset['validation']['source_id']),
                         {str(i) for i in range(7)})

    def test_subsets_filter_missing_and_invalid_headers_independently_and_refuse_overwrite(self):
        from build_flux3_recolor_dataset import build_dataset
        (self.images / '主颜色B/0.png').unlink()
        (self.images / '主颜色A/1.png').write_bytes(b'bad image')
        with self.assertLogs(level='WARNING'):
            dataset = build_dataset(self.images, self.root / 'filtered', seed=3)
        rows = [row for split in dataset.values() for row in split]
        ids = {row['sample_id'] for row in rows}
        self.assertIn('0:0.png', ids)
        self.assertNotIn('1:0.png', ids)
        self.assertNotIn('0:1.png', ids)
        self.assertIn('1:1.png', ids)
        with self.assertRaisesRegex(ValueError, '存在|非空'):
            build_dataset(self.images, self.root / 'filtered')

    def test_cache_memory_maps_released_before_shared_filesystem_cleanup(self):
        from build_flux3_recolor_dataset import build_dataset
        parent = Path.cwd() / 'datasets'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.recolor-test-', dir=parent) as directory:
            directory = Path(directory)
            dataset = build_dataset(self.images, directory / 'prepared')
            self.assertEqual(sum(len(rows) for rows in dataset.values()), 14)
            self.assertFalse(list(directory.glob('.recolor-build-*')))
            # The returned Dataset also memory-maps the final output on NFS.
            del dataset
            import gc
            gc.collect()


if __name__ == '__main__':
    unittest.main()
