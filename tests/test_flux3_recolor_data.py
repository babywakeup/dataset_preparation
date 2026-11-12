import tempfile
import unittest
from pathlib import Path

from PIL import Image


class RecolorDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folders = ['原图A', '经过涂灰的原图AB', '生成图B', '主颜色A']
        for folder in self.folders:
            (self.root / folder).mkdir()
            for i in range(10):
                size = (32, 32) if folder == '主颜色A' else (64, 96)
                Image.new('RGB', size, (128, i, 32)).save(self.root / folder / f'{i:03}.png')

    def test_inspect_decodes_all_pairs_and_fingerprints_pixels(self):
        from flux3_recolor.data import inspect_dataset
        report = inspect_dataset(self.root)
        self.assertEqual(report['count'], 10)
        self.assertEqual(report['sizes']['原图A'], {'64x96': 10})
        before = report['fingerprint']
        Image.new('RGB', (32, 32), 'blue').save(self.root / '主颜色A/009.png')
        self.assertNotEqual(inspect_dataset(self.root)['fingerprint'], before)

    def test_missing_pair_is_not_silently_dropped(self):
        from flux3_recolor.data import inspect_dataset
        (self.root / '生成图B/009.png').unlink()
        with self.assertRaisesRegex(ValueError, '009.png'):
            inspect_dataset(self.root)

    def test_corrupt_image_names_the_file(self):
        from flux3_recolor.data import inspect_dataset
        (self.root / '主颜色A/009.png').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, '009.png'):
            inspect_dataset(self.root)

    def test_scene_size_mismatch_is_rejected(self):
        from flux3_recolor.data import inspect_dataset
        Image.new('RGB', (64, 80)).save(self.root / '生成图B/009.png')
        with self.assertRaisesRegex(ValueError, '009.png'):
            inspect_dataset(self.root)

    def test_split_is_stable_disjoint_and_groups_do_not_leak(self):
        from flux3_recolor.data import split_samples
        names = [f'{i:03}.png' for i in range(10)]
        groups = {name: str(i // 2) for i, name in enumerate(names)}
        split = split_samples(names, 0.2, 42, groups)
        self.assertEqual(split, split_samples(list(reversed(names)), 0.2, 42, groups))
        self.assertEqual(len(split['validation']), 2)
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertEqual(set(split['train'] + split['validation']), set(names))
        self.assertFalse({groups[n] for n in split['train']} & {groups[n] for n in split['validation']})
        with self.assertRaises(ValueError):
            split_samples(['one.png'], 0.1, 42)
        with self.assertRaises(ValueError):
            split_samples(names, 0.1, 42, {'000.png': 'only'})

    def test_shared_geometry_and_swatch_color_preserved(self):
        from flux3_recolor.data import prepare_images
        scene = Image.linear_gradient('L').resize((64, 96)).convert('RGB')
        swatch = Image.new('RGB', (32, 32), (220, 20, 80))
        target, conditions = prepare_images(scene, [scene.copy(), scene.copy(), swatch], 2048)
        self.assertEqual(target.size, (32, 48))
        self.assertEqual(target.tobytes(), conditions[0].tobytes())
        self.assertEqual(target.tobytes(), conditions[1].tobytes())
        self.assertEqual(conditions[2].size, (32, 32))
        self.assertEqual(conditions[2].getpixel((0, 0)), (220, 20, 80))
        again, refs = prepare_images(target, conditions, 2048)
        self.assertEqual(again.tobytes(), target.tobytes())
        self.assertEqual([x.tobytes() for x in refs], [x.tobytes() for x in conditions])

    def test_exif_orientation_and_rgb_are_applied_before_geometry_check(self):
        from flux3_recolor.data import load_rgb
        path = self.root / 'oriented.png'
        image = Image.new('L', (48, 64), 77)
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
        actual = load_rgb(path)
        self.assertEqual(actual.mode, 'RGB')
        self.assertEqual(actual.size, (64, 48))

    def test_check_cli_needs_no_model_and_persists_split(self):
        import json
        import subprocess
        import sys
        output = self.root / 'reports'
        result = subprocess.run(
            [sys.executable, 'train_flux3_recolor_lora.py', '--check_dataset',
             '--dataset_root', str(self.root), '--output_dir', str(output)],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        split = json.loads((output / 'split.json').read_text())
        self.assertEqual(len(split['train']), 9)
        self.assertEqual(len(split['validation']), 1)

    def test_output_protection_rejects_nonempty_and_input_overlap(self):
        from flux3_recolor.config import prepare_output
        with self.assertRaises(ValueError):
            prepare_output(self.root, protected=[self.root / '原图A'])
        with self.assertRaises(ValueError):
            prepare_output(self.root / '原图A/new', protected=[self.root])
        with self.assertRaises(ValueError):
            prepare_output(self.root / '原图A')

    def test_invalid_effective_geometry_names_sample(self):
        from flux3_recolor.data import load_sample
        with self.assertRaisesRegex(ValueError, '009.png'):
            load_sample(self.root, '009.png', 256)


if __name__ == '__main__':
    unittest.main()
