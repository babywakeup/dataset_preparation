import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from object_inpainting.config import PipelineConfig
from object_inpainting.exporting import choose_split
from object_inpainting.geometry import align_square
from object_inpainting.masking import expand_mask, make_source
from object_inpainting.pipeline import build_dataset, prepare_sample
from object_inpainting.prompting import build_prompt
from object_inpainting.quality import PreparedSample, validate_sample
from object_inpainting.records import load_images, read_records
from object_inpainting.reference import extract_reference
from object_inpainting.selection import select_object
from PIL import Image, ImageDraw


class ObjectInpaintingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = Image.new("RGB", (96, 64), (20, 40, 60))
        ImageDraw.Draw(self.target).rectangle((55, 20, 74, 39), fill=(220, 30, 40))
        self.mask = Image.new("L", self.target.size, 0)
        ImageDraw.Draw(self.mask).rectangle((55, 20, 74, 39), fill=255)
        self.target.save(self.root / "target.png")
        self.mask.save(self.root / "mask.png")
        self.manifest = self.root / "input.jsonl"
        self.write_records([{"target": "target.png", "mask": "mask.png", "object_name": "cup"}])
        self.config = PipelineConfig(resolution=64, reference_resolution=32, dilation_radius=2, validation_fraction=0)

    def write_records(self, rows):
        with self.manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_config_rejects_invalid_values(self):
        for kwargs in (
            {"resolution": 63},
            {"reference_resolution": 0},
            {"dilation_radius": -1},
            {"reference_padding": -0.1},
            {"min_object_fraction": 0.7},
            {"validation_fraction": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig(**kwargs)

    def test_records_resolve_paths_and_reject_duplicates(self):
        record = next(read_records(self.manifest))
        self.assertEqual(record.target_path, self.root / "target.png")
        self.assertEqual(record.mask_path, self.root / "mask.png")
        self.assertEqual(len(record.sample_id), 24)
        row = {"target": "target.png", "mask": "mask.png", "object_name": "cup"}
        self.write_records([row, row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            list(read_records(self.manifest))

    def test_loading_binarizes_mask(self):
        soft_mask = Image.new("L", self.target.size, 127)
        soft_mask.putpixel((60, 30), 128)
        soft_mask.save(self.root / "mask.png")
        _, mask = load_images(next(read_records(self.manifest)))
        self.assertEqual(mask.getpixel((0, 0)), 0)
        self.assertEqual(mask.getpixel((60, 30)), 255)

    def test_loading_rejects_misaligned_dimensions(self):
        Image.new("L", (32, 32), 255).save(self.root / "mask.png")
        with self.assertRaisesRegex(ValueError, "aligned dimensions"):
            load_images(next(read_records(self.manifest)))

    def test_selection_rejects_empty_tiny_and_huge_masks(self):
        for mask in (Image.new("L", (64, 64), 0), Image.new("L", (64, 64), 255)):
            with self.assertRaises(ValueError):
                select_object(mask, self.config)
        tiny = Image.new("L", (64, 64), 0)
        tiny.putpixel((10, 10), 255)
        with self.assertRaisesRegex(ValueError, "area fraction"):
            select_object(tiny, self.config)

    def test_dilation_covers_object(self):
        expanded = expand_mask(self.mask, 2)
        self.assertEqual(expanded.getbbox(), (53, 18, 77, 42))
        self.assertTrue(np.all(np.asarray(expanded)[np.asarray(self.mask) > 127] == 255))
        self.assertEqual(expand_mask(self.mask, 0).tobytes(), self.mask.tobytes())

    def test_square_crop_retains_edge_object_and_binary_masks(self):
        edit_mask = expand_mask(self.mask, 2)
        target, object_mask, edit_mask = align_square(self.target, self.mask, edit_mask, 64)
        self.assertEqual(target.size, (64, 64))
        self.assertEqual((np.asarray(object_mask) > 127).sum(), 400)
        self.assertEqual((np.asarray(edit_mask) > 127).sum(), 576)
        self.assertTrue(np.isin(np.asarray(edit_mask), [0, 255]).all())

    def test_square_crop_rejects_truncation(self):
        wide_mask = Image.new("L", self.target.size, 0)
        ImageDraw.Draw(wide_mask).rectangle((4, 20, 90, 30), fill=255)
        with self.assertRaisesRegex(ValueError, "truncation"):
            align_square(self.target, wide_mask, wide_mask, 64)

    def test_reference_removes_background_and_keeps_object(self):
        reference = extract_reference(self.target, self.mask, self.mask.getbbox(), 32, 0.2)
        self.assertEqual(reference.size, (32, 32))
        self.assertEqual(reference.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(reference.getpixel((16, 16)), (220, 30, 40))

    def test_masking_preserves_exterior_exactly(self):
        source = make_source(self.target, self.mask)
        editable = np.asarray(self.mask) > 127
        self.assertTrue(np.array_equal(np.asarray(source)[~editable], np.asarray(self.target)[~editable]))
        self.assertTrue((np.asarray(source)[editable] == 128).all())

    def test_prompt_mentions_object_and_image_roles(self):
        prompt = build_prompt("cup")
        for phrase in ("cup", "first image", "second image", "unchanged"):
            self.assertIn(phrase, prompt)

    def test_quality_rejects_changed_exterior(self):
        sample = prepare_sample(next(read_records(self.manifest)), self.config)
        source = sample.source.copy()
        source.putpixel((0, 0), (0, 0, 0))
        broken = PreparedSample(sample.target, source, sample.mask, sample.reference, sample.prompt)
        with self.assertRaisesRegex(ValueError, "outside the edit mask"):
            validate_sample(broken, sample.mask, self.config)

    def test_split_is_deterministic_and_grouped_by_original_image(self):
        path = self.root / "target.png"
        expected = choose_split(path, 0.5, 42)
        for _ in range(10):
            self.assertEqual(choose_split(path, 0.5, 42), expected)
        self.assertEqual(choose_split(path, 0, 42), "train")

    def test_end_to_end_exports_training_schema_and_preview(self):
        output = self.root / "dataset"
        summary = build_dataset(self.manifest, output, self.config)
        self.assertEqual(summary["counts"], {"train": 1, "validation": 0, "rejected": 0})
        entry = json.loads((output / "train.jsonl").read_text())
        self.assertEqual(set(entry), {"target", "source", "mask", "references", "prompt"})
        self.assertEqual(len(entry["references"]), 1)
        for name in ("source", "target", "mask"):
            with Image.open(output / entry[name]) as image:
                self.assertEqual(image.size, (64, 64))
        with Image.open(output / entry["references"][0]) as reference:
            self.assertEqual(reference.size, (32, 32))
        self.assertEqual(len(list((output / "previews").glob("*.png"))), 1)
        self.assertTrue((output / "summary.json").is_file())
        self.assertEqual((output / "validation.jsonl").read_text(), "")
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            build_dataset(self.manifest, output, self.config)

    def test_missing_image_is_reported_and_valid_sample_survives(self):
        self.write_records(
            [
                {"target": "missing.png", "mask": "mask.png", "object_name": "cup"},
                {"target": "target.png", "mask": "mask.png", "object_name": "cup"},
            ]
        )
        output = self.root / "dataset"
        summary = build_dataset(self.manifest, output, self.config, preview_count=0)
        self.assertEqual(summary["counts"], {"train": 1, "validation": 0, "rejected": 1})
        rejection = json.loads((output / "rejected.jsonl").read_text())
        self.assertIn("missing.png", rejection["target"])
        self.assertFalse((output / "previews").exists())

    def test_no_usable_samples_fails_with_report(self):
        Image.new("L", self.target.size, 0).save(self.root / "mask.png")
        output = self.root / "dataset"
        with self.assertRaisesRegex(ValueError, "No usable samples"):
            build_dataset(self.manifest, output, self.config)
        self.assertEqual(json.loads((output / "rejected.jsonl").read_text())["reason"], "Instance mask is empty.")

    def test_cli_creates_dataset(self):
        entrypoint = Path(__file__).resolve().parents[1] / "build_object_inpainting_dataset.py"
        result = subprocess.run(
            [
                sys.executable,
                str(entrypoint),
                "--input-manifest",
                str(self.manifest),
                "--output-dir",
                str(self.root / "cli_dataset"),
                "--resolution",
                "64",
                "--reference-resolution",
                "32",
                "--validation-fraction",
                "0",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["counts"]["train"], 1)


if __name__ == "__main__":
    unittest.main()
