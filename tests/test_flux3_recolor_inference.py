import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


class RecolorInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_preprocess_matches_real_flux3_multireference_processor(self):
        import numpy as np
        from flux3_recolor.data import prepare_images, image_tensor
        from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor
        scene = Image.linear_gradient('L').resize((95, 143)).convert('RGB')
        refs = [scene, scene, Image.new('RGB', (32, 32), 'red')]
        _, prepared = prepare_images(None, refs, 4096)
        processor = Flux3ImageProcessor()
        for reference in prepared:
            result, _ = processor.preprocess(reference, resize_mode='crop')
            torch.testing.assert_close(result[0], image_tensor(reference), rtol=0, atol=1e-7)
        self.assertEqual(np.array(prepared[2])[0, 0].tolist(), [255, 0, 0])

    def test_real_tiny_pipeline_saves_validation_and_refuses_overwrite(self):
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel
        from flux3_recolor import CONDITION_FOLDERS, TARGET_FOLDER
        from flux3_recolor.inference import build_pipeline, generate_validation
        vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), layers_per_block=1,
                                latent_channels=2, norm_num_groups=4, mid_block_add_attention=False).eval()
        transformer = Flux3Transformer2DModel(
            in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=1, joint_attention_dim=12,
            timestep_channels=8, axes_dims_rope=(4, 4, 4, 4), mlp_ratio=2).eval()
        scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False)
        original_sigmas = scheduler.sigmas.clone()
        pipeline = build_pipeline({'vae': vae, 'transformer': transformer, 'scheduler': scheduler}, 'cpu')
        embeddings = (torch.ones(1, 2, 12), torch.zeros(1, 2, 12))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dataset'
            for folder in (TARGET_FOLDER, *CONDITION_FOLDERS):
                (root / folder).mkdir(parents=True)
                Image.new('RGB', (32, 32), 'red').save(root / folder / 'example.png')
            output = Path(tmp) / 'validation'
            from build_flux3_recolor_dataset import generate_samples
            records = list(generate_samples(root, 'b_to_a', ['example.png'], 0))
            generate_validation(pipeline, root, records, {'max_area': 1024}, output, 42, 1, embeddings)
            sample = output / '0000'
            with Image.open(sample / 'generated.png') as image:
                self.assertEqual(image.size, (32, 32))
            with Image.open(sample / 'comparison.png') as image:
                self.assertEqual(image.size, (160, 32))
            self.assertTrue((output / 'manifest.json').is_file())
            # Validation sampling must never mutate the 1000-step training noise schedule.
            torch.testing.assert_close(scheduler.sigmas, original_sigmas, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                generate_validation(pipeline, root, records, {'max_area': 1024}, output, 42, 1, embeddings)
            # Explicit training bucket is authoritative during inference, too.
            from flux3_recolor.inference import generate_sample
            config = {'max_area': 1024, 'preprocessing': {'max_area': 1024,
                      'buckets': [[32, 48]], 'use_aspect_ratio_buckets': True}}
            generated, refs = generate_sample(pipeline, [Image.new('RGB', (32, 32), 'red')] * 3,
                                               config, 42, 1, embeddings)
            self.assertEqual(generated.size, (48, 32))
            self.assertEqual(refs[0].size, (48, 32))

    def test_adapter_config_rejects_wrong_reference_order(self):
        from flux3_recolor.inference import load_run_config
        from flux3_recolor.config import write_json
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / 'recolor_config.json', {'format_version': 1, 'condition_folders': ['bad']})
            with self.assertRaises(ValueError):
                load_run_config(tmp)


if __name__ == '__main__':
    unittest.main()
