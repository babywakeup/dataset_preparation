import json
import tempfile
import unittest
from pathlib import Path


class StandalonePipelineTests(unittest.TestCase):
    def test_raw_pipeline_and_optional_text_offload_generate_without_lora(self):
        from flux3_recolor_test_utils import make_tiny_repository, make_dataset
        from test_flux3_recolor_pipeline import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data = make_tiny_repository(root / 'model'), make_dataset(root / 'data')
            index = {'_class_name': 'Flux3ModularPipeline', '_blocks_class_name': 'Flux3AutoBlocks'}
            for name, lib, cls, folder in [('vae', 'diffusers', 'AutoencoderKLFlux2', 'vae'),
                    ('transformer', 'diffusers', 'Flux3Transformer2DModel', 'transformer'),
                    ('text_encoder', 'transformers', 'Qwen3VLForConditionalGeneration', 'text_encoder'),
                    ('tokenizer', 'transformers', 'AutoProcessor', 'tokenizer')]:
                index[name] = [lib, cls, {'pretrained_model_name_or_path': str(model), 'subfolder': folder,
                                         'type_hint': [lib, cls]}]
            (model / 'modular_model_index.json').write_text(json.dumps(index))
            for offload in (False, True):
                output = root / str(offload)
                main(['--pretrained_model_name_or_path', str(model), '--dataset_root', str(data),
                      '--sample', '0.png', '--device', 'cpu', '--mixed_precision', 'no', '--resolution', '32',
                      '--color_swatch_size', '32', '--num_inference_steps', '1', '--max_sequence_length', '160',
                      '--text_encoder_out_layers', '1', '--output_dir', str(output),
                      *(['--offload_text_encoder'] if offload else [])])
                self.assertTrue((output / 'generated.png').is_file())
                config = json.loads((output / 'run_config.json').read_text())
                self.assertEqual(config['condition_order'], ['gray', 'source', 'swatch'])
                self.assertFalse(config['lora_loaded'])
                self.assertNotIn('black', config['prompt'].lower())
