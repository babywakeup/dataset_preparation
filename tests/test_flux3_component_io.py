import tempfile
import unittest
from pathlib import Path
import torch


class ComponentIOTests(unittest.TestCase):
    def test_only_trained_components_saved_with_base_reference(self):
        from flux3_recolor.component_io import save_components, load_component_config
        from flux3_recolor.text_encoder import TextEncoderModel
        from test_flux3_recolor_training import tiny_transformer
        from flux3_recolor.modeling import add_lora
        model = tiny_transformer()
        add_lora(model, 2, 2, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_components(model, root / 'lora', 'base-model')
            config = load_component_config(root / 'lora')
            self.assertEqual(config['base_model'], 'base-model')
            self.assertEqual(config['text_encoder']['type'], 'base')
            self.assertTrue((root/'lora/transformer/pytorch_lora_weights.safetensors').is_file())
            self.assertFalse((root/'lora/text_encoder').exists())
            self.assertFalse((root/'lora/vae').exists())
            encoder = TextEncoderModel.from_features(torch.randn(1,3,12))
            save_components(model, root / 'task', 'base-model', encoder)
            restored = TextEncoderModel.from_pretrained(root/'task/text_encoder')
            torch.testing.assert_close(restored.weight, encoder.weight, rtol=0, atol=0)
            self.assertEqual(load_component_config(root/'task')['text_encoder']['type'], 'learned')
            self.assertEqual({p.name for p in (root/'task').iterdir() if p.is_dir()}, {'transformer','text_encoder'})
