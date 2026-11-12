"""Local HF cache and official FLUX.3 modular component-layout regressions."""
import json
import tempfile
import unittest
from pathlib import Path


class LocalFlux3ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for folder in ('transformer', 'vae', 'text_encoder'):
            path = self.root / folder
            path.mkdir()
            (path / 'weights.safetensors').write_bytes(b'fixture')
            (path / 'config.json').write_text(json.dumps({'_class_name': 'Flux3Transformer2DModel'}))
        (self.root / 'text_encoder/tokenizer_config.json').write_text('{}')

    def test_base_model_accepts_processor_in_text_encoder_folder(self):
        from flux3_recolor.config import validate_local_model, tokenizer_subfolder
        (self.root / 'modular_model_index.json').write_text(json.dumps({
            'tokenizer': ['transformers', 'Qwen3VLProcessor', {'subfolder': 'text_encoder'}]}))
        path, signature = validate_local_model(self.root)
        self.assertEqual(path, str(self.root.resolve()))
        self.assertEqual(tokenizer_subfolder(path), 'text_encoder')
        self.assertEqual(len(signature), 64)

    def test_hf_cache_root_resolves_main_snapshot(self):
        from flux3_recolor.config import validate_local_model
        cache = self.root / 'cache'
        snapshot = cache / 'snapshots' / ('a' * 40)
        snapshot.mkdir(parents=True)
        (cache / 'refs').mkdir()
        (cache / 'refs/main').write_text('a' * 40)
        for folder in ('transformer', 'vae', 'text_encoder'):
            (snapshot / folder).symlink_to(self.root / folder, target_is_directory=True)
        path, _ = validate_local_model(cache)
        self.assertEqual(path, str(snapshot.resolve()))

    def test_unfinished_index_or_escaping_component_metadata_is_rejected(self):
        from flux3_recolor.config import validate_local_model
        index = self.root / 'transformer/diffusion_pytorch_model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a': 'missing.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'missing.safetensors'):
            validate_local_model(self.root)
        index.unlink()
        (self.root / 'modular_model_index.json').write_text(json.dumps({
            'tokenizer': ['transformers', 'Qwen3VLProcessor', {'subfolder': '../outside'}]}))
        with self.assertRaises(ValueError):
            validate_local_model(self.root)


if __name__ == '__main__':
    unittest.main()
