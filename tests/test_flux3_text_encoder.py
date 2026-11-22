import tempfile
import unittest
from pathlib import Path

import torch


class TextEncoderTests(unittest.TestCase):
    def test_independent_encoder_backpropagates_through_transformer(self):
        from flux3_recolor.text_encoder import TextEncoderModel
        from test_flux3_recolor_training import tiny_transformer
        from flux3_recolor.modeling import add_lora, image_position_ids, text_position_ids
        torch.set_num_threads(1)
        model = tiny_transformer()
        add_lora(model, 2, 2, 0)
        original_forward = model.forward.__func__
        initial = torch.randn(1, 3, 12)
        encoder = TextEncoderModel.from_features(initial)
        features = encoder(batch_size=2).last_hidden_state
        output = model(hidden_states=torch.randn(2,4,8), img_ids=image_position_ids(2,2,'cpu'),
            timestep=torch.ones(2), encoder_hidden_states=features, txt_ids=text_position_ids(3,'cpu'), return_dict=False)[0]
        output.square().mean().backward()
        self.assertGreater(encoder.weight.grad.abs().sum().item(), 0)
        self.assertEqual(encoder.weight.dtype, torch.float32)
        self.assertIs(model.forward.__func__, original_forward)
        self.assertFalse(hasattr(model, 'recolor_text_encoder'))
        self.assertFalse(any(p is encoder.weight for p in model.parameters()))

    def test_transformers_save_load_and_inplace_checkpoint_restore(self):
        from flux3_recolor.text_encoder import TextEncoderModel, save_text_encoder, load_text_encoder_into
        encoder = TextEncoderModel.from_features(torch.randn(1,3,12))
        parameter = encoder.weight
        with tempfile.TemporaryDirectory() as tmp:
            save_text_encoder(encoder, tmp)
            self.assertTrue((Path(tmp)/'text_encoder/config.json').is_file())
            self.assertTrue((Path(tmp)/'text_encoder/model.safetensors').is_file())
            restored = TextEncoderModel.from_pretrained(Path(tmp)/'text_encoder')
            torch.testing.assert_close(restored.weight, parameter, rtol=0, atol=0)
            expected = parameter.detach().clone()
            with torch.no_grad():
                parameter.zero_()
            load_text_encoder_into(encoder, tmp)
            self.assertIs(encoder.weight, parameter)
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'shape'):
                load_text_encoder_into(TextEncoderModel.from_features(torch.randn(1,4,12)), tmp)

    def test_config_length_controls_tokens_and_survives_save_load(self):
        from flux3_recolor.text_encoder import TextEncoderConfig, TextEncoderModel
        encoder = TextEncoderModel(TextEncoderConfig(length=7, hidden_size=12))
        self.assertEqual(tuple(encoder(batch_size=2).last_hidden_state.shape), (2,7,12))
        with tempfile.TemporaryDirectory() as tmp:
            encoder.save_pretrained(tmp)
            loaded = TextEncoderModel.from_pretrained(tmp)
            self.assertEqual(loaded.config.length, 7)
            self.assertNotIn('sequence_length', loaded.config.to_dict())
            self.assertEqual(tuple(loaded.weight.shape), (1,7,12))
        for length in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                TextEncoderConfig(length=length, hidden_size=12)

    def test_default_initial_tensor_loading_and_token_length_alignment(self):
        from safetensors.torch import load_file
        from flux3_recolor.text_encoder import TextEncoderConfig, TextEncoderModel, DEFAULT_INITIAL_PATH
        if not Path(DEFAULT_INITIAL_PATH).is_file():
            self.skipTest(f'{DEFAULT_INITIAL_PATH} does not exist')
        expected = load_file(DEFAULT_INITIAL_PATH)['embedding']
        # 1. 默认配置加载 (length=240, hidden_size=20480)
        model = TextEncoderModel(TextEncoderConfig())
        torch.testing.assert_close(model.weight, expected)
        # 2. 长度不同时的切片/拼接
        model_short = TextEncoderModel(TextEncoderConfig(length=100))
        torch.testing.assert_close(model_short.weight, expected[:, :100, :])
        model_long = TextEncoderModel(TextEncoderConfig(length=300))
        torch.testing.assert_close(model_long.weight[:, :240, :], expected)
        self.assertTrue((model_long.weight[:, 240:, :] == 0).all())

    def test_batch_expansion_and_input_validation(self):
        from flux3_recolor.text_encoder import TextEncoderModel
        value = torch.randn(1,3,12)
        encoder = TextEncoderModel.from_features(value)
        torch.testing.assert_close(encoder(batch_size=4).last_hidden_state, value.expand(4,-1,-1))
        with self.assertRaises(ValueError):
            TextEncoderModel.from_features(torch.randn(2,3,12))


class TextEncoderIntegrationTests(unittest.TestCase):
    def test_training_updates_validation_uses_saved_embedding_and_resume_is_exact(self):
        import json
        from unittest.mock import patch
        from safetensors.torch import load_file
        from flux3_recolor_test_utils import make_tiny_repository, make_dataset
        from train_flux3_recolor_text_encoder import main as train
        from flux3_recolor.inference import main as infer, generate_validation
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data = make_tiny_repository(root / 'model'), make_dataset(root / 'data')
            options = ['--pretrained_model_name_or_path', str(model), '--instance_data_dir', str(data),
                '--mixed_precision', 'no', '--rank', '2', '--lora_alpha', '2', '--resolution', '32',
                '--color_swatch_size', '32', '--max_sequence_length', '8', '--text_encoder_out_layers', '1',
                '--train_batch_size', '2', '--gradient_accumulation_steps', '2', '--gradient_checkpointing',
                '--checkpointing_steps', '1', '--report_to', 'none', '--num_validation_images', '1', '--validation_steps', '1', '--validation_epochs', '999',
                '--validation_inference_steps', '1', '--skip_final_inference', '--lora_dropout', '0.1']
            full, resumed = root / 'full', root / 'resumed'
            validation_calls = []
            def capture(*args):
                saved = load_file(str(full / f'checkpoint-{len(validation_calls) + 1}' /
                                      'text_encoder/model.safetensors'))['weight']
                torch.testing.assert_close(args[7][0].cpu(), saved, rtol=0, atol=0)
                self.assertFalse(args[7][0].requires_grad)
                validation_calls.append(args[7][0].clone())
                return generate_validation(*args)
            with patch('flux3_recolor.inference.generate_validation', side_effect=capture):
                result = train([*options, '--output_dir', str(full), '--max_train_steps', '3'])
            self.assertEqual(len(validation_calls), 3)
            self.assertFalse(torch.equal(validation_calls[0], validation_calls[-1]))
            self.assertTrue(result['text_encoder'].weight.requires_grad)
            train([*options, '--output_dir', str(resumed), '--max_train_steps', '1'])
            train([*options, '--output_dir', str(resumed), '--max_train_steps', '3', '--resume_from_checkpoint', 'latest'])
            for filename in ('text_encoder/model.safetensors', 'transformer/pytorch_lora_weights.safetensors'):
                a, b = [load_file(str(path / filename)) for path in (full, resumed)]
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            config = json.loads((full / 'recolor_config.json').read_text())
            self.assertTrue(config['train_text_encoder'])
            expected = load_file(str(full / 'text_encoder/model.safetensors'))['weight']
            def infer_capture(*args):
                torch.testing.assert_close(args[7][0].cpu(), expected, rtol=0, atol=0)
                return generate_validation(*args)
            with patch('flux3_recolor.inference.generate_validation', side_effect=infer_capture):
                infer(['--lora_dir', str(full), '--dataset_root', str(data), '--output_dir', str(root / 'inferred'),
                       '--device', 'cpu', '--mixed_precision', 'no', '--num_inference_steps', '1', '--limit', '1'])
            self.assertTrue((root / 'inferred/manifest.json').is_file())
            from examples.recolor.demo_flux3_recolor_trained_pipeline import main as demo
            from flux3_recolor import CONDITION_FOLDERS
            name = next((data / CONDITION_FOLDERS[0]).glob('*.png')).name
            demo_options = ['--model_dir', str(full / 'checkpoint-3'),
                            '--output_dir', str(root / 'demo'), '--device', 'cpu',
                            '--mixed_precision', 'no', '--num_inference_steps', '1']
            for flag, folder in zip(('--gray_image', '--source_image', '--swatch_image'), CONDITION_FOLDERS):
                demo_options.extend([flag, str(data / folder / name)])
            from flux3_recolor.inference import generate_sample
            def demo_capture(pipeline, *args, **kwargs):
                torch.testing.assert_close(pipeline.recolor_text_encoder.weight.cpu(), expected, rtol=0, atol=0)
                return generate_sample(pipeline, *args, **kwargs)
            with patch('examples.recolor.demo_flux3_recolor_trained_pipeline.generate_sample', side_effect=demo_capture):
                image = demo(demo_options)
            self.assertEqual(image.size, (32, 32))
            self.assertTrue((root / 'demo/generated.png').is_file())
            self.assertTrue((root / 'demo/comparison.png').is_file())

    def test_invalid_prompt_modes_are_rejected(self):
        from train_flux3_recolor_lora import parse_args
        for extra in (['--prompt_augmentation'], ['--caption_dropout', '0.1'], ['--validation_prompt', 'other'],
                      ['--text_encoder_learning_rate', '0.01']):
            with self.assertRaises(SystemExit):
                parse_args(['--pretrained_model_name_or_path', 'unused-local-model', '--train_text_encoder', *extra])


class TextEncoderDistributedTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').environ.get('RUN_FLUX3_DDP_TESTS') == '1', 'opt-in two-rank CPU DDP')
    def test_two_ranks_sync_and_resume_text_encoder_and_lora(self):
        import os, subprocess, sys
        from safetensors.torch import load_file
        from flux3_recolor_test_utils import make_tiny_repository, make_dataset
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data = make_tiny_repository(root / 'model'), make_dataset(root / 'data', 13)
            options = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2',
                'tests/flux3_reference_ddp_worker.py', '--pretrained_model_name_or_path', str(model),
                '--instance_data_dir', str(data), '--train_text_encoder', '--mixed_precision', 'no',
                '--report_to', 'none', '--rank', '2', '--lora_alpha', '2', '--resolution', '32',
                '--color_swatch_size', '32', '--max_sequence_length', '8', '--text_encoder_out_layers', '1',
                '--train_batch_size', '2', '--gradient_accumulation_steps', '2', '--gradient_checkpointing', '--checkpointing_steps', '1', '--validation_steps', '1',
                '--num_validation_images', '1', '--validation_inference_steps', '1', '--skip_final_inference']
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'ACCELERATE_USE_CPU': 'true',
                   'OMP_NUM_THREADS': '1', 'PYTHONPATH': str(Path.cwd()) + ':' + str(Path.cwd() / 'tests')}
            def run(path, steps, resume=False):
                command = [*options, '--output_dir', str(path), '--max_train_steps', str(steps)]
                if resume:
                    command += ['--resume_from_checkpoint', 'latest']
                log = root / (path.name + str(steps) + '.log')
                with log.open('w') as stream:
                    result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
                self.assertEqual(result.returncode, 0, log.read_text()[-10000:])
            full, resumed = root / 'full', root / 'resumed'
            run(full, 2)
            run(resumed, 1)
            run(resumed, 2, True)
            for path in (full, resumed):
                a, b = [load_file(str(path / f'rank-artifacts/rank-{rank}.safetensors')) for rank in (0, 1)]
                self.assertIn('text_encoder.weight', a)
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            for filename in ('text_encoder/model.safetensors', 'transformer/pytorch_lora_weights.safetensors'):
                a, b = [load_file(str(path / filename)) for path in (full, resumed)]
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


class TextEncoderCudaTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').environ.get('RUN_FLUX3_CUDA_TESTS') == '1', 'opt-in single-GPU BF16')
    def test_bf16_optimizer_updates_fp32_feature_and_validation_continues(self):
        from unittest.mock import patch
        from flux3_recolor_test_utils import make_tiny_repository, make_dataset
        from flux3_recolor.text_encoder import TextEncoderModel
        from train_flux3_recolor_text_encoder import main
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 1)
        initial = []
        original = TextEncoderModel.from_features
        def capture(value):
            encoder = original(value)
            initial.append(encoder.weight.detach().cpu().clone())
            return encoder
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data = make_tiny_repository(root / 'model'), make_dataset(root / 'data', 4)
            with patch.object(TextEncoderModel, 'from_features', side_effect=capture):
                result = main(['--pretrained_model_name_or_path', str(model), '--instance_data_dir', str(data),
                    '--output_dir', str(root / 'trained'), '--mixed_precision', 'bf16', '--rank', '2',
                    '--lora_alpha', '2', '--resolution', '32', '--color_swatch_size', '32',
                    '--max_sequence_length', '8', '--text_encoder_out_layers', '1', '--max_train_steps', '2',
                    '--checkpointing_steps', '1', '--validation_steps', '1', '--num_validation_images', '1',
                    '--validation_inference_steps', '1', '--report_to', 'none', '--gradient_checkpointing',
                    '--learning_rate', '0.01', '--skip_final_inference'])
            model = result['model']
            self.assertEqual(model.dtype, torch.bfloat16)
            self.assertEqual(result['text_encoder'].weight.dtype, torch.float32)
            self.assertFalse(torch.equal(initial[0], result['text_encoder'].weight.detach().cpu()))
            self.assertTrue((root / 'trained/validation-step-2/manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
