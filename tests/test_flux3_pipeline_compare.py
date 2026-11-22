import tempfile
import unittest
from pathlib import Path


class PipelineCompareTests(unittest.TestCase):
    def test_real_training_helper_and_native_pipeline_are_compared_without_lora_or_grad(self):
        import torch
        import json
        from flux3_recolor_test_utils import make_tiny_repository, make_dataset
        from tools.flux3.compare_flux3_training_pipeline import main
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data = make_tiny_repository(root / 'model'), make_dataset(root / 'data')
            (model / 'modular_model_index.json').write_text(json.dumps(
                {'_class_name': 'Flux3ModularPipeline', '_blocks_class_name': 'Flux3AutoBlocks',
                 'vae': ['diffusers', 'AutoencoderKLFlux2',
                         {'pretrained_model_name_or_path': str(model), 'subfolder': 'vae',
                          'type_hint': ['diffusers', 'AutoencoderKLFlux2']}]}))
            report = main(['--pretrained_model_name_or_path', str(model), '--instance_data_dir', str(data),
                           '--sample', '0.png', '--direction', 'a_to_b', '--output_dir', str(root / 'report'),
                           '--device', 'cpu', '--mixed_precision', 'no', '--resolution', '32',
                           '--max_sequence_length', '160', '--text_encoder_out_layers', '1', '--seeds', '42', '43',
                           '--save_tensors'])
            self.assertEqual(report['lora_parameter_count'], 0)
            self.assertEqual(report['trainable_parameter_count'], 0)
            self.assertFalse(report['grad_enabled'])
            self.assertEqual(report['pipeline_vae_dtype'], 'torch.float32')
            self.assertEqual(report['pipeline_autocast'], 'off')
            self.assertEqual(report['pipeline_class'], 'Flux3ModularPipeline')
            self.assertEqual(len(report['cases']), 2)
            self.assertTrue((root / 'report/report.json').is_file())
            for case in report['cases']:
                self.assertEqual(case['pipeline_forward_calls'], 1)  # No unconditional / CFG pass.
                self.assertTrue(case['matched_input_replay']['allclose'])
                self.assertTrue(case['training_context_replay']['exact_equal'])
                self.assertTrue(case['manual_loss_vs_training_loss']['allclose'])
                self.assertTrue(case['inputs']['hidden_states']['exact_equal'])
                self.assertTrue(case['inputs']['timestep']['allclose'])
                self.assertTrue(case['inputs']['image_cond_ids']['exact_equal'])
                self.assertTrue(case['text_embeddings']['allclose'])
                self.assertEqual(case['training_prediction']['shape'], [1, 4, 8])
                self.assertEqual(case['one_step_latent_difference']['left']['shape'],
                                 case['one_step_latent_difference']['right']['shape'])
            self.assertNotEqual(report['cases'][0]['sigma'], report['cases'][1]['sigma'])
            self.assertTrue(all(value['exact_equal'] for value in report['rgb_preprocessing'].values()))
            from tools.flux3.analyze_flux3_numeric_precision import main as numeric_main
            numeric = numeric_main(['--pretrained_model_name_or_path', str(model), '--report_root', str(root / 'report'),
                                    '--output_dir', str(root / 'numeric'), '--device', 'cpu', '--compute_dtype', 'fp32'])
            for case in numeric['cases']:
                self.assertTrue(case['repeated_training_forward']['exact_equal'])
                self.assertTrue(case['fp32_arithmetic']['repeat']['exact_equal'])
            self.assertTrue(numeric['no_lora'])
            self.assertEqual(numeric['parameters_with_grad'], 0)

    def test_difference_report_does_not_hide_shape_or_value_mismatches(self):
        import torch
        from tools.flux3.compare_flux3_training_pipeline import tensor_difference
        self.assertFalse(tensor_difference(torch.ones(1, 2), torch.ones(2, 1), 1e-5, 1e-4)['allclose'])
        result = tensor_difference(torch.zeros(2), torch.ones(2), 1e-5, 1e-4)
        self.assertFalse(result['allclose'])
        self.assertEqual(result['max_abs'], 1.)


if __name__ == '__main__':
    unittest.main()
