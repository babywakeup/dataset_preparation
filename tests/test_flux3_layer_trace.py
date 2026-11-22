import unittest
import tempfile
from pathlib import Path

import torch


class LayerTraceTests(unittest.TestCase):
    def test_all_three_stages_run_on_a_real_tiny_repository(self):
        from flux3_recolor_test_utils import make_tiny_repository, make_dataset
        from tools.flux3.compare_flux3_training_pipeline import main as compare
        from tools.flux3.trace_flux3_tensor_path import main as trace
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data = make_tiny_repository(root / 'model'), make_dataset(root / 'data')
            compare(['--pretrained_model_name_or_path', str(model), '--instance_data_dir', str(data),
                     '--sample', '0.png', '--output_dir', str(root / 'compare'), '--resolution', '32',
                     '--device', 'cpu', '--mixed_precision', 'no', '--text_encoder_out_layers', '1',
                     '--max_sequence_length', '160', '--seeds', '42', '--save_tensors'])
            result = trace(['--pretrained_model_name_or_path', str(model), '--instance_data_dir', str(data),
                            '--report_dir', str(root / 'compare'), '--output_dir', str(root / 'trace'), '--device', 'cpu'])
            self.assertTrue(result['transformer']['baseline_difference']['exact_equal'])
            self.assertTrue(result['vae_encoder']['official_helper_vs_native_encoder']['exact_equal'])
            self.assertTrue(result['vae_decoder']['official_manual_vs_native_decoder']['exact_equal'])
            from tools.flux3.trace_flux3_tensor_path import verify_transformer_controls
            controls = verify_transformer_controls(str(model), root / 'compare', torch.device('cpu'), root / 'trace')
            self.assertEqual(len(controls), 1)
            self.assertTrue(controls[0]['layernorm_intervention']['exact_equal'])

    def test_fingerprint_indices_remain_in_bounds_for_large_activations(self):
        from tools.flux3.trace_flux3_tensor_path import fingerprint
        torch.set_num_threads(1)
        result = fingerprint(torch.zeros(16_777_220))
        self.assertEqual(len(result['sample']), 256)

    def test_first_mismatch_is_localized_and_layernorm_intervention_is_scoped(self):
        from tools.flux3.trace_flux3_tensor_path import trace_forward, first_difference, disable_norm_autocast
        torch.set_num_threads(1)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 2)).eval()
        x = torch.randn(2, 4)
        _, before = trace_forward(model, lambda: model(x), max_leaf_events=20)
        with torch.no_grad():
            model[1].bias.add_(1)
        _, after = trace_forward(model, lambda: model(x), max_leaf_events=20)
        mismatch = first_difference(before, after)
        self.assertEqual(mismatch['left']['module'], '1')
        self.assertEqual(mismatch['left']['event'], 'post')
        original = model[1].forward
        with disable_norm_autocast(model, (torch.nn.LayerNorm,)) as names:
            self.assertEqual(names, ['1'])
            self.assertIsNot(model[1].forward, original)
        self.assertEqual(model[1].forward, original)


if __name__ == '__main__':
    unittest.main()
