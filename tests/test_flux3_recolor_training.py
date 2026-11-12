import json
import tempfile
import unittest
from pathlib import Path

import torch


def tiny_transformer():
    from diffusers import Flux3Transformer2DModel
    return Flux3Transformer2DModel(
        in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=1, joint_attention_dim=12,
        timestep_channels=8, axes_dims_rope=(4, 4, 4, 4), mlp_ratio=2,
    )


class RecolorTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_three_reference_positions_and_different_token_counts(self):
        from flux3_recolor.modeling import reference_tokens
        refs = [torch.full((1, 8, 2, 3), 1.), torch.full((1, 8, 2, 3), 2.), torch.full((1, 8, 1, 1), 3.)]
        tokens, ids = reference_tokens(refs)
        self.assertEqual(tokens.shape, (1, 13, 8))
        self.assertEqual(tokens[0, :, 0].tolist(), [1.] * 6 + [2.] * 6 + [3.])
        self.assertEqual(ids[:, 3].tolist(), [-1.] * 6 + [-2.] * 6 + [-3.])
        self.assertEqual(ids[5].tolist(), [0., 0.5, 1., -1.])
        with self.assertRaises(ValueError):
            reference_tokens(refs[:2])

    def test_latent_patchify_matches_flux3_pipeline(self):
        from flux3_recolor.modeling import patchify_latents
        from diffusers.modular_pipelines.flux3.encoders import Flux3VaeEncoderStep
        image = torch.arange(48).reshape(1, 2, 4, 6)
        torch.testing.assert_close(patchify_latents(image), Flux3VaeEncoderStep._patchify_latents(image))

    def test_flow_loss_trains_only_lora_and_reaches_reference_branch(self):
        from flux3_recolor.modeling import add_lora, flow_matching_loss, text_position_ids
        from diffusers import FlowMatchEulerDiscreteScheduler
        torch.manual_seed(4)
        model = tiny_transformer()
        add_lora(model, rank=2, alpha=2, dropout=0)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        target = torch.randn(1, 8, 2, 2)
        refs = [torch.randn(1, 8, 2, 2), torch.randn(1, 8, 2, 2), torch.randn(1, 8, 1, 1)]
        loss = flow_matching_loss(model, target, refs, torch.randn(1, 3, 12), text_position_ids(3, 'cpu'),
                                  FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any('image_cond' in n and p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.named_parameters() if p.requires_grad))
        optimizer.step()
        for name, param in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(param, frozen[name], rtol=0, atol=0)

    def test_standard_lora_save_reload_preserves_predictions(self):
        from flux3_recolor.modeling import add_lora, save_adapter, load_adapter_state
        from diffusers.loaders import Flux3LoraLoaderMixin
        model = tiny_transformer()
        base = {k: v.detach().clone() for k, v in model.state_dict().items()}
        add_lora(model, 2, 2, 0)
        for p in model.parameters():
            if p.requires_grad:
                p.data.fill_(0.123)
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(model, tmp)
            state = Flux3LoraLoaderMixin.lora_state_dict(tmp)
            self.assertTrue(all(k.startswith('transformer.') and 'lora' in k for k in state))
            restored = tiny_transformer()
            restored.load_state_dict(base)
            add_lora(restored, 2, 2, 0)
            load_adapter_state(restored, tmp)
        for (n, p), (n2, p2) in zip(model.named_parameters(), restored.named_parameters()):
            self.assertEqual(n, n2)
            torch.testing.assert_close(p, p2)

    def test_inference_loading_preserves_nondefault_lora_alpha(self):
        from flux3_recolor.modeling import add_lora, save_adapter
        from flux3_recolor.inference import build_pipeline
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
        model = tiny_transformer()
        add_lora(model, 2, 6, 0)
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(model, tmp)
            restored = tiny_transformer()
            pipeline = build_pipeline({'transformer': restored,
                'vae': AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), latent_channels=2,
                                         norm_num_groups=4, layers_per_block=1),
                'scheduler': FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False)}, 'cpu')
            pipeline.load_lora_weights(tmp, local_files_only=True)
            scales = [module.scaling['default_0'] for module in restored.modules()
                      if hasattr(module, 'scaling') and 'default_0' in module.scaling]
            self.assertTrue(scales)
            self.assertTrue(all(scale == 3.0 for scale in scales), scales)



    def test_local_model_check_rejects_distilled_or_missing_components(self):
        from flux3_recolor.config import validate_local_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                validate_local_model(root)
            for component in ['vae', 'transformer', 'text_encoder', 'tokenizer']:
                (root / component).mkdir()
            (root / 'transformer/config.json').write_text(json.dumps({'flow_map': True}))
            with self.assertRaisesRegex(ValueError, 'flow_map'):
                validate_local_model(root)




if __name__ == '__main__':
    unittest.main()
