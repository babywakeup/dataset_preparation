import unittest
from unittest.mock import patch

import torch


class VaeBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diffusers import AutoencoderKLFlux2
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), latent_channels=2,
            norm_num_groups=4, layers_per_block=1, mid_block_add_attention=False).requires_grad_(False).eval()
        with torch.no_grad():
            n = cls.vae.bn.running_mean.numel()
            cls.vae.bn.running_mean.copy_(torch.linspace(-0.3, 0.5, n))
            cls.vae.bn.running_var.copy_(torch.linspace(0.7, 1.3, n))

    def inputs(self, batch_size):
        generator = torch.Generator().manual_seed(100 + batch_size)
        scenes = [torch.rand(batch_size, 3, 32, 48, generator=generator) * 2 - 1 for _ in range(3)]
        swatch = torch.rand(batch_size, 3, 16, 16, generator=generator) * 2 - 1
        return scenes[0], [*scenes[1:], swatch]

    def test_two_forwards_match_separate_encoding_and_preserve_order(self):
        from flux3_recolor.modeling import encode_recolor_latents, encode_latents
        for size in (1, 2, 3):
            with self.subTest(batch_size=size):
                target, refs = self.inputs(size)
                expected = [encode_latents(self.vae, pixels) for pixels in [target, *refs]]
                state = torch.get_rng_state().clone()
                with patch.object(self.vae, 'encode', wraps=self.vae.encode) as encode:
                    actual_target, actual_refs = encode_recolor_latents(self.vae, target, refs)
                self.assertEqual(encode.call_count, 2)
                self.assertEqual(tuple(encode.call_args_list[0].args[0].shape), (3 * size, 3, 32, 48))
                self.assertEqual(tuple(encode.call_args_list[1].args[0].shape), (size, 3, 16, 16))
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                for actual, original in zip([actual_target, *actual_refs], expected):
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertFalse(actual.requires_grad)
                    # Batch convolution can reorder floating-point sums. Keep a
                    # small absolute allowance near zero and relative bound elsewhere.
                    torch.testing.assert_close(actual, original, rtol=1e-4, atol=1e-5)

    def test_native_fp32_vae_is_unchanged_by_outer_bf16_autocast(self):
        from flux3_recolor.modeling import encode_recolor_latents
        target, refs = self.inputs(2)
        expected = encode_recolor_latents(self.vae, target, refs)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = encode_recolor_latents(self.vae, target, refs)
        for a, b in zip([actual[0], *actual[1]], [expected[0], *expected[1]]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(a.dtype, torch.float32)

    def test_invalid_shapes_fail_before_vae_forward(self):
        from flux3_recolor.modeling import encode_recolor_latents
        target, refs = self.inputs(2)
        for bad_refs in [refs[:2], [refs[0][:1], *refs[1:]], [*refs[:2], refs[2][:1]]]:
            with patch.object(self.vae, 'encode', wraps=self.vae.encode) as encode:
                with self.assertRaises(ValueError):
                    encode_recolor_latents(self.vae, target, bad_refs)
                self.assertEqual(encode.call_count, 0)

    def test_same_noise_loss_and_lora_gradients_match_unbatched_reference(self):
        from diffusers import Flux3Transformer2DModel, FlowMatchEulerDiscreteScheduler
        from flux3_recolor.modeling import (encode_latents, encode_recolor_latents, add_lora,
                                           flow_matching_loss, text_position_ids)
        torch.manual_seed(9)
        model = Flux3Transformer2DModel(in_channels=8, cond_in_channels=8, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=1,
            joint_attention_dim=16, timestep_channels=8, axes_dims_rope=(4,4,4,4), mlp_ratio=2)
        add_lora(model, rank=2, alpha=2, dropout=0.0, target_modules=['to_q','to_v'])
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.fill_(0.02)
        target, refs = self.inputs(2)
        encoded = [encode_latents(self.vae, pixels) for pixels in [target, *refs]]
        combined = encode_recolor_latents(self.vae, target, refs)
        embeds = torch.randn(2,4,16)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93)
        losses, gradients = [], []
        for y, cond in [(encoded[0], encoded[1:]), combined]:
            model.zero_grad(set_to_none=True)
            torch.manual_seed(42)  # Identical noise and sampled timestep.
            loss = flow_matching_loss(model, y, cond, embeds, text_position_ids(4,'cpu'), scheduler,
                                      compute_dtype=torch.float32)
            loss.backward()
            losses.append(loss.detach())
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None]))
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-4, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
