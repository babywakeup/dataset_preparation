"""Independent numerical checks against the FLUX.3 pipeline and analytic flow objective."""
import unittest
from unittest.mock import patch

import torch


class Flux3MathAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_flow_objective_and_gradient_match_analytic_oracle(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from flux3_recolor.modeling import flow_matching_loss, text_position_ids
        scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False)
        target = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 4) / 10
        noise = torch.flip(target, dims=[-1]) - 0.7
        refs = [target + 10, target + 20, torch.full((2, 2, 1, 1), 30.)]
        u = torch.tensor([0.1, 0.8])
        indices = (u * 1000).long()
        # Shifted sigma schedule, derived independently from the scheduler's recipe.
        base_sigma = torch.linspace(1., 1. / 1000, 1000)[indices]
        sigma = 6.93 * base_sigma / (1 + (6.93 - 1) * base_sigma)
        expected_input = ((1 - sigma[:, None, None, None]) * target
                          + sigma[:, None, None, None] * noise).flatten(2).transpose(1, 2)
        velocity = (noise - target).flatten(2).transpose(1, 2)
        for scheme in ('none', 'sigma_sqrt', 'cosmap', 'logit_normal', 'mode'):
            for offset in (0., 0.25):
                with self.subTest(scheme=scheme, offset=offset):
                    residual = torch.nn.Parameter(torch.tensor(offset))
                    def model(**kwargs):
                        torch.testing.assert_close(kwargs['hidden_states'], expected_input)
                        torch.testing.assert_close(kwargs['timestep'], sigma)
                        expected_refs = torch.cat([r.flatten(2).transpose(1, 2) for r in refs], 1)
                        torch.testing.assert_close(kwargs['image_cond_hidden_states'], expected_refs)
                        self.assertEqual(kwargs['image_cond_ids'][:, 3].tolist(), [-1.] * 8 + [-2.] * 8 + [-3.])
                        self.assertEqual(kwargs['img_ids'][-1].tolist(), [0., 0.5, 1.5, 0.])
                        return (velocity + residual,)
                    with patch('torch.randn_like', return_value=noise), patch(
                            'diffusers.training_utils.compute_density_for_timestep_sampling', return_value=u):
                        loss = flow_matching_loss(model, target, refs, torch.zeros(2, 3, 4),
                                                  text_position_ids(3, 'cpu'), scheduler, weighting_scheme=scheme)
                    weights = (sigma.pow(-2) if scheme == 'sigma_sqrt' else
                               2 / (torch.pi * (1 - 2 * sigma + 2 * sigma.square())) if scheme == 'cosmap'
                               else torch.ones_like(sigma))
                    torch.testing.assert_close(loss, weights.mean() * offset ** 2)
                    loss.backward()
                    torch.testing.assert_close(residual.grad, weights.mean() * 2 * offset)

    def test_rgb_preprocessing_and_vae_normalization_match_pipeline(self):
        import numpy as np
        from PIL import Image
        from diffusers import AutoencoderKLFlux2
        from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor
        from diffusers.modular_pipelines.flux3.encoders import Flux3VaeEncoderStep
        from diffusers.modular_pipelines.flux3.decoders import Flux3DecodeStep
        from flux3_recolor.data import image_tensor, prepare_images
        from flux3_recolor.modeling import encode_latents
        torch.manual_seed(11)
        scene = Image.fromarray(np.random.default_rng(4).integers(0, 256, (79, 61, 3), dtype=np.uint8))
        _, refs = prepare_images(scene, [scene, scene.copy(), Image.new('RGB', (32, 32), (71, 118, 231))], 4096)
        processor = Flux3ImageProcessor()
        vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), latent_channels=2,
                                norm_num_groups=4, layers_per_block=1, mid_block_add_attention=False).eval()
        vae.bn.running_mean.copy_(torch.linspace(-0.3, 0.6, 8))
        vae.bn.running_var.copy_(torch.linspace(0.3, 2.3, 8))
        for dtype in (torch.float32, torch.bfloat16):
            vae.to(dtype=dtype)
            for ref in refs:
                pixels = image_tensor(ref).unsqueeze(0)
                official_pixels, _ = processor.preprocess(ref, resize_mode='crop')
                torch.testing.assert_close(pixels, official_pixels, rtol=0, atol=0)
                with torch.no_grad():
                    raw = vae.encode(official_pixels.to(dtype)).latent_dist.mode()
                    patched = Flux3VaeEncoderStep._patchify_latents(raw)
                    mean = vae.bn.running_mean[None, :, None, None]
                    std = torch.sqrt(vae.bn.running_var[None, :, None, None] + vae.config.batch_norm_eps)
                    expected = (patched - mean) / std
                # Equal tensor layout avoids convolution-kernel differences.
                actual = encode_latents(vae, official_pixels)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                tokens = actual.flatten(2).transpose(1, 2)
                unpacked = tokens.reshape(1, *actual.shape[-2:], actual.shape[1]).permute(0, 3, 1, 2)
                restored = Flux3DecodeStep._unpatchify_latents(unpacked * std + mean)
                torch.testing.assert_close(restored, raw, rtol=0.04 if dtype == torch.bfloat16 else 1e-5,
                                           atol=0.008 if dtype == torch.bfloat16 else 1e-6)

    def test_loss_rejects_broadcastable_wrong_prediction_shape(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from flux3_recolor.modeling import flow_matching_loss, text_position_ids
        target = torch.ones(2, 2, 2, 2)
        def wrong_batch_model(**kwargs):
            return (kwargs['hidden_states'][:1],)
        with self.assertRaisesRegex(ValueError, '预测.*形状'):
            flow_matching_loss(wrong_batch_model, target, [target, target, target],
                               torch.zeros(2, 3, 4), text_position_ids(3, 'cpu'),
                               FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False))

    def test_prompt_encoding_matches_official_pipeline_including_padding(self):
        import tempfile
        from pathlib import Path
        from flux3_recolor_test_utils import make_tiny_repository
        from flux3_recolor.modeling import get_qwen3vl_prompt_embeds
        from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        with tempfile.TemporaryDirectory() as tmp:
            root = make_tiny_repository(Path(tmp) / 'model')
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(root / 'text_encoder').eval()
            tokenizer = AutoProcessor.from_pretrained(root / 'tokenizer')
            for prompt in ('', 'recolor red', 'recolor red ' * 63):
                with torch.no_grad():
                    actual = get_qwen3vl_prompt_embeds(encoder, tokenizer, prompt, 'cpu', torch.float32, [1], 160)
                    expected = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                        encoder, tokenizer, prompt, 'cpu', max_sequence_length=160, hidden_states_layers=(1,))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_and_inference_transformer_inputs_match_at_same_noise(self):
        from PIL import Image
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
        from test_flux3_recolor_training import tiny_transformer
        from flux3_recolor.data import image_tensor
        from flux3_recolor.inference import build_pipeline
        from flux3_recolor.modeling import encode_latents, flow_matching_loss, text_position_ids
        torch.manual_seed(27)
        vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), latent_channels=2,
                                norm_num_groups=4, layers_per_block=1, mid_block_add_attention=False).eval()
        model = tiny_transformer().eval()
        scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False)
        images = [Image.new('RGB', (32, 48), (128, 128, 128)),
                  Image.new('RGB', (32, 48), (70, 140, 200)), Image.new('RGB', (128, 128), (20, 30, 80))]
        vae_inputs = []
        vae_outputs = []
        vae_debug = []
        initial_vae = {key: value.detach().clone() for key, value in vae.state_dict().items()}
        def record_input(module, args):
            vae_inputs.append(args[0].detach().clone())
            vae_debug.append((args[0].stride(), module.training, torch.is_autocast_enabled('cpu'),
                              args[0].dtype, next(module.parameters()).dtype))
            # Controlled differential test: feed both encoder calls the same
            # physical layout, rather than comparing different convolution kernels.
            return (args[0].contiguous(),)
        encoder_hook = vae.encoder.register_forward_pre_hook(
            record_input)
        quant_hook = vae.quant_conv.register_forward_hook(
            lambda module, args, output: vae_outputs.append(output.detach().clone()))
        refs = [encode_latents(vae, image_tensor(im).unsqueeze(0)) for im in images]
        target = refs[1]
        noise = torch.randn_like(target)
        embeddings = torch.randn(1, 3, 12)
        calls = []
        def capture(module, args, kwargs):
            calls.append({k: v.detach().clone() for k, v in kwargs.items() if isinstance(v, torch.Tensor)})
        handle = model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with patch('torch.randn_like', return_value=noise), patch(
                    'diffusers.training_utils.compute_density_for_timestep_sampling', return_value=torch.zeros(1)):
                flow_matching_loss(model, target, refs, embeddings, text_position_ids(3, 'cpu'), scheduler)
            pipeline = build_pipeline({'transformer': model, 'vae': vae, 'scheduler': scheduler}, 'cpu')
            with torch.no_grad():
                pipeline(image=images, prompt_embeds=embeddings, negative_prompt_embeds=embeddings,
                         height=48, width=32, latents=noise, num_inference_steps=1, output_type='latent', output='images')
        finally:
            handle.remove()
            encoder_hook.remove()
            quant_hook.remove()
        self.assertEqual(len(vae_inputs), 6)
        for key, value in vae.state_dict().items():
            torch.testing.assert_close(value, initial_vae[key], rtol=0, atol=0, msg=lambda m: f'{key}: {m}')
        self.assertTrue(all(not record[1] and not record[2] for record in vae_debug))
        for index in range(3):
            torch.testing.assert_close(vae_inputs[index], vae_inputs[index + 3], rtol=0, atol=0)
            torch.testing.assert_close(vae_outputs[index], vae_outputs[index + 3], rtol=1e-4, atol=1e-5,
                                       msg=lambda message: f'raw VAE output: {message}')
        self.assertGreaterEqual(len(calls), 2)
        for call in calls[1:]:
            for key in ('hidden_states', 'img_ids', 'timestep', 'image_cond_hidden_states',
                        'image_cond_ids', 'encoder_hidden_states', 'txt_ids'):
                torch.testing.assert_close(calls[0][key], call[key], rtol=1e-4, atol=1e-5,
                                           msg=lambda message: f'{key}: {message}')

    def test_zero_initial_lora_delta_preserves_base_prediction(self):
        from test_flux3_recolor_training import tiny_transformer
        from flux3_recolor.modeling import add_lora, reference_tokens, image_position_ids, text_position_ids
        torch.manual_seed(35)
        model = tiny_transformer().eval()
        refs, ids = reference_tokens([torch.randn(1, 8, 2, 2), torch.randn(1, 8, 2, 2), torch.randn(1, 8, 1, 1)])
        inputs = dict(hidden_states=torch.randn(1, 4, 8), img_ids=image_position_ids(2, 2, 'cpu'),
                      timestep=torch.tensor([0.4]), image_cond_hidden_states=refs, image_cond_ids=ids,
                      encoder_hidden_states=torch.randn(1, 3, 12), txt_ids=text_position_ids(3, 'cpu'), return_dict=False)
        with torch.no_grad():
            expected = model(**inputs)[0]
        add_lora(model, rank=2, alpha=6, dropout=0)
        self.assertTrue(all(torch.count_nonzero(p) == 0 for n, p in model.named_parameters() if 'lora_B' in n))
        self.assertTrue(all(torch.count_nonzero(p) > 0 for n, p in model.named_parameters() if 'lora_A' in n))
        with torch.no_grad():
            torch.testing.assert_close(model(**inputs)[0], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
