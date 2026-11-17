import unittest
from types import SimpleNamespace
import torch


class FixedPrecisionTests(unittest.TestCase):
    def test_pipeline_forward_temporarily_bypasses_accelerate_wrapper(self):
        from flux3_recolor.precision import pipeline_forward
        model = torch.nn.Linear(4, 4).to(torch.bfloat16)
        original = model.forward
        model._original_forward = original
        model.forward = lambda x: original(x).float()
        wrapped = model.forward
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with pipeline_forward(model):
                self.assertEqual(model(torch.ones(1, 4, dtype=torch.bfloat16)).dtype, torch.bfloat16)
                raise RuntimeError('test')
        self.assertIs(model.forward, wrapped)
        self.assertEqual(model._original_forward, original)

    def test_shared_pipeline_is_cached_and_sees_parameter_updates(self):
        from flux3_recolor_test_utils import make_tiny_repository
        from flux3_recolor.modeling import load_components
        from train_flux3_recolor_lora import make_validation_pipeline
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            components = load_components(make_tiny_repository(Path(tmp) / 'model'), torch.float32, 'cpu')
            model = components['transformer']
            accelerator = SimpleNamespace(device=torch.device('cpu'), unwrap_model=lambda m: m)
            a = make_validation_pipeline(SimpleNamespace(), components, model, accelerator, torch.float32, False)
            b = make_validation_pipeline(SimpleNamespace(), components, model, accelerator, torch.float32, False)
            self.assertIs(a, b)
            self.assertIs(a.transformer, model)
            p = next(model.parameters())
            with torch.no_grad():
                p.add_(1)
            self.assertIs(next(a.transformer.parameters()), p)
            torch.testing.assert_close(next(a.transformer.parameters()), p, rtol=0, atol=0)

    def test_vae_retains_original_fp32_weights_modules_and_forward(self):
        from diffusers import AutoencoderKLFlux2
        from flux3_recolor.modeling import load_components, encode_latents
        from flux3_recolor_test_utils import make_tiny_repository
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = make_tiny_repository(Path(tmp)/'model')
            original = AutoencoderKLFlux2.from_pretrained(root, subfolder='vae', torch_dtype=torch.float32)
            components = load_components(root, torch.bfloat16, 'cpu')
            vae = components['vae']
            self.assertEqual(vae.dtype, torch.float32)
            self.assertIs(vae.forward.__func__, original.forward.__func__)
            self.assertNotIn('forward', vae.__dict__)
            self.assertFalse(hasattr(vae, '_recolor_precision_recipe'))
            for name, module in vae.named_modules():
                self.assertIs(type(module), type(original.get_submodule(name)))
            for name, value in vae.state_dict().items():
                torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
            pixels = torch.randn(1,3,32,32)
            expected = encode_latents(vae, pixels)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = encode_latents(vae, pixels)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(all(type(m) is torch.nn.LayerNorm for m in components['transformer'].modules()
                                if isinstance(m, torch.nn.LayerNorm)))

    def test_bf16_pipeline_keeps_vae_encode_and_decode_in_fp32(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from flux3_recolor.modeling import load_components
        from flux3_recolor.inference import build_pipeline, generate_sample
        from flux3_recolor_test_utils import make_tiny_repository
        with tempfile.TemporaryDirectory() as tmp:
            components = load_components(make_tiny_repository(Path(tmp)/'model'), torch.bfloat16, 'cpu')
            observations = []
            def record(module, inputs):
                observations.append((inputs[0].dtype, torch.is_autocast_enabled('cpu')))
            vae = components['vae']
            handles = [vae.encoder.conv_in.register_forward_pre_hook(record),
                       vae.decoder.conv_in.register_forward_pre_hook(record)]
            try:
                pipeline = build_pipeline(components, 'cpu')
                generate_sample(pipeline, [Image.new('RGB',(32,32),'gray')]*2 +
                    [Image.new('RGB',(128,128),'red')], {'max_area':1024,'preprocessing':
                    {'max_area':1024,'color_swatch_size':128}}, 42, 1,
                    [torch.zeros(1,8,16,dtype=torch.bfloat16)]*2)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertGreaterEqual(len(observations), 2)
            self.assertTrue(all(dtype == torch.float32 and not amp for dtype, amp in observations))

    def test_flow_retains_fp32_noise_labels_loss_and_explicit_forward_dtype(self):
        from unittest.mock import patch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from flux3_recolor.modeling import flow_matching_loss
        target = torch.randn(1, 2, 2, 2)
        noise = torch.randn_like(target)
        sigma = FlowMatchEulerDiscreteScheduler().sigmas[500]
        parameter = torch.nn.Parameter(torch.zeros(1))
        def model(**kw):
            self.assertEqual(kw['hidden_states'].dtype, torch.bfloat16)
            self.assertEqual(kw['image_cond_hidden_states'].dtype, torch.bfloat16)
            self.assertEqual(kw['timestep'].dtype, torch.bfloat16)
            expected = ((1 - sigma) * target + sigma * noise).flatten(2).transpose(1, 2).to(torch.bfloat16)
            torch.testing.assert_close(kw['hidden_states'], expected, rtol=0, atol=0)
            return (torch.zeros_like(kw['hidden_states']) + parameter,)
        with patch('torch.randn_like', return_value=noise), patch(
                'diffusers.training_utils.compute_density_for_timestep_sampling', return_value=torch.tensor([.5])):
            loss = flow_matching_loss(model, target, [target] * 3, torch.zeros(1, 2, 2),
                                      torch.zeros(2, 4), FlowMatchEulerDiscreteScheduler(), compute_dtype=torch.bfloat16)
        torch.testing.assert_close(loss, (noise - target).square().mean(), rtol=0, atol=0)
        loss.backward()
        self.assertIsNotNone(parameter.grad)

    def test_updated_fp32_lora_computes_identically_inside_and_outside_amp(self):
        from flux3_recolor.precision import set_transformer_autocast
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4)).to(torch.bfloat16),
                               LoraConfig(r=2, target_modules=['0']))
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = torch.randn_like(p.float())
        import inspect
        original_signature = inspect.signature(model.forward)
        set_transformer_autocast(model, torch.bfloat16)
        self.assertEqual(inspect.signature(model.forward), original_signature)
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        expected = model(x)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().sum().backward()
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    unittest.main()
