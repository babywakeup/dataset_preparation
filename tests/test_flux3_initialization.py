import io
from contextlib import redirect_stdout
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flux3_recolor_test_utils import make_tiny_repository
        cls.shared = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.shared.cleanup)
        cls.model = make_tiny_repository(Path(cls.shared.name)/'model')

    def setUp(self):
        from flux3_recolor_test_utils import make_dataset
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = make_dataset(self.root/'data')
        self.options = ['--instance_data_dir', str(self.data), '--mixed_precision','no',
            '--resolution','32','--max_sequence_length','160','--text_encoder_out_layers','1',
            '--training_directions','b_to_a','--train_batch_size','1','--report_to','none',
            '--num_validation_images','0','--checkpointing_steps','1','--logging_steps','1']

    def source(self, learned=False):
        from train_flux3_recolor_lora import main
        path = self.root/'source'
        main([*self.options,'--pretrained_model_name_or_path',str(self.model),'--output_dir',str(path),
              '--rank','2','--lora_alpha','6','--lora_layers','to_q,to_v','--max_train_steps','2',
              *(['--train_text_encoder'] if learned else [])])
        return path

    def test_new_round_inherits_weights_and_structure_but_resets_state(self):
        from safetensors.torch import load_file
        from peft.utils import get_peft_model_state_dict
        from train_flux3_recolor_lora import main
        from train_flux3_recolor_text_encoder import main as warm_train
        source = self.source(learned=True)/'checkpoint-2'
        expected = {k.removeprefix('transformer.'):v for k,v in
                    load_file(str(source/'transformer/pytorch_lora_weights.safetensors')).items()}
        feature = load_file(str(source/'text_encoder/model.safetensors'))['weight']
        output = self.root/'new-round'
        from flux3_recolor_test_utils import make_dataset
        new_data = make_dataset(self.root/'new-data', 9)
        from flux3_recolor.modeling import flow_matching_loss
        calls = []
        def check(model, target, references, embeds, *args, **kwargs):
            for name, value in get_peft_model_state_dict(model).items():
                torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
            torch.testing.assert_close(embeds.detach().cpu(), feature.expand(embeds.shape[0],-1,-1), rtol=0, atol=0)
            calls.append(True)
            return flow_matching_loss(model,target,references,embeds,*args,**kwargs)
        logs = io.StringIO()
        with redirect_stdout(logs), patch('flux3_recolor.modeling.flow_matching_loss', side_effect=check):
            result = warm_train([*self.options,'--init_from',str(source),'--output_dir',str(output),
                           '--instance_data_dir',str(new_data),'--learning_rate','0.00002','--max_train_steps','1'])
        for component in ('VAE', 'Transformer', 'Qwen text_encoder'):
            self.assertIn(f'[基础权重] 加载 {component}:', logs.getvalue())
        self.assertIn(f'[已训练权重] 已加载 Transformer LoRA: {source / "transformer"}', logs.getvalue())
        self.assertIn(f'[已训练权重] 已加载 text_encoder: {source / "text_encoder"}', logs.getvalue())
        self.assertEqual(len(calls),1)
        self.assertEqual(result['state']['global_step'],1)
        self.assertIsNotNone(result['text_encoder'])
        self.assertTrue(result['text_encoder'].training)
        config = json.loads((output/'recolor_config.json').read_text())
        self.assertEqual((config['rank'],config['lora_alpha']),(2,6))
        self.assertEqual(set(config['lora_layers'].split(',')), {'to_q','to_v'})
        state = torch.load(output/'checkpoint-1/optimizer.bin',map_location='cpu',weights_only=True)
        self.assertTrue(all(float(value['step']) == 1 for value in state['state'].values()))
        # Resume the new round without needing the original initialization source.
        import shutil
        shutil.rmtree(self.root/'source')
        resume_logs = io.StringIO()
        with redirect_stdout(resume_logs):
            result = main([*self.options,'--output_dir',str(output),'--resume_from_checkpoint','latest',
                           '--instance_data_dir',str(new_data),'--learning_rate','0.00002','--max_train_steps','2'])
        for component, folder in (('Transformer LoRA', 'transformer'), ('text_encoder', 'text_encoder')):
            self.assertIn(f'[已训练权重] 恢复 {component}: {output / "checkpoint-1" / folder}', resume_logs.getvalue())
        self.assertEqual(result['state']['global_step'],2)
        self.assertEqual(json.loads((output/'recolor_config.json').read_text())['init_from'],str(source))

    def test_lora_only_source_and_conflicting_options(self):
        from train_flux3_recolor_lora import main, parse_args
        source = self.source()
        result = main([*self.options,'--init_from',str(source),'--output_dir',str(self.root/'round2'),
                       '--max_train_steps','1'])
        self.assertIsNone(result['text_encoder'])
        for extra in [['--rank','4'], ['--lora_alpha','2'], ['--lora_layers','to_q'],
                      ['--resume_from_checkpoint','latest']]:
            with self.assertRaises((SystemExit,ValueError)):
                parse_args([*self.options,'--init_from',str(source),*extra])
        with self.assertRaisesRegex(ValueError,'重叠|非空'):
            main([*self.options,'--init_from',str(source),'--output_dir',str(source/'new'),
                  '--max_train_steps','1'])

    def test_fsdp_validation_proxy_preserves_inherited_adapter_scaling(self):
        from types import SimpleNamespace
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from diffusers import Flux3Transformer2DModel
        from train_flux3_recolor_lora import make_validation_pipeline, parse_args
        layer = 'mode_blocks.image.0.attn.to_q'
        adapter = LoraConfig(r=2,lora_alpha=6,target_modules=['to_q','to_v'],use_rslora=True,
                             rank_pattern={layer:3},alpha_pattern={layer:9})
        model = Flux3Transformer2DModel.from_pretrained(self.model,subfolder='transformer',torch_dtype=torch.float32)
        model.add_adapter(adapter)
        args = parse_args(['--pretrained_model_name_or_path',str(self.model),'--rank','2','--lora_alpha','6'])
        args._adapter_config = adapter.to_dict()
        accelerator = SimpleNamespace(device=torch.device('cpu'),unwrap_model=lambda value:value)
        with patch('flux3_recolor.inference.build_pipeline',side_effect=lambda components,device:
                   SimpleNamespace(transformer=components['transformer'])):
            pipeline = make_validation_pipeline(args,{},model,accelerator,torch.float32,True,
                                                get_peft_model_state_dict(model))
        self.assertTrue(pipeline.transformer.peft_config['default'].use_rslora)
        self.assertEqual(pipeline.transformer.get_submodule(layer).lora_A['default'].weight.shape[0],3)
        self.assertAlmostEqual(pipeline.transformer.get_submodule(layer).scaling['default'],9/(3**0.5))

    @unittest.skipUnless(__import__('os').environ.get('RUN_FLUX3_DDP_TESTS') == '1', 'opt-in CPU DDP initialization')
    def test_init_from_two_ranks_updates_loaded_components(self):
        import os, subprocess, sys
        from safetensors.torch import load_file
        source = self.source(learned=True)/'checkpoint-2'
        output = self.root/'ddp-new-round'
        command = [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
            'tests/flux3_reference_ddp_worker.py',*self.options,'--init_from',str(source),
            '--output_dir',str(output),'--max_train_steps','1','--gradient_accumulation_steps','2',
            '--num_validation_images','1','--validation_steps','1','--validation_inference_steps','1']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'ACCELERATE_USE_CPU':'true','OMP_NUM_THREADS':'1',
               'PYTHONPATH':str(Path.cwd())+':'+str(Path.cwd()/'tests'),'PYTHONDONTWRITEBYTECODE':'1'}
        log = self.root/'warm-ddp.log'
        with log.open('w') as stream:
            result = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        self.assertEqual(result.returncode,0,log.read_text()[-12000:])
        a,b = [load_file(str(output/f'rank-artifacts/rank-{rank}.safetensors')) for rank in (0,1)]
        self.assertIn('text_encoder.weight',a)
        for key in a:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        self.assertEqual(json.loads((output/'training_state.json').read_text())['global_step'],1)
        optimizer = torch.load(output/'checkpoint-1/optimizer.bin',map_location='cpu',weights_only=True)
        self.assertTrue(all(float(value['step']) == 1 for value in optimizer['state'].values()))

class ModuleConflictTests(unittest.TestCase):
    def test_conflict_lists_sources_and_module_differences(self):
        from types import SimpleNamespace
        from flux3_recolor.initialization import inherit_model_args
        args = SimpleNamespace(rank=None, lora_alpha=None, lora_dropout=None,
                               lora_layers='to_q, context_embedder')
        metadata = {'transformer.r': 2, 'transformer.lora_alpha': 2,
                    'transformer.target_modules': ['to_q', 'to_v']}
        with patch('flux3_recolor.initialization.load_component_config',
                   return_value={'transformer': {'path': 'transformer'}}), patch('safetensors.safe_open') as opened:
            opened.return_value.__enter__.return_value.metadata.return_value = {
                'lora_adapter_metadata': json.dumps(metadata)}
            with self.assertRaises(ValueError) as caught:
                inherit_model_args(args, '/tmp/lora-conflict-source')
        message = str(caught.exception)
        for expected in ('init_from (/tmp/lora-conflict-source): to_q, to_v',
                         'lora_layers: context_embedder, to_q',
                         '新增模块: context_embedder', '缺少模块: to_v'):
            self.assertIn(expected, message)
