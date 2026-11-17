import tempfile
import unittest
from pathlib import Path


class ValidationRunTests(unittest.TestCase):
    def test_same_step_and_settings_render_once_but_changes_render_again(self):
        from flux3_recolor.validation import ValidationRuns
        runs, calls = ValidationRuns(), []
        def run(step=100, prompt='swatch', seed=42, steps=8, ids=('a',)):
            return runs.run(step, {'prompt': prompt}, ids, seed, steps,
                            lambda: calls.append(prompt))
        run(); run(); run()  # step / epoch / final with identical settings
        self.assertEqual(len(calls), 1)
        run(prompt='different'); run(step=101); run(seed=43); run(steps=9); run(ids=('b',))
        self.assertEqual(len(calls), 6)
        # A resumed process has new weights: never reuse previous-run completion.
        ValidationRuns().run(100, {'prompt': 'swatch'}, ('a',), 42, 8, lambda: calls.append('resume'))
        self.assertEqual(calls[-1], 'resume')

    def test_pipeline_reads_live_text_encoder_and_keeps_negative_condition(self):
        import torch
        from types import SimpleNamespace
        from flux3_recolor.text_encoder import TextEncoderModel
        from flux3_recolor.validation import pipeline_text_embeddings
        encoder = TextEncoderModel.from_features(torch.ones(1,3,4))
        pipeline = SimpleNamespace(recolor_text_encoder=encoder)
        original = (torch.zeros(1,3,4), torch.randn(1,2,4))
        with torch.no_grad():
            encoder.weight.add_(2)
        positive, negative = pipeline_text_embeddings(pipeline, original)
        self.assertTrue(torch.equal(positive, torch.full((1,3,4),3.)))
        self.assertFalse(positive.requires_grad)
        self.assertIs(negative, original[1])
        self.assertIs(pipeline_text_embeddings(SimpleNamespace(), original), original)

    def test_failed_render_can_retry(self):
        from flux3_recolor.validation import ValidationRuns
        runs = ValidationRuns()
        with self.assertRaises(RuntimeError):
            runs.run(1, {}, (), 1, 1, lambda: (_ for _ in ()).throw(RuntimeError('failed')))
        calls = []
        runs.run(1, {}, (), 1, 1, lambda: calls.append(True))
        self.assertEqual(calls, [True])

    def test_output_replaces_stale_files_only_after_success(self):
        from flux3_recolor.validation import validation_output
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'validation-step-100'
            dest.mkdir()
            (dest / 'old.png').write_bytes(b'old')
            with self.assertRaises(RuntimeError):
                with validation_output(dest, protected=[]) as staging:
                    (staging / 'new.png').write_bytes(b'partial')
                    raise RuntimeError('failed')
            self.assertEqual((dest / 'old.png').read_bytes(), b'old')
            with validation_output(dest, protected=[]) as staging:
                (staging / 'new.png').write_bytes(b'new')
            self.assertEqual(list(p.name for p in dest.iterdir()), ['new.png'])
            self.assertEqual([p.name for p in Path(tmp).iterdir()], [dest.name])

    def test_output_cannot_replace_inputs(self):
        from flux3_recolor.validation import validation_output
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                with validation_output(tmp, protected=[tmp]):
                    self.fail('must reject input overlap')


if __name__ == '__main__':
    unittest.main()
