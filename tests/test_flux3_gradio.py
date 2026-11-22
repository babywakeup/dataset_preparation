"""CPU interaction contracts; no model weights are loaded."""

import io
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import gradio_flux3_recolor as demo


class GradioInputTests(unittest.TestCase):
    def editor_value(self):
        background = Image.new('RGBA', (8, 6), (20, 40, 60, 255))
        layer = Image.new('RGBA', background.size, (0, 0, 0, 0))
        layer.putpixel((3, 2), (128, 128, 128, 32))
        return {'background': background, 'layers': [layer], 'composite': background.copy()}

    def test_brush_alpha_marks_gray_without_modifying_original(self):
        value = self.editor_value()
        before = np.array(value['background']).copy()
        gray, source = demo.prepare_painted_inputs(value)
        self.assertEqual(source.size, (8, 6))
        self.assertEqual(gray.getpixel((3, 2)), (128, 128, 128))
        self.assertEqual(gray.getpixel((0, 0)), (20, 40, 60))
        self.assertEqual(source.getpixel((3, 2)), (20, 40, 60))
        np.testing.assert_array_equal(np.array(value['background']), before)

    def test_empty_clear_and_erased_layers(self):
        self.assertEqual(demo.prepare_painted_inputs(None), (None, None))
        value = self.editor_value()
        value['layers'] = [Image.new('RGBA', (8, 6))]
        gray, source = demo.prepare_painted_inputs(value)
        np.testing.assert_array_equal(np.array(gray), np.array(source))
        with self.assertRaisesRegex(ValueError, '画笔'):
            demo.prepare_painted_inputs(value, require_strokes=True)

    def test_multiple_layers_and_layer_dimensions(self):
        value = self.editor_value()
        layer = Image.new('RGBA', (8, 6))
        layer.putpixel((1, 1), (255, 0, 0, 255))
        value['layers'].append(layer)
        gray, _ = demo.prepare_painted_inputs(value)
        self.assertEqual(gray.getpixel((1, 1)), (128, 128, 128))
        self.assertEqual(gray.getpixel((3, 2)), (128, 128, 128))
        value['layers'] = [Image.new('RGBA', (4, 3), (0, 0, 0, 255))]
        with self.assertRaisesRegex(ValueError, '尺寸'):
            demo.prepare_painted_inputs(value)

    def test_custom_colors(self):
        for color, expected in [('#12abEF', (18, 171, 239)), ('#abc', (170, 187, 204)),
                                ('rgb(1, 2, 3)', (1, 2, 3)),
                                ('rgba(11, 22, 33, 0.5)', (11, 22, 33))]:
            with self.subTest(color=color):
                swatch = demo.make_swatch_from_color(color)
                self.assertEqual(swatch.size, (128, 128))
                self.assertEqual(swatch.getpixel((0, 0)), expected)
        with self.assertRaises(ValueError):
            demo.make_swatch_from_color('not a color')

    def test_generation_reads_current_editor_and_color_not_stale_preview(self):
        pipe = SimpleNamespace(recolor_config={})
        def generate(**kwargs):
            refs = kwargs['conditions']
            self.assertEqual(refs[0].getpixel((3, 2)), (128, 128, 128))
            self.assertEqual(refs[1].getpixel((3, 2)), (20, 40, 60))
            self.assertEqual(refs[2].getpixel((0, 0)), (18, 52, 86))
            return refs[1].copy(), refs
        with patch.object(demo, 'load_model_if_needed', return_value=(pipe, 'cached')), \
                patch.object(demo, 'generate_sample', side_effect=generate) as mocked:
            result = demo.run_recolor('unused', '', 'cpu', 'no', None, None, None,
                                      42, False, 1, 3.5, False, '',
                                      editor_value=self.editor_value(), color_hex='#123456',
                                      input_mode=demo.PAINT_MODE, progress=lambda *a, **k: None)
        mocked.assert_called_once()
        self.assertEqual(result[-1], 42)

    def test_ui_has_upload_brush_color_and_current_input_bindings(self):
        ui = demo.create_ui('outputs/flux3_task_embeddingnew5/checkpoint-600')
        config = ui.get_config_file()
        components = {c['id']: c for c in config['components']}
        editor = next(c for c in components.values() if c['type'] == 'imageeditor')
        props = editor['props']
        self.assertEqual(props['sources'], ['upload'])
        self.assertTrue(props['interactive'])
        self.assertEqual(props['brush']['color_mode'], 'fixed')
        self.assertEqual(props['transforms'], [])
        picker = next(c for c in components.values() if c['type'] == 'colorpicker')
        self.assertTrue(picker['props']['interactive'])
        run = next(d for d in config['dependencies'] if d.get('api_name') == 'run_recolor')
        self.assertIn(editor['id'], run['inputs'])
        self.assertIn(picker['id'], run['inputs'])


@unittest.skipUnless(os.environ.get('RUN_GRADIO_BROWSER_TESTS') == '1',
                     'Set RUN_GRADIO_BROWSER_TESTS=1 for real Chromium interaction')
class GradioBrowserTests(unittest.TestCase):
    def test_upload_paint_pick_color_and_generate(self):
        from playwright.sync_api import sync_playwright, expect

        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        ui = demo.create_ui('outputs/flux3_task_embeddingnew5/checkpoint-600')
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs['conditions'])
            return kwargs['conditions'][1].copy(), kwargs['conditions']

        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.object(demo, 'load_model_if_needed', return_value=(SimpleNamespace(recolor_config={}), 'test')), \
                    patch.object(demo, 'generate_sample', side_effect=fake_generate):
                original = Path(tmp) / 'original.png'
                Image.new('RGB', (256, 192), (20, 40, 60)).save(original)
                ui.queue().launch(server_name='127.0.0.1', server_port=port,
                                  prevent_thread_lock=True, quiet=True)
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
                    try:
                        page = browser.new_page(viewport={'width': 1600, 'height': 1200})
                        page.goto(f'http://127.0.0.1:{port}')
                        page.wait_for_load_state('networkidle')
                        editor = page.locator('#recolor-paint-editor')
                        editor.locator('input[type=file]').set_input_files(str(original))
                        gray_preview = page.locator('#recolor-gray-preview img')
                        expect(gray_preview).to_be_visible(timeout=30000)
                        before = gray_preview.get_attribute('src')
                        editor.get_by_role('button', name='Brush', exact=True).click()
                        canvas = editor.locator('canvas').last
                        canvas.scroll_into_view_if_needed()
                        box = canvas.bounding_box()
                        page.mouse.move(box['x'] + box['width'] * .45, box['y'] + box['height'] * .5)
                        page.mouse.down()
                        page.mouse.move(box['x'] + box['width'] * .55, box['y'] + box['height'] * .5, steps=12)
                        page.mouse.up()
                        expect(gray_preview).not_to_have_attribute('src', before, timeout=30000)
                        response = page.request.get(gray_preview.get_attribute('src'))
                        gray = np.array(Image.open(io.BytesIO(response.body())).convert('RGB'))
                        self.assertTrue(np.all(gray == (128, 128, 128), axis=-1).any())
                        self.assertTrue(np.all(gray == (20, 40, 60), axis=-1).any())
                        picker = page.locator('#recolor-target-color')
                        picker.locator('button').first.click()
                        hex_input = picker.locator('input[type=text]')
                        hex_input.fill('#123456')
                        hex_input.press('Tab')
                        swatch_preview = page.locator('#recolor-swatch-preview img')
                        # Wait for the backend preview rather than only the CSS picker color.
                        expect(picker.locator('button').first).to_have_css('background-color', 'rgb(18, 52, 86)')
                        page.wait_for_function("""async () => {
                            const img = document.querySelector('#recolor-swatch-preview img');
                            if (!img || !img.complete) return false;
                            const c = document.createElement('canvas'); c.width = c.height = 1;
                            c.getContext('2d').drawImage(img, 0, 0, 1, 1);
                            const p = c.getContext('2d').getImageData(0,0,1,1).data;
                            return p[0] === 18 && p[1] === 52 && p[2] === 86;
                        }""", timeout=30000)
                        response = page.request.get(swatch_preview.get_attribute('src'))
                        self.assertEqual(Image.open(io.BytesIO(response.body())).getpixel((0, 0))[:3], (18, 52, 86))
                        checkbox = page.get_by_label('保存结果至本地磁盘', exact=True)
                        if not checkbox.is_visible():
                            page.get_by_text('🎛️ 推理超参数 (Inference Hyperparameters)', exact=True).click()
                        checkbox.uncheck()
                        page.get_by_role('button', name='🚀 开始换色推理 (Generate)', exact=True).click()
                        expect(page.get_by_text('🚀 推理完成', exact=False)).to_be_visible(timeout=30000)
                        self.assertEqual(len(calls), 1)
                        np.testing.assert_array_equal(np.array(calls[0][0]), gray)
                        self.assertEqual(calls[0][1].getpixel((128, 96)), (20, 40, 60))
                        self.assertEqual(calls[0][2].getpixel((0, 0)), (18, 52, 86))
                    finally:
                        browser.close()
        finally:
            ui.close()


if __name__ == '__main__':
    unittest.main()
