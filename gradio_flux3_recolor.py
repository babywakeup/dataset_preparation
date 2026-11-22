"""FLUX.3 换色 LoRA / Text Encoder 推理 Gradio 交互界面。"""

import argparse
import glob
import re
import secrets
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageColor

from flux3_recolor.component_io import load_component_config
from flux3_recolor.config import write_json
from flux3_recolor.inference import generate_sample, load_recolor_pipeline


PAINT_MODE = '单图画笔（推荐）'
UPLOAD_MODE = '三图上传（高级）'


# 全局模型缓存状态
CURRENT_MODEL = {
    'pipeline': None,
    'model_dir': None,
    'base_model': None,
    'device': None,
    'mixed_precision': None,
    'guidance_scale': None,
}


def find_available_checkpoints() -> list[str]:
    """搜索 outputs 目录下所有可用的 checkpoint / 导出模型。"""
    candidates = []
    for pattern in ['outputs/**/checkpoint-*', 'outputs/**/model_config.json']:
        for p in glob.glob(pattern, recursive=True):
            path = Path(p)
            target = path.parent if path.is_file() else path
            if (target / 'model_config.json').is_file():
                resolved = str(target.resolve())
                if resolved not in candidates:
                    candidates.append(resolved)
    candidates.sort(reverse=True)
    return candidates


def load_model_if_needed(
    model_dir: str,
    base_model_override: Optional[str] = None,
    device: str = 'cuda',
    mixed_precision: str = 'bf16',
    guidance_scale: float = 3.5,
) -> Tuple[object, str]:
    """加载或复用已缓存的模型 pipeline。"""
    global CURRENT_MODEL

    model_dir_path = Path(model_dir).expanduser().resolve()
    if not model_dir_path.exists():
        raise FileNotFoundError(f'模型目录不存在: {model_dir_path}')

    base_override = base_model_override.strip() if base_model_override and base_model_override.strip() else None

    # 检查是否已缓存且无需重载
    if (
        CURRENT_MODEL['pipeline'] is not None
        and CURRENT_MODEL['model_dir'] == str(model_dir_path)
        and CURRENT_MODEL['base_model'] == base_override
        and CURRENT_MODEL['device'] == device
        and CURRENT_MODEL['mixed_precision'] == mixed_precision
    ):
        pipe = CURRENT_MODEL['pipeline']
        # 仅动态更新 guidance_scale
        if CURRENT_MODEL['guidance_scale'] != guidance_scale:
            pipe.update_components(guider=pipe.guider.new(guidance_scale=guidance_scale))
            CURRENT_MODEL['guidance_scale'] = guidance_scale
        return pipe, f'复用已缓存模型: {model_dir_path.name}'

    # 重新加载
    start_time = time.time()
    manifest = load_component_config(model_dir_path)
    base = base_override or manifest['base_model']

    pipeline = load_recolor_pipeline(
        model_dir_path,
        pretrained_model_name_or_path=base_override,
        device=device,
        mixed_precision=mixed_precision,
        guidance_scale=guidance_scale,
    )

    CURRENT_MODEL['pipeline'] = pipeline
    CURRENT_MODEL['model_dir'] = str(model_dir_path)
    CURRENT_MODEL['base_model'] = base_override
    CURRENT_MODEL['device'] = device
    CURRENT_MODEL['mixed_precision'] = mixed_precision
    CURRENT_MODEL['guidance_scale'] = guidance_scale

    elapsed = time.time() - start_time
    text_encoder_desc = (
        f'learned (shape={tuple(pipeline.recolor_text_encoder.weight.shape)})'
        if manifest['text_encoder']['type'] == 'learned'
        else 'fixed prompt'
    )
    msg = (
        f'✅ 模型加载成功 (耗时 {elapsed:.2f}s)\n'
        f'• 基础模型: {base}\n'
        f'• 权重目录: {model_dir_path}\n'
        f'• LoRA: {manifest["transformer"]["path"]}\n'
        f'• Text Encoder: {text_encoder_desc}\n'
        f'• 设备/精度: {device} / {mixed_precision}'
    )
    return pipeline, msg


def make_swatch_from_color(color_hex: str) -> Image.Image:
    """接受取色器的 HEX / RGB / RGBA 值，生成不透明目标色块。"""
    color = (color_hex or '').strip()
    # 浏览器取色器可返回带小数 alpha 的 CSS rgba；色块只使用 RGB。
    rgba = re.fullmatch(r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\)', color)
    if rgba:
        color = f'rgb({rgba[1]}, {rgba[2]}, {rgba[3]})'
    try:
        rgb = ImageColor.getrgb(color)[:3]
        if any(not 0 <= channel <= 255 for channel in rgb):
            raise ValueError('RGB 越界')
    except (ValueError, TypeError) as exc:
        raise ValueError('请选择有效颜色或输入 #RRGGBB 色值') from exc
    return Image.new('RGB', (128, 128), rgb)


def prepare_painted_inputs(editor_value, require_strokes=False):
    """从原图及笔画 alpha 构建条件图；不把编辑器 composite 当作源图。"""
    if not editor_value or editor_value.get('background') is None:
        if require_strokes:
            raise ValueError('请先在画布上传一张原图')
        return None, None
    source = editor_value['background'].convert('RGB')
    mask = np.zeros((source.height, source.width), dtype=bool)
    for layer in editor_value.get('layers') or []:
        if layer is None:
            continue
        if layer.size != source.size:
            raise ValueError('画笔图层与原图尺寸不一致，请重新上传原图')
        mask |= np.asarray(layer.convert('RGBA').getchannel('A')) > 0
    if require_strokes and not mask.any():
        raise ValueError('请先用画笔涂抹需要换色的区域')
    gray = np.array(source)
    gray[mask] = (128, 128, 128)
    return Image.fromarray(gray), source


def generate_gray_image(source_img: Optional[Image.Image], mask_img: Optional[Image.Image]) -> Image.Image:
    """使用源图和黑白遮罩生成涂灰图 (遮罩白色区域涂为 128 灰色)。"""
    if source_img is None:
        raise ValueError('请先上传源图')
    if mask_img is None:
        raise ValueError('请上传黑白遮罩图')

    src = source_img.convert('RGB')
    mask = mask_img.convert('L')
    if mask.size != src.size:
        mask = mask.resize(src.size, Image.Resampling.NEAREST)

    # 遮罩白色 (>127) 区域为需要涂灰换色的区域
    mask_np = np.array(mask) > 127
    src_np = np.array(src)
    src_np[mask_np] = (128, 128, 128)
    return Image.fromarray(src_np)


def run_recolor(
    model_dir: str,
    base_model_override: str,
    device: str,
    mixed_precision: str,
    gray_image: Optional[Image.Image],
    source_image: Optional[Image.Image],
    swatch_image: Optional[Image.Image],
    seed: int,
    random_seed: bool,
    num_inference_steps: int,
    guidance_scale: float,
    save_output: bool,
    output_dir_prefix: str,
    editor_value=None,
    color_hex='#B83040',
    input_mode=UPLOAD_MODE,
    progress=gr.Progress(track_tqdm=True),
):
    """执行换色推理主流程。"""
    if input_mode == PAINT_MODE:
        try:
            # 生成时直接读取最新笔画和颜色，避免依赖异步预览的旧值。
            gray_image, source_image = prepare_painted_inputs(editor_value, require_strokes=True)
            swatch_image = make_swatch_from_color(color_hex)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
    if gray_image is None or source_image is None or swatch_image is None:
        raise gr.Error('必须同时提供三张条件图像：涂灰图、源图、目标色块！')

    if random_seed or seed < 0:
        actual_seed = secrets.randbelow(2**32)
    else:
        actual_seed = int(seed)

    progress(0.1, desc='检查/加载模型...')
    try:
        pipeline, model_msg = load_model_if_needed(
            model_dir=model_dir,
            base_model_override=base_model_override,
            device=device,
            mixed_precision=mixed_precision,
            guidance_scale=guidance_scale,
        )
    except Exception as exc:
        raise gr.Error(f'模型加载失败: {exc}')

    progress(0.3, desc='预处理参考图像...')
    conditions = [
        gray_image.convert('RGB'),
        source_image.convert('RGB'),
        swatch_image.convert('RGB'),
    ]

    progress(0.4, desc='生成中 (Inference)...')
    start_time = time.time()
    try:
        generated, references = generate_sample(
            pipeline=pipeline,
            conditions=conditions,
            config=pipeline.recolor_config,
            seed=actual_seed,
            steps=int(num_inference_steps),
        )
    except Exception as exc:
        raise gr.Error(f'生成失败: {exc}')

    elapsed = time.time() - start_time
    progress(0.9, desc='合成对比图...')

    # 生成对比图 (按顺序: 涂灰图, 源图, 色块, 生成图)
    width, height = generated.size
    images = [*references, generated]
    canvas = Image.new('RGB', (width * len(images), height), 'white')
    for index, image in enumerate(images):
        tile = image.copy()
        tile.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.paste(tile, (width * index + (width - tile.width) // 2, (height - tile.height) // 2))

    save_info = '未保存至磁盘'
    if save_output:
        ts = int(time.time())
        out_path = Path(output_dir_prefix.strip() or 'outputs/gradio_recolor') / f'{ts}_{actual_seed}'
        out_path.mkdir(parents=True, exist_ok=True)
        generated.save(out_path / 'generated.png')
        canvas.save(out_path / 'comparison.png')
        for i, ref in enumerate(references, 1):
            ref.save(out_path / f'condition_{i}.png')
        manifest = load_component_config(Path(model_dir).expanduser().resolve())
        write_json(
            out_path / 'run_config.json',
            {
                'model_dir': str(model_dir),
                'base_model': manifest['base_model'],
                'seed': actual_seed,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': device,
                'mixed_precision': mixed_precision,
                'condition_order': ['gray', 'source', 'swatch'],
            },
        )
        save_info = f'已保存至: `{out_path}`'

    log_markdown = (
        f'### 🚀 推理完成 (耗时: {elapsed:.2f}s)\n\n'
        f'- **实际使用 Seed**: `{actual_seed}`\n'
        f'- **生成分辨率**: `{width} x {height}`\n'
        f'- **推理步数 / 引导尺度**: `{num_inference_steps}` 步 / `{guidance_scale}`\n'
        f'- **保存状态**: {save_info}\n'
    )

    return (
        generated,
        canvas,
        references[0],
        references[1],
        references[2],
        log_markdown,
        actual_seed,
    )


def create_ui(default_model_dir: str):
    """构建 Gradio 界面。"""
    available_ckpts = find_available_checkpoints()
    if default_model_dir not in available_ckpts:
        available_ckpts.insert(0, default_model_dir)

    with gr.Blocks(title='FLUX.3 换色模型交互体验') as demo:
        gr.Markdown(
            """
            # 🎨 FLUX.3 三参考图换色交互界面 (Gradio Demo)
            通过 **涂灰图 (Condition 1)**、**源图 (Condition 2)** 与 **目标色块 (Condition 3)** 驱动 FLUX.3 LoRA 进行精准区域换色。
            """
        )

        with gr.Row():
            # 左侧：配置与输入
            with gr.Column(scale=5):
                with gr.Accordion('⚙️ 模型配置与环境 (Model Settings)', open=True):
                    model_dir_input = gr.Dropdown(
                        label='模型权重目录 (model_dir: Checkpoint 或最终导出目录)',
                        choices=available_ckpts,
                        value=default_model_dir,
                        allow_custom_value=True,
                    )
                    base_model_override = gr.Textbox(
                        label='基础模型路径覆盖 (可选，留空则自动读取配置文件记录的基模)',
                        placeholder='留空自动使用训练时记录的基础模型',
                        value='',
                    )
                    with gr.Row():
                        device_choice = gr.Dropdown(label='推理设备', choices=['cuda', 'cpu'], value='cuda')
                        precision_choice = gr.Dropdown(
                            label='计算精度', choices=['bf16', 'fp16', 'no'], value='bf16'
                        )
                    load_btn = gr.Button('🔄 预加载 / 切换模型', variant='secondary')
                    model_status = gr.Markdown('状态：就绪，点击生成时将自动加载。')

                with gr.Accordion('🖼️ 三参考图输入 (Condition Images)', open=True):
                    gr.Markdown(
                        """
                        **只需一张原图**：上传 → 用画笔涂抹衣物区域 → 选择目标颜色 → 开始换色。
                        笔画区域自动填充为 (128,128,128)，未涂抹的原图作为源图保留。
                        可调整笔刷大小，用橡皮擦修正；清空后可重新上传。
                        """
                    )
                    input_mode = gr.Radio(
                        choices=[PAINT_MODE, UPLOAD_MODE], value=PAINT_MODE, label='输入方式',
                    )
                    editor = gr.ImageEditor(
                        label='上传原图，并用灰色画笔涂抹需要换色的区域',
                        type='pil', image_mode='RGBA', sources=['upload'], interactive=True,
                        brush=gr.Brush(colors=['#808080'], default_color='#808080',
                                       color_mode='fixed', default_size=30),
                        eraser=gr.Eraser(default_size=30), transforms=[],
                        format='png', height=480, elem_id='recolor-paint-editor',
                    )
                    color_picker = gr.ColorPicker(
                        label='目标颜色（点击色块自选，或直接输入 HEX / RGB 色值）',
                        value='#B83040', interactive=True, elem_id='recolor-target-color',
                    )
                    gr.Markdown('下方实时预览自动生成的三张条件图。高级模式下可直接上传三图。')
                    with gr.Row():
                        gray_in = gr.Image(label='1. 涂灰图 (Gray Image)', type='pil', height=260,
                                                                                     elem_id='recolor-gray-preview', format='png')
                        source_in = gr.Image(label='2. 源图 (Source Image)', type='pil', height=260,
                                                                                         elem_id='recolor-source-preview', format='png')
                        swatch_in = gr.Image(label='3. 目标色块 (Swatch)', type='pil', height=260,
                                             value=make_swatch_from_color('#B83040'),
                                                                                         elem_id='recolor-swatch-preview', format='png')

                    with gr.Accordion('高级：从已有黑白遮罩生成涂灰图', open=False):
                        with gr.Tab('从遮罩生成涂灰图'):
                            mask_in = gr.Image(label='黑白遮罩 (白色为换色区域)', type='pil', height=200)
                            make_gray_btn = gr.Button('源图 + 遮罩 自动生成涂灰图')

                with gr.Accordion('🎛️ 推理超参数 (Inference Hyperparameters)', open=True):
                    with gr.Row():
                        steps_slider = gr.Slider(
                            label='推理步数 (num_inference_steps)', minimum=1, maximum=50, value=28, step=1
                        )
                        guidance_slider = gr.Slider(
                            label='CFG 引导尺度 (guidance_scale)', minimum=1.0, maximum=10.0, value=3.5, step=0.1
                        )
                    with gr.Row():
                        seed_input = gr.Number(label='随机种子 (Seed)', value=42, precision=0)
                        random_seed_cb = gr.Checkbox(label='🎲 每次随机 Seed', value=False)
                        rand_seed_btn = gr.Button('🎲 刷新种子')

                    with gr.Row():
                        save_output_cb = gr.Checkbox(label='保存结果至本地磁盘', value=True)
                        output_prefix = gr.Textbox(label='保存目录前缀', value='outputs/gradio_recolor')

                run_btn = gr.Button('🚀 开始换色推理 (Generate)', variant='primary', size='lg')

            # 右侧：输出结果展示
            with gr.Column(scale=5):
                with gr.Tab('✨ 生成换色结果'):
                    gen_out = gr.Image(label='换色结果 (Generated)', type='pil', height=480)
                with gr.Tab('🔍 四图对比视图'):
                    comp_out = gr.Image(
                        label='对比图 (涂灰图 | 源图 | 目标色块 | 生成结果)', type='pil', height=360
                    )
                with gr.Tab('📐 预处理对齐预览'):
                    gr.Markdown('模型内部实际对齐与补齐后的三图尺寸：')
                    with gr.Row():
                        ref1_out = gr.Image(label='对齐后涂灰图', type='pil', height=200)
                        ref2_out = gr.Image(label='对齐后源图', type='pil', height=200)
                        ref3_out = gr.Image(label='对齐后色块', type='pil', height=200)

                log_box = gr.Markdown('### 状态：等待提交推理任务...')

        # 示例区域
        example_gray = 'test/经过涂灰的原图AB/00019_00.png'
        example_source = 'test/生成图B/00019_00.png'
        example_swatch = 'test/主颜色A/00019_00.png'
        if Path(example_gray).exists() and Path(example_source).exists() and Path(example_swatch).exists():
            gr.Examples(
                examples=[
                    [
                        default_model_dir,
                        example_gray,
                        example_source,
                        example_swatch,
                        42,
                        28,
                        3.5,
                        UPLOAD_MODE,
                    ]
                ],
                inputs=[
                    model_dir_input,
                    gray_in,
                    source_in,
                    swatch_in,
                    seed_input,
                    steps_slider,
                    guidance_slider,
                    input_mode,
                ],
                label='💡 点击加载默认测试示例 (00019_00)',
            )

        # 交互事件绑定
        def on_load_model(m_dir, b_override, dev, prec, g_scale):
            try:
                _, msg = load_model_if_needed(m_dir, b_override, dev, prec, g_scale)
                return msg
            except Exception as e:
                return f'❌ 加载失败: {e}'

        load_btn.click(
            fn=on_load_model,
            inputs=[model_dir_input, base_model_override, device_choice, precision_choice, guidance_slider],
            outputs=[model_status],
        )

        gr.on(
            triggers=[editor.upload, editor.input, editor.clear],
            fn=prepare_painted_inputs,
            inputs=[editor], outputs=[gray_in, source_in],
            trigger_mode='always_last',
        )

        color_picker.change(
            fn=make_swatch_from_color,
            inputs=[color_picker],
            outputs=[swatch_in],
            trigger_mode='always_last',
        )

        make_gray_btn.click(
            fn=generate_gray_image,
            inputs=[source_in, mask_in],
            outputs=[gray_in],
        )

        rand_seed_btn.click(
            fn=lambda: secrets.randbelow(2**32),
            outputs=[seed_input],
        )

        run_btn.click(
            fn=run_recolor,
            inputs=[
                model_dir_input,
                base_model_override,
                device_choice,
                precision_choice,
                gray_in,
                source_in,
                swatch_in,
                seed_input,
                random_seed_cb,
                steps_slider,
                guidance_slider,
                save_output_cb,
                output_prefix,
                editor,
                color_picker,
                input_mode,
            ],
            outputs=[
                gen_out,
                comp_out,
                ref1_out,
                ref2_out,
                ref3_out,
                log_box,
                seed_input,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_checkpoint = 'outputs/flux3_task_embeddingnew5/checkpoint-600'
    parser.add_argument('--model_dir', default=default_checkpoint, help='默认权重目录')
    parser.add_argument('--server_name', default='0.0.0.0', help='监听地址')
    parser.add_argument('--server_port', type=int, default=7860, help='端口号')
    parser.add_argument('--share', action='store_true', help='是否生成 Gradio 公网分享链接')
    args = parser.parse_args()

    demo = create_ui(default_model_dir=str(Path(args.model_dir).resolve()))
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == '__main__':
    main()
