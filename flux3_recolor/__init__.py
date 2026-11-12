"""Standalone FLUX.3 three-reference recoloring tools (no Studio dependency)."""

DEFAULT_PROMPT = (
    "Use image 1 as the base image. Recolor the gray clothing region using the color swatch in image 3. "
    "Use image 2 as a reference for garment structure, texture, and shading. "
    "Preserve the person, pose, garment details, and background."
)
CONDITION_FOLDERS = ('经过涂灰的原图AB', '生成图B', '主颜色A')
TARGET_FOLDER = '原图A'
