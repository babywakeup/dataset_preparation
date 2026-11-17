"""Standalone FLUX.3 three-reference recoloring tools (no Studio dependency)."""

DEFAULT_PROMPT = (
    "Edit image 2, the photograph to be recolored. Image 1 is the corresponding photograph with only "
    "the editable clothing region covered in uniform gray. The region replaced by gray in image 1 compared "
    "with image 2 marks where to recolor; all other regions must stay unchanged, including naturally gray objects. "
    "Image 3 is the desired color swatch. Recolor only the marked clothing region to match image 3. "
    "Use image 2, not the gray fill in image 1, to preserve all original clothing details. "
    "Preserve the exact person's identity, pose, garment silhouette, seams, buttons, prints, logos, "
    "fabric weave, texture, folds, shadows, highlights, camera framing, and background. "
    "Change fabric color only, not material or geometry. Do not add objects, text, mask markings, "
    "or the swatch to the output. Return only the edited photograph."
)
_PRESERVE = (
    "Preserve the person's identity, pose, garment shape, seams, buttons, prints, logos, fabric texture, "
    "folds, shadows, highlights, framing, and background. Change fabric color only; keep naturally gray "
    "objects unchanged. Do not add objects, text, gray markings, or the swatch. Return only the edited photograph."
)
PROMPT_VARIANTS = (DEFAULT_PROMPT, *(
    instruction + ' ' + _PRESERVE for instruction in (
        "Image 1 marks the editable clothing with uniform gray. Image 2 is the photograph to edit, and "
        "image 3 is the target color swatch. Recolor image 2 only where image 1 differs by the gray fill. "
        "Match image 3 and retain the original clothing details from image 2.",
        "Change the clothing color in image 2 to match the swatch in image 3. Use the uniform gray region "
        "in image 1 to locate the clothing to change. Recover texture and lighting from image 2, not the gray fill.",
        "Use image 2 as the source photograph. Image 1 shows its editable clothing area painted gray; "
        "image 3 supplies the desired fabric color. Apply that color only to the marked area of image 2, "
        "preserving the underlying clothing details.",
        "Perform a localized clothing recolor on image 2. The area covered with uniform gray in image 1 "
        "defines the edit region; the color swatch in image 3 defines the new color. Keep the clothing "
        "texture and illumination of image 2 throughout the edit.",
        "Return an edited version of image 2 with only the selected garment recolored to the color of "
        "image 3. Identify the selected garment from the uniform gray replacement in image 1. "
        "Use the original garment details in image 2 rather than reproducing the gray fill.",
    )
))
CONDITION_FOLDERS = ('经过涂灰的原图AB', '生成图B', '主颜色A')
TARGET_FOLDER = '原图A'
CONDITION_FOLDERS2 = ('经过涂灰的原图AB', '原图A', '主颜色B')
TARGET_FOLDER2 = '生成图B'


def task_folders(direction):
    if direction == 'b_to_a':
        return TARGET_FOLDER, CONDITION_FOLDERS
    if direction == 'a_to_b':
        return TARGET_FOLDER2, CONDITION_FOLDERS2
    raise ValueError(f'未知换色方向: {direction}')


def subdataset_config(directions):
    return {direction: {'target_folder': task_folders(direction)[0],
                        'condition_folders': list(task_folders(direction)[1])} for direction in directions}
