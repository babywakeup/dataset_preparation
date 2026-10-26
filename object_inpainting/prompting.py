def build_prompt(object_name: str) -> str:
    return (
        f"Restore the {object_name} in the gray-masked region of the first image "
        "using the object in the second image as reference. "
        "Keep the rest of the image unchanged."
    )
