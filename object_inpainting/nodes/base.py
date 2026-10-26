class Node:
    INPUT_TYPES: dict[str, str] = {}
    OUTPUT_TYPES: dict[str, str] = {}
    CATEGORY = "dataset"

    def execute(self, **inputs) -> dict:
        raise NotImplementedError
