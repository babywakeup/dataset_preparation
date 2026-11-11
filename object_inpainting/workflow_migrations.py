"""Non-destructive workflow compatibility migrations for Studio and the CLI."""

from copy import deepcopy


def migrate_sam3_dilation(definition: dict) -> dict:
    result = deepcopy(definition)
    nodes = result.get("nodes")
    if not isinstance(nodes, dict):
        return result
    for node_id, spec in list(nodes.items()):
        if not isinstance(spec, dict) or spec.get("class_type") != "SAM3TextMaskNode":
            continue
        inputs = spec.get("inputs")
        if not isinstance(inputs, dict) or not {"dilation_steps", "dilation_kernel"}.intersection(inputs):
            continue
        steps = inputs.pop("dilation_steps", 0)  # Preserve the old SAM3 default, not the new node default.
        kernel = inputs.pop("dilation_kernel", 3)
        dilation_id = f"{node_id}_dilation"
        suffix = 2
        while dilation_id in nodes:
            dilation_id = f"{node_id}_dilation_{suffix}"
            suffix += 1
        bindings = []
        for destination in nodes.values():
            if isinstance(destination, dict) and isinstance(destination.get("inputs"), dict):
                bindings.extend(destination["inputs"].values())
        if isinstance(result.get("outputs"), dict):
            bindings.extend(result["outputs"].values())
        for binding in bindings:
            if (
                isinstance(binding, dict)
                and set(binding) == {"node", "output"}
                and binding["node"] == node_id
                and isinstance(binding["output"], str)
                and binding["output"] in {"object_mask", "bbox"}
            ):
                binding["node"] = dilation_id
                if binding["output"] == "object_mask":
                    binding["output"] = "mask"
        # Add only after rewiring so this raw-mask input cannot become a self-link.
        nodes[dilation_id] = {
            "class_type": "MaskDilationNode",
            "inputs": {
                "mask": {"node": node_id, "output": "object_mask"},
                "dilation_steps": steps,
                "dilation_kernel": kernel,
            },
        }
    return result
