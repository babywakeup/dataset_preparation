from collections.abc import Mapping

from ..node_contracts import port_schema


class Node:
    INPUT_TYPES: dict[str, str] = {}
    OUTPUT_TYPES: dict[str, str] = {}
    INPUT_SCHEMA: type | None = None
    OUTPUT_SCHEMA: type | None = None
    CATEGORY = "dataset"

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "INPUT_SCHEMA" in cls.__dict__ and cls.INPUT_SCHEMA is not None:
            cls.INPUT_TYPES = port_schema(cls.INPUT_SCHEMA)
        if "OUTPUT_SCHEMA" in cls.__dict__ and cls.OUTPUT_SCHEMA is not None:
            cls.OUTPUT_TYPES = port_schema(cls.OUTPUT_SCHEMA)

    def execute(self, **inputs) -> Mapping[str, object]:
        raise NotImplementedError

    @classmethod
    def prepare_spec(cls, spec: dict) -> None:
        """Normalize per-instance port metadata before graph validation."""

    @classmethod
    def input_types(cls, spec: dict) -> dict[str, str]:
        return cls.INPUT_TYPES

    @classmethod
    def output_types(cls, spec: dict) -> dict[str, str]:
        return cls.OUTPUT_TYPES

    @classmethod
    def optional_input(cls, port: str) -> bool:
        return port in getattr(cls, "OPTIONAL_INPUTS", set())

    def execute_with_spec(self, inputs: dict, spec: dict) -> Mapping[str, object]:
        return self.execute(**inputs)
