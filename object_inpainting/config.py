from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    resolution: int = 1024
    reference_resolution: int = 512
    dilation_radius: int = 4
    reference_padding: float = 0.1
    min_object_fraction: float = 0.01
    max_object_fraction: float = 0.6
    validation_fraction: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if self.resolution <= 0 or self.resolution % 16:
            raise ValueError("resolution must be positive and divisible by 16.")
        if self.reference_resolution <= 0 or self.reference_resolution % 16:
            raise ValueError("reference_resolution must be positive and divisible by 16.")
        if self.dilation_radius < 0:
            raise ValueError("dilation_radius must be non-negative.")
        if not 0 <= self.reference_padding <= 1:
            raise ValueError("reference_padding must be between 0 and 1.")
        if not 0 < self.min_object_fraction < self.max_object_fraction < 1:
            raise ValueError("Object fractions must satisfy 0 < min < max < 1.")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1).")
