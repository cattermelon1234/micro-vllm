from dataclasses import dataclass
from math import isfinite


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 128
    ignore_eos: bool = False

    def __post_init__(self):
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

