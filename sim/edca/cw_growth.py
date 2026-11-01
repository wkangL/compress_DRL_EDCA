"""Piecewise CW growth, with n=1 denoting the first failed transmission."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING


@dataclass(frozen=True)
class Growth:
    alpha: int
    beta1: float
    beta2: float
    gamma: int

    def __post_init__(self) -> None:
        if not 1 <= self.alpha <= self.gamma or not 1 <= self.beta1 <= self.beta2:
            raise ValueError("Require 1 <= alpha <= gamma and 1 <= beta1 <= beta2")

    def after_failure(self, cw: int, cwmax: int, n: int) -> int:
        """Update the previous CW; retain it after the saturation stage."""
        if n < 1 or not 0 <= cw <= cwmax:
            raise ValueError("Invalid failure stage or contention window")
        if n > self.gamma:
            return cw
        beta = self.beta1 if n <= self.alpha else self.beta2
        # Decimal avoids spurious ceil increments at decimal candidate boundaries.
        value = (Decimal(str(beta)) * (cw + 1)).to_integral_value(rounding=ROUND_CEILING)
        return min(cwmax, int(value) - 1)

    def table(self, cwmin: int, cwmax: int, failures: int) -> list[int]:
        values = [cwmin]
        for n in range(1, failures + 1):
            values.append(self.after_failure(values[-1], cwmax, n))
        return values
