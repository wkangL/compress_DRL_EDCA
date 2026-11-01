from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import copy
import numpy as np


@dataclass
class Transition:
    state: dict
    action: np.ndarray
    reward: float
    next_state: dict
    done: bool
    encoder_version: int
    truncated: bool = False
    scenario: str = ""
    exploratory: bool = False
    # CMDP runs retain raw components; legacy scalar-reward runs leave them None.
    utility: float | None = None
    costs: np.ndarray | None = None


class ReplayBuffer:
    """Raw generated windows and ages survive encoder updates; no latent-only replay."""
    def __init__(self, capacity: int, seed: int):
        self.data: deque[Transition]=deque(maxlen=capacity)
        self.rng=np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        self.data.append(copy.deepcopy(transition))

    def sample(self, size: int) -> list[Transition]:
        indices=self.rng.choice(len(self.data),size=size,replace=False)
        return [self.data[int(i)] for i in indices]

    def __len__(self) -> int: return len(self.data)
