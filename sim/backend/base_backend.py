from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlAction:
    profile: int = 0
    custom_growth: bool = False
    alpha: int = 1
    beta1: float = 2.0
    beta2: float = 2.0
    gamma: int = 7
    # None preserves the legacy/unified protocol: AP and every STA use
    # ``profile`` and ``custom_growth``.  Split-profile control supplies the
    # STA values explicitly while keeping the four CW-growth parameters shared.
    sta_profile: int | None = None
    sta_custom_growth: bool | None = None


@dataclass(frozen=True)
class Event:
    kind: str
    values: tuple[float, ...]


class BaseBackend(ABC):
    """Advance real MAC time; return ordered events without learning dependencies."""
    @abstractmethod
    def step(self, action: ControlAction, duration_s: float) -> list[Event]: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> BaseBackend:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
