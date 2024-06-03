"""Action types."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrimitiveAction:
    """A primitive action that can be directly executed in the environment."""

    action: Any
