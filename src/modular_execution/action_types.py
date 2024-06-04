"""Action types."""

from dataclasses import dataclass
from typing import Any, FrozenSet

from relational_structs import GroundAtom, GroundOperator


@dataclass(frozen=True)
class PrimitiveAction:
    """A primitive action that can be directly executed in the environment."""

    action: Any


@dataclass(frozen=True)
class RandomAction:
    """A random action of some subtype."""

    action_type: Any


@dataclass(frozen=True)
class GoalStrAction:
    """A goal action represented with a string, e.g., natural language."""

    goal_str: str


@dataclass(frozen=True)
class SymbolicGoalAction:
    """A goal action represented symbolically."""

    goal_atoms: FrozenSet[GroundAtom]


@dataclass(frozen=True)
class OperatorAction:
    """A symbolic operator action."""

    operator: GroundOperator
