"""A modular executor."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import (
    Any,
    Generic,
    Set,
    Tuple,
    TypeAlias,
    TypeVar,
)

import numpy as np
from modular_perception.perceiver import ModularPerceiver
from tomsutils.utils import draw_dag

Action = TypeVar("Action")
Status: TypeAlias = Any


class ModularExecutor:
    """A modular executor."""

    def __init__(self, modules: Set[ExecutionModule]) -> None:
        self._modules = modules
        # Create connection from each module to this perceiver.
        for module in self._modules:
            module.set_executor(self)
        # For drawing and debugging, record the "connectivity" of the modules
        # in terms of which has ever sent an action to which.
        self._module_edges: Set[Tuple[ExecutionModule, ExecutionModule]] = set()

    def reset(self, seed: int | None = None) -> None:
        """Reset the modules of the perceiver."""
        for module in self._modules:
            module.reset(seed)

    def tick(self) -> None:
        """Advance time for all modules."""
        for module in self._modules:
            module.tick()

    def execute(
        self,
        action: Any | None = None,
        sender: ExecutionModule | None = None,
    ) -> Any:
        """Find a module that can execute the action, and execute it.

        Conventionally, an action of None is reserved for the top level.
        So at each step, the agent should do executor.execute().

        The sender is provided just for logging purposes. A sender of
        None means that the request came from outside the perceiver.
        """
        response = None
        responder: ExecutionModule | None = None
        for module in self._modules:
            try:
                response = module.execute(action)
                assert responder is None, "Multiple modules can execute action"
                responder = module
            except ModuleCannotExecuteAction:
                continue
        assert responder is not None, f"No module can execute action: {action}"
        if sender is not None:
            self._module_edges.add((responder, sender))
        return response

    def draw_connections(self, outfile: Path) -> None:
        """Draw the module connections based on queries sent so far."""
        edges = {
            (r.__class__.__name__, s.__class__.__name__) for r, s in self._module_edges
        }
        draw_dag(edges, outfile)


class ModuleCannotExecuteAction(Exception):
    """Raised when a module is given an action it cannot execute."""


class ExecutionModule(abc.ABC, Generic[Action]):
    """Base class for a module."""

    def __init__(self, perceiver: ModularPerceiver, seed: int = 0) -> None:
        self._perceiver = perceiver
        self._set_seed(seed)
        self._time = 0
        self._executor: ModularExecutor | None = None

    def set_executor(self, executor: ModularExecutor):
        """Set the executor for this module."""
        self._executor = executor

    def _set_seed(self, seed: int) -> None:
        """Set the internal random number generator."""
        self._rng = np.random.default_rng(seed)

    ################### Handling actions FROM other modules ###################

    @abc.abstractmethod
    def execute(self, action: Action) -> Status:
        """Module-specific logic for execution."""
        raise NotImplementedError

    #################### Sending actions TO other modules #####################

    def _send_action(self, action: Any) -> Any:
        """Convenient method for delegating execution to another module.

        Note that the action type is not Action because it should be the
        parent's type, not this module's type. Same for status.
        """
        assert self._executor is not None
        return self._executor.execute(action, self)

    ################### Managing internal state (memory) ######################

    def reset(self, seed: int | None = None) -> None:
        """Reset the module."""
        self._time = 0
        if seed is not None:
            self._set_seed(seed)

    def tick(self) -> None:
        """Advance time."""
        self._time += 1
