"""An executor that directly executes in the environment."""

from typing import Any, Callable

from modular_execution.action_types import PrimitiveAction
from modular_execution.executor import (
    Action,
    ExecutionModule,
    ModuleCannotExecuteAction,
    Status,
)


class PrimitiveActionModule(ExecutionModule[PrimitiveAction]):
    """Directly executes actions in the environment."""

    def __init__(self, execution_fn: Callable[[Any], None], *args, **kwargs) -> None:
        self._execution_fn = execution_fn
        super().__init__(*args, **kwargs)

    def execute(self, action: Action) -> Status:
        if not isinstance(action, PrimitiveAction):
            raise ModuleCannotExecuteAction
        return self._execution_fn(action.action)
