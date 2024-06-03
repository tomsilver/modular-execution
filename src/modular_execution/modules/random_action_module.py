"""An executor that samples and executes random actions."""

from typing import Any, Callable, Type, TypeAlias

from modular_execution.action_types import RandomAction
from modular_execution.executor import (
    Action,
    ExecutionModule,
    ModuleCannotExecuteAction,
    Status,
)

_SampledActionType: TypeAlias = Any


class RandomActionModule(ExecutionModule[_SampledActionType]):
    """Samples and executes random actions."""

    def __init__(
        self,
        sampled_action_type: Type[_SampledActionType],
        sample_fn: Callable[[], None],
        *args,
        **kwargs,
    ) -> None:
        self._sampled_action_type = sampled_action_type
        self._sample_fn = sample_fn
        super().__init__(*args, **kwargs)

    def execute(self, action: Action) -> Status:
        if not isinstance(action, RandomAction):
            raise ModuleCannotExecuteAction
        if action.action_type != self._sampled_action_type:
            raise ModuleCannotExecuteAction
        sampled_action = self._sample_fn()
        return self._delegate_execution(sampled_action)
