"""An executor that translates a goal string into a symbolic goal."""

from typing import Callable, Collection, Set, TypeAlias

from modular_perception.query_types import AllObjectDetectionQuery
from relational_structs import GroundAtom, Object

from modular_execution.action_types import GoalStrAction, SymbolicGoalAction
from modular_execution.executor import (
    Action,
    ExecutionModule,
    ModuleCannotExecuteAction,
    Status,
)

ObjectDetector: TypeAlias = Callable[[], Set[Object]]


class GoalTranslationModule(ExecutionModule[GoalStrAction]):
    """Translates a goal string into a symbolic goal."""

    def __init__(
        self,
        translate_fn: Callable[[ObjectDetector, str], Collection[GroundAtom]],
        *args,
        **kwargs,
    ) -> None:
        self._translate_fn = translate_fn
        super().__init__(*args, **kwargs)

    def execute(self, action: Action) -> Status:
        if not isinstance(action, GoalStrAction):
            raise ModuleCannotExecuteAction
        detect_objects = lambda: self._perceiver.get_response(AllObjectDetectionQuery())
        atoms = self._translate_fn(detect_objects, action.goal_str)
        goal = SymbolicGoalAction(frozenset(atoms))
        return self._delegate_execution(goal)
