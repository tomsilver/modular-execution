"""Executes a symbolic operator."""

from typing import Callable, Dict, List, TypeAlias

from modular_perception.query_types import ObjectFeatureQuery
from relational_structs import LiftedOperator, Object
from typing_extensions import Unpack

from modular_execution.action_types import OperatorAction
from modular_execution.executor import (
    Action,
    ExecutionModule,
    ModuleCannotExecuteAction,
    Status,
)

FeatureDetector: TypeAlias = Callable[[Object, str], float]
OperatorInterpretation: TypeAlias = Callable[
    [FeatureDetector, Unpack[Object]], List[Action]
]


class OperatorExecutionModule(ExecutionModule[OperatorAction]):
    """Executes a symbolic operator."""

    def __init__(
        self,
        operator_interpretations: Dict[LiftedOperator, OperatorInterpretation],
        *args,
        **kwargs,
    ) -> None:
        self._operator_interpretations = operator_interpretations
        super().__init__(*args, **kwargs)

    def execute(self, action: Action) -> Status:
        if not isinstance(action, OperatorAction):
            raise ModuleCannotExecuteAction
        operator = action.operator
        detect_feature = lambda o, f: self._perceiver.get_response(
            ObjectFeatureQuery(o, f)
        )
        interpreter = self._operator_interpretations[operator.parent]
        plan = interpreter(detect_feature, *operator.parameters)
        for plan_step in plan:
            self._delegate_execution(plan_step)
