"""Executes a symbolic goal by task planning with known operators."""

from typing import Collection

from modular_perception.query_types import AllGroundAtomsQuery, AllObjectDetectionQuery
from relational_structs import LiftedOperator, PDDLDomain, PDDLProblem, Predicate, Type
from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pddl_planner

from modular_execution.action_types import OperatorAction, SymbolicGoalAction
from modular_execution.executor import (
    Action,
    ExecutionModule,
    ModuleCannotExecuteAction,
    Status,
)


class TaskPlanningModule(ExecutionModule[SymbolicGoalAction]):
    """Executes a symbolic goal by task planning with known operators."""

    def __init__(
        self,
        operators: Collection[LiftedOperator],
        predicates: Collection[Predicate],
        types: Collection[Type],
        *args,
        **kwargs,
    ) -> None:
        self._domain_name = "task-planning-module-domain"
        self._pddl_domain = PDDLDomain(self._domain_name, operators, predicates, types)
        super().__init__(*args, **kwargs)

    def execute(self, action: Action) -> Status:
        if not isinstance(action, SymbolicGoalAction):
            raise ModuleCannotExecuteAction
        # Construct the task planning problem.
        objects = self._perceiver.get_response(AllObjectDetectionQuery())
        init = self._perceiver.get_response(AllGroundAtomsQuery())
        goal = action.goal_atoms
        pddl_problem = PDDLProblem(
            self._domain_name, "task-planning-module-problem", objects, init, goal
        )
        # Run task planning.
        plan_strs = run_pddl_planner(
            str(self._pddl_domain), str(pddl_problem), planner="pyperplan"
        )
        plan = parse_pddl_plan(plan_strs, self._pddl_domain, pddl_problem)
        for ground_operator in plan:
            self._delegate_execution(OperatorAction(ground_operator))
