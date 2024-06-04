"""Tests for planning and execution with a hierarchy."""

import gymnasium as gym
import numpy as np
from modular_perception.modules.object_detection_module import ObjectDetectionModule
from modular_perception.modules.object_feature_module import ObjectFeatureModule
from modular_perception.modules.predicate_modules import (
    ImagePredicateModule,
    LocalPredicateModule,
    PredicateDispatchModule,
)
from modular_perception.perceiver import (
    ModularPerceiver,
)
from modular_perception.query_types import (
    SensorQuery,
)
from modular_perception.utils import wrap_gym_env_with_sensor_module
from relational_structs import LiftedOperator, Predicate, Type

from modular_execution.action_types import GoalStrAction, PrimitiveAction
from modular_execution.executor import ModularExecutor
from modular_execution.modules.goal_translation_module import GoalTranslationModule
from modular_execution.modules.operator_execution_module import OperatorExecutionModule
from modular_execution.modules.task_planning_module import TaskPlanningModule
from modular_execution.utils import create_execution_module_from_gym_env

################################################################################
#                                Environment                                   #
################################################################################


class _GridEnv(gym.Env):
    """Collect all the letters."""

    _grid = np.array(
        [
            ["R", "X", "X", "X", "X", "X", "X", "X", "G"],
            ["X", "A", "X", "X", "X", "X", "X", "X", "X"],
            ["X", "B", "X", "X", "C", "X", "X", "X", "X"],
            ["X", "X", "X", "X", "D", "X", "X", "X", "X"],
            ["X", "F", "X", "X", "X", "X", "X", "X", "X"],
            ["X", "X", "X", "X", "X", "X", "X", "X", "X"],
            ["X", "X", "X", "X", "X", "X", "E", "X", "X"],
            ["X", "X", "X", "X", "X", "X", "X", "X", "X"],
            ["X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ],
        dtype=object,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._obs = self._grid.copy()
        self.terminated = False

    def _get_observation(self):
        return self._obs.copy()

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._obs = self._grid.copy()
        self.terminated = False
        return self._get_observation(), {}

    def step(self, action):
        assert action in ("up", "down", "left", "right")
        dr, dc = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }[action]
        r, c = np.argwhere(self._obs == "R")[0]
        nr, nc = r + dr, c + dc
        if not (0 <= r < self._grid.shape[0] and 0 <= c < self._grid.shape[1]):
            nr, nc = r, c
        self._obs[r, c] = "X"
        self._obs[nr, nc] = "R"
        self.terminated = len(np.unique(self._obs)) == 2
        return self._get_observation(), 0.0, self.terminated, False, {}

    def render(self, *args, **kwargs):
        raise NotImplementedError


################################################################################
#                       Environment-Specific Knowledge                         #
################################################################################


def _create_types():
    return {Type("Letter"), Type("Robot")}


def _create_predicates(types):
    Letter = {t.name: t for t in types}["Letter"]
    return {
        Predicate("Collected", [Letter]),
        Predicate("NotCollected", [Letter]),
        Predicate("IsDirectlyAbove", [Letter, Letter]),
        Predicate("IsAnywhereAbove", [Letter, Letter]),
        Predicate("InOneThickEmptySpace", [Letter]),
        Predicate("InTwoThickEmptySpace", [Letter]),
    }


def _create_symbolic_goal_detector(predicates, types):
    type_name_to_type = {t.name: t for t in types}
    Letter = type_name_to_type["Letter"]
    pred_name_to_pred = {p.name: p for p in predicates}
    Collected = pred_name_to_pred["Collected"]

    def _detect_symbolic_goal(get_objects, goal_str):
        assert goal_str == "collect all the letters"
        objects = get_objects()
        letters = {o for o in objects if o.is_instance(Letter)}
        return {Collected([letter]) for letter in letters}

    return _detect_symbolic_goal


def _create_object_detector(types):
    type_name_to_type = {t.name: t for t in types}
    Robot = type_name_to_type["Robot"]
    Letter = type_name_to_type["Letter"]

    def _detect_objects(img):
        letters = set(np.unique(img)) - {"R", "X"}
        letter_objs = frozenset(Letter(l) for l in letters)
        return letter_objs | {Robot("R")}

    return _detect_objects


def _create_feature_detector(types):
    type_name_to_type = {t.name: t for t in types}
    Letter = type_name_to_type["Letter"]

    def _detect_object_features(img, obj, feature):
        assert feature in ("r", "c")
        idxs = np.argwhere(img == obj.name)
        if len(idxs) == 0:
            assert obj.is_instance(Letter)
            return -1  # does not exist
        assert len(idxs) == 1
        r, c = idxs[0]
        if feature == "r":
            return r
        assert feature == "c"
        return c

    return _detect_object_features


def _create_local_predicate_interpretations(predicates):
    pred_name_to_pred = {p.name: p for p in predicates}
    pred_to_interpretation = {}

    Collected = pred_name_to_pred["Collected"]

    def _Collected_holds(get_feature, obj):
        return get_feature(obj, "r") == -1

    pred_to_interpretation[Collected] = _Collected_holds

    NotCollected = pred_name_to_pred["NotCollected"]

    def _NotCollected_holds(get_feature, obj):
        return not _Collected_holds(get_feature, obj)

    pred_to_interpretation[NotCollected] = _NotCollected_holds

    IsDirectlyAbove = pred_name_to_pred["IsDirectlyAbove"]

    def _IsDirectlyAbove_holds(get_feature, obj1, obj2):
        if _Collected_holds(get_feature, obj1) or _Collected_holds(get_feature, obj2):
            return False
        r1 = get_feature(obj1, "r")
        c1 = get_feature(obj1, "c")
        r2 = get_feature(obj2, "r")
        c2 = get_feature(obj2, "c")
        return (r1 == r2 - 1) and (c1 == c2)

    pred_to_interpretation[IsDirectlyAbove] = _IsDirectlyAbove_holds

    IsAnywhereAbove = pred_name_to_pred["IsAnywhereAbove"]

    def _IsAnywhereAbove_holds(get_feature, obj1, obj2):
        if _Collected_holds(get_feature, obj1) or _Collected_holds(get_feature, obj2):
            return False
        r1 = get_feature(obj1, "r")
        c1 = get_feature(obj1, "c")
        r2 = get_feature(obj2, "r")
        c2 = get_feature(obj2, "c")
        return (r1 < r2) and (c1 == c2)

    pred_to_interpretation[IsAnywhereAbove] = _IsAnywhereAbove_holds

    return pred_to_interpretation


def _create_image_predicate_interpretations(predicates):
    pred_name_to_pred = {p.name: p for p in predicates}

    InOneThickEmptySpace = pred_name_to_pred["InOneThickEmptySpace"]
    InTwoThickEmptySpace = pred_name_to_pred["InTwoThickEmptySpace"]
    image_predicates = {InOneThickEmptySpace, InTwoThickEmptySpace}

    def _detect_image_predicates(predicates, objects, get_feature, get_image):
        # This could be implemented in a general way with a VLM instead.
        pred_to_pad = {InOneThickEmptySpace: 1, InTwoThickEmptySpace: 2}
        assert predicates.issubset(set(pred_to_pad))
        img = get_image()
        true_ground_atoms = set()

        def _has_empty_space(obj_r, obj_c, padding):
            for r in range(obj_r - padding, obj_r + padding + 1):
                if not 0 <= r < img.shape[0]:
                    continue
                for c in range(obj_c - padding, obj_c + padding + 1):
                    if not 0 <= c < img.shape[1]:
                        continue
                    if (r, c) == (obj_r, obj_c):
                        continue
                    if img[r, c] != "X":
                        return False
            return True

        for predicate in predicates:
            padding = pred_to_pad[predicate]
            for obj in objects:
                obj_r = int(get_feature(obj, "r"))
                obj_c = int(get_feature(obj, "c"))
                if _has_empty_space(obj_r, obj_c, padding):
                    ground_atom = predicate([obj])
                    true_ground_atoms.add(ground_atom)

        return true_ground_atoms

    return _detect_image_predicates, image_predicates


def _create_operators(predicates, types):
    pred_name_to_pred = {p.name: p for p in predicates}
    type_name_to_type = {t.name: t for t in types}
    Collected = pred_name_to_pred["Collected"]
    NotCollected = pred_name_to_pred["NotCollected"]
    Letter = type_name_to_type["Letter"]
    Robot = type_name_to_type["Robot"]

    robot = Robot("?robot")
    letter = Letter("?letter")
    CollectLetter = LiftedOperator(
        name="CollectLetter",
        parameters=[letter, robot],
        preconditions={NotCollected([letter])},
        add_effects={Collected([letter])},
        delete_effects={NotCollected([letter])},
    )

    return {CollectLetter}


def _create_operator_interpretations(operators):
    operator_name_to_operator = {o.name: o for o in operators}
    CollectLetter = operator_name_to_operator["CollectLetter"]

    def _execute_CollectLetter(get_feature, letter, robot):
        rob_r = get_feature(robot, "r")
        rob_c = get_feature(robot, "c")
        let_r = get_feature(letter, "r")
        let_c = get_feature(letter, "c")
        plan = []
        if let_r == -1:
            return plan
        if rob_r < let_r:
            for _ in range(let_r - rob_r):
                plan.append(PrimitiveAction("down"))
        if rob_r > let_r:
            for _ in range(rob_r - let_r):
                plan.append(PrimitiveAction("up"))
        if rob_c < let_c:
            for _ in range(let_c - rob_c):
                plan.append(PrimitiveAction("right"))
        if rob_c > let_c:
            for _ in range(rob_c - let_c):
                plan.append(PrimitiveAction("left"))
        return plan

    return {CollectLetter: _execute_CollectLetter}


################################################################################
#                              Building the Agent                              #
################################################################################


def _create_perceiver(sensor_module):
    # Create environment-specific models.
    types = _create_types()
    predicates = _create_predicates(types)
    local_interp = _create_local_predicate_interpretations(predicates)
    image_interp, image_preds = _create_image_predicate_interpretations(predicates)
    detect_object_features = _create_feature_detector(types)
    detect_objects = _create_object_detector(types)

    # Build the modules.
    image_query = SensorQuery("gym_observation")
    object_detection_module = ObjectDetectionModule(
        detect_objects, sensory_input_query=image_query
    )
    object_feature_module = ObjectFeatureModule(
        detect_object_features, sensory_input_query=image_query
    )
    local_predicate_module = LocalPredicateModule(
        local_interp,
    )
    image_predicate_module = ImagePredicateModule(
        image_interp,
        image_query=image_query,
    )
    predicate_dispatch_module = PredicateDispatchModule(
        local_predicates=frozenset(local_interp),
        image_predicates=frozenset(image_preds),
    )

    # Finalize the perceiver.
    perceiver = ModularPerceiver(
        {
            sensor_module,
            object_feature_module,
            object_detection_module,
            local_predicate_module,
            image_predicate_module,
            predicate_dispatch_module,
        }
    )

    return perceiver


def _create_executor(primitive_action_module, perceiver, seed):
    # Create environment-specific models.
    types = _create_types()
    predicates = _create_predicates(types)
    detect_goal = _create_symbolic_goal_detector(predicates, types)
    operators = _create_operators(predicates, types)
    operator_interpretations = _create_operator_interpretations(operators)

    # Create the modules.
    goal_translation_module = GoalTranslationModule(
        detect_goal,
        perceiver,
        seed,
    )

    task_planning_module = TaskPlanningModule(
        operators,
        predicates,
        types,
        perceiver,
        seed,
    )

    operator_module = OperatorExecutionModule(
        operator_interpretations,
        perceiver,
        seed,
    )

    # Finalize the executor.
    executor = ModularExecutor(
        {
            goal_translation_module,
            task_planning_module,
            operator_module,
            primitive_action_module,
        }
    )

    return executor


################################################################################
#                               Running the Test                               #
################################################################################


def test_hierarchical_actions():
    """Tests for planning and execution with a hierarchy."""

    # Create the environment.
    env = _GridEnv()
    env, sensor_module = wrap_gym_env_with_sensor_module(env)
    seed = 123

    # Create the perceiver.
    perceiver = _create_perceiver(sensor_module)

    # Create the executor.
    primitive_action_module = create_execution_module_from_gym_env(env, perceiver, seed)
    agent = _create_executor(primitive_action_module, perceiver, seed)

    # Create a goal, which is like a top-level action.
    goal = GoalStrAction("collect all the letters")

    # Run.
    env.reset()
    agent.execute(goal)
    assert env.unwrapped.terminated
