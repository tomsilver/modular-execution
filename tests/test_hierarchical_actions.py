"""Tests for planning and execution with a hierarchy."""

from typing import Set

import gymnasium as gym
import numpy as np
from modular_perception.modules.object_detection_module import ObjectDetectionModule
from modular_perception.modules.object_feature_module import ObjectFeatureModule
from modular_perception.modules.predicate_modules import (
    ImagePredicateModule,
    LocalPredicateModule,
    PredicateDispatchModule,
)
from modular_perception.modules.sensor_module import SensorModule
from modular_perception.perceiver import (
    ModularPerceiver,
    ModuleCannotAnswerQuery,
    PerceptionModule,
)
from modular_perception.query_types import (
    PredicatesQuery,
    SensorQuery,
)
from modular_perception.utils import wrap_gym_env_with_sensor_module
from relational_structs import GroundAtom, Predicate, Type

################################################################################
#                                Environment                                   #
################################################################################


class _GridEnv(gym.Env):
    """A simple 2D grid environment for testing."""

    _grid = np.array(
        [
            ["X", "X", "X", "X", "X", "X", "X", "X", "G"],
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
        self._robot_loc = (0, 0)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._robot_loc = (0, 0)
        return self._robot_loc, {}

    def step(self, action):
        assert action in ("up", "down", "left", "right")
        dr, dc = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }[action]
        r, c = self._robot_loc
        nr, nc = r + dr, c + dc
        if 0 <= r < self._grid.shape[0] and 0 <= c < self._grid.shape[1]:
            self._robot_loc = (nr, nc)
        return self._robot_loc, 0.0, False, False, {}

    def render(self, *args, **kwargs):
        raise NotImplementedError


################################################################################
#                       Environment-Specific Knowledge                         #
################################################################################


def _create_environment_specific_perceiver_models():
    # Types.
    Letter = Type("Letter")
    object_types = {Type("Letter")}

    # Objects.
    def _detect_objects(img, types):
        assert set(types) == object_types
        letters = set(np.unique(img)) - {"X"}
        return frozenset(Letter(l) for l in letters)

    # Object features.
    def _detect_object_features(img, obj, feature):
        assert feature in ("r", "c")
        idxs = np.argwhere(img == obj.name)
        assert len(idxs) == 1
        r, c = idxs[0]
        if feature == "r":
            return r
        assert feature == "c"
        return c

    # Local predicates.
    IsDirectlyAbove = Predicate("IsDirectlyAbove", [Letter, Letter])

    def _IsDirectlyAbove_holds(get_feature, obj1, obj2):
        r1 = get_feature(obj1, "r")
        c1 = get_feature(obj1, "c")
        r2 = get_feature(obj2, "r")
        c2 = get_feature(obj2, "c")
        return (r1 == r2 - 1) and (c1 == c2)

    IsAnywhereAbove = Predicate("IsAnywhereAbove", [Letter, Letter])

    def _IsAnywhereAbove_holds(get_feature, obj1, obj2):
        r1 = get_feature(obj1, "r")
        c1 = get_feature(obj1, "c")
        r2 = get_feature(obj2, "r")
        c2 = get_feature(obj2, "c")
        return (r1 < r2) and (c1 == c2)

    predicate_interpretations = {
        IsDirectlyAbove: _IsDirectlyAbove_holds,
        IsAnywhereAbove: _IsAnywhereAbove_holds,
    }

    # Image predicates.
    InOneThickEmptySpace = Predicate("InOneThickEmptySpace", [Letter])
    InTwoThickEmptySpace = Predicate("InTwoThickEmptySpace", [Letter])
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

    return (
        object_types,
        _detect_objects,
        _detect_object_features,
        predicate_interpretations,
        image_predicates,
        _detect_image_predicates,
    )


################################################################################
#                              Building the Agent                              #
################################################################################


def _create_perceiver(sensor_module):
    # Get environment-specific models.
    (
        object_types,
        _detect_objects,
        _detect_object_features,
        predicate_interpretations,
        image_predicates,
        _detect_image_predicates,
    ) = _create_environment_specific_perceiver_models()

    # Query to get image.
    image_query = SensorQuery("gym_observation")

    # Build the modules.
    object_detection_module = ObjectDetectionModule(
        _detect_objects, sensory_input_query=image_query
    )
    object_feature_module = ObjectFeatureModule(
        _detect_object_features, sensory_input_query=image_query
    )
    local_predicate_module = LocalPredicateModule(
        predicate_interpretations,
    )
    image_predicate_module = ImagePredicateModule(
        _detect_image_predicates,
        image_query=image_query,
    )
    predicate_dispatch_module = PredicateDispatchModule(
        local_predicates=frozenset(predicate_interpretations),
        image_predicates=frozenset(image_predicates),
        object_types=frozenset(object_types),
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


################################################################################
#                               Running the Test                               #
################################################################################


def test_hierarchical_actions():
    """Tests for planning and execution with a hierarchy."""

    # Create the environment.
    env = _GridEnv()
    env, sensor_module = wrap_gym_env_with_sensor_module(env)

    # Create the perceiver.
    perceiver = _create_perceiver(sensor_module)
