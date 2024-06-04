"""Utility functions."""

import gymnasium as gym
from modular_perception.perceiver import ModularPerceiver

from modular_execution.modules.primitive_action_module import PrimitiveActionModule


def create_execution_module_from_gym_env(
    env: gym.Env, perceiver: ModularPerceiver, seed: int
) -> PrimitiveActionModule:
    """Create an execution module that directly steps in a gym environment."""

    def _execution_fn(action):
        env.step(action)
        perceiver.tick()

    return PrimitiveActionModule(_execution_fn, perceiver, seed)
