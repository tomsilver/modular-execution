"""Example showing how to interact with a gym environment."""

import gymnasium as gym
from modular_perception.perceiver import ModularPerceiver
from modular_perception.utils import wrap_gym_env_with_sensor_module

from modular_execution.action_types import PrimitiveAction, RandomAction
from modular_execution.executor import ModularExecutor
from modular_execution.modules.primitive_action_module import PrimitiveActionModule
from modular_execution.modules.random_action_module import RandomActionModule


def test_gym_api():
    """Example showing how to interact with a gym environment."""

    # Create the environment.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Uncomment to create a video.
    # from gymnasium.wrappers.record_video import RecordVideo
    # env = RecordVideo(env, "videos")
    seed = 123
    env.action_space.seed(seed)

    # Create a sensor module that captures observations whenever env.reset()
    # or env.step() are called. The observation are not actually used in this
    # test, but we include sensing for illustrative purposes.
    sensor_name = "gym_observation"
    env, sensor_module = wrap_gym_env_with_sensor_module(env, sensor_name)
    perceiver = ModularPerceiver({sensor_module})

    # Create an executor that executes random actions.
    def _execution_fn(action):
        env.step(action)

    primitive_action_module = PrimitiveActionModule(_execution_fn, perceiver, seed)

    def _sample_fn():
        return PrimitiveAction(env.action_space.sample())

    random_action_module = RandomActionModule(
        PrimitiveAction, _sample_fn, perceiver, seed
    )

    executor = ModularExecutor({random_action_module, primitive_action_module})

    env.reset()
    executor.reset(seed)
    for _ in range(10):
        executor.execute(RandomAction(PrimitiveAction))
    env.close()
