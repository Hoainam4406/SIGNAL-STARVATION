"""Reward factory for instantiating and wrapping NavSim environments.

Supports loading reward configurations from YAML files or Python dicts,
and wrapping environments with navsim.wrappers.Reward.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Type, Union
import yaml
import gymnasium as gym
import navsim.wrappers

from src.rewards.base_reward import BaseReward
from src.rewards.rewards import (
    PureSparseReward,
    CollisionPenalizedSparseReward,
    DistanceBasedDenseReward,
    ProgressDenseReward,
    ProgressGoalReward,
    PotentialBasedShapedReward,
)

# Registry mapping canonical names and aliases to reward classes
REWARD_REGISTRY: Dict[str, Type[BaseReward]] = {
    # R1: Pure Sparse
    "r1": PureSparseReward,
    "pure_sparse": PureSparseReward,
    "sparse": PureSparseReward,
    # R2: Collision-Penalized Sparse
    "r2": CollisionPenalizedSparseReward,
    "collision_penalized_sparse": CollisionPenalizedSparseReward,
    "sparse_collision": CollisionPenalizedSparseReward,
    "penalized_sparse": CollisionPenalizedSparseReward,
    # R3: Distance-Based Dense
    "r3": DistanceBasedDenseReward,
    "distance_dense": DistanceBasedDenseReward,
    "distance_based_dense": DistanceBasedDenseReward,
    "distance": DistanceBasedDenseReward,
    # R4: Progress Dense
    "r4": ProgressDenseReward,
    "progress_dense": ProgressDenseReward,
    "progress": ProgressDenseReward,
    # R5: Progress + Goal Bonus
    "r5": ProgressGoalReward,
    "progress_goal": ProgressGoalReward,
    "progress_goal_bonus": ProgressGoalReward,
    "progress_dense_goal": ProgressGoalReward,
    # R6: Potential-Based Shaped
    "r6": PotentialBasedShapedReward,
    "potential_shaped": PotentialBasedShapedReward,
    "potential_based_shaped": PotentialBasedShapedReward,
    "potential": PotentialBasedShapedReward,
}


def register_reward(name: str, reward_cls: Type[BaseReward]) -> None:
    """Register a custom reward class under a given name/alias."""
    if not issubclass(reward_cls, BaseReward):
        raise TypeError(f"{reward_cls} must be a subclass of BaseReward")
    REWARD_REGISTRY[name.lower().strip()] = reward_cls


def make_reward(
    config: Union[str, Path, Dict[str, Any], BaseReward]
) -> BaseReward:
    """Instantiate a BaseReward object from a YAML file, dict, or existing instance.

    Supported formats:
        1. File path to YAML (e.g., 'configs/reward_r4.yaml')
        2. Dict with nested 'reward' key: {'reward': {'type': 'progress_dense', ...}}
        3. Flat dict: {'type': 'progress_dense', ...}
        4. Plain string of reward type: 'progress_dense'
        5. Existing BaseReward instance (returned as-is)

    Args:
        config: Configuration source.

    Returns:
        BaseReward instance.
    """
    if isinstance(config, BaseReward):
        return config

    cfg_dict: Dict[str, Any] = {}

    if isinstance(config, (str, Path)):
        # Check if it's a file path or a direct reward type string
        path = Path(config)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if not isinstance(loaded, dict):
                    raise ValueError(f"YAML config in {path} must parse to a dict")
                cfg_dict = loaded
        else:
            # Treated as a reward type string (e.g. 'progress_dense' or 'R4')
            cfg_dict = {"type": str(config)}

    elif isinstance(config, dict):
        cfg_dict = dict(config)
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    # Handle nested 'reward' section if present (e.g. standard YAML structure)
    if "reward" in cfg_dict and isinstance(cfg_dict["reward"], dict):
        cfg_dict = dict(cfg_dict["reward"])

    reward_type = cfg_dict.pop("type", None)
    if not reward_type:
        raise ValueError("Reward configuration must specify 'type'. Example: type: 'progress_dense'")

    key = str(reward_type).lower().strip()
    if key not in REWARD_REGISTRY:
        available = sorted(set(REWARD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown reward type: '{reward_type}'. Available types: {available}"
        )

    reward_cls = REWARD_REGISTRY[key]
    return reward_cls(**cfg_dict)


def wrap_env_with_reward(
    env: gym.Env,
    reward_config: Union[str, Path, Dict[str, Any], BaseReward, Callable],
) -> navsim.wrappers.Reward:
    """Wrap a NavSim environment with navsim.wrappers.Reward using a reward function.

    Wrapper call:
        navsim.wrappers.Reward(env, lambda state, info: reward_fn(state, info))

    Args:
        env: The Gymnasium / NavSim environment to wrap.
        reward_config: YAML path, config dict, BaseReward instance, or callable.

    Returns:
        navsim.wrappers.Reward wrapped environment.
    """
    if callable(reward_config) and not isinstance(reward_config, BaseReward):
        reward_fn = reward_config
    else:
        reward_fn = make_reward(reward_config)

    wrapped_env = navsim.wrappers.Reward(
        env,
        lambda state, info: reward_fn(state, info)
    )
    # Store reference to reward_fn on wrapped env for convenience
    setattr(wrapped_env, "reward_fn", reward_fn)
    return wrapped_env
