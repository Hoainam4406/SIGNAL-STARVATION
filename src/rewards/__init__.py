"""Rewards package for Signal Starvation navigation experiments."""

from src.rewards.base_reward import BaseReward
from src.rewards.rewards import (
    PureSparseReward,
    CollisionPenalizedSparseReward,
    DistanceBasedDenseReward,
    ProgressDenseReward,
    ProgressGoalReward,
    PotentialBasedShapedReward,
)
from src.rewards.reward_factory import (
    make_reward,
    wrap_env_with_reward,
    register_reward,
    REWARD_REGISTRY,
)

__all__ = [
    "BaseReward",
    "PureSparseReward",
    "CollisionPenalizedSparseReward",
    "DistanceBasedDenseReward",
    "ProgressDenseReward",
    "ProgressGoalReward",
    "PotentialBasedShapedReward",
    "make_reward",
    "wrap_env_with_reward",
    "register_reward",
    "REWARD_REGISTRY",
]
