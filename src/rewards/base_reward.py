"""Base class for pluggable navigation reward functions.

All reward functions take (state, info) from NavSim and return a scalar float reward.
Compatible with navsim.wrappers.Reward:
    env = navsim.wrappers.Reward(env, lambda state, info: reward_fn(state, info))
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseReward(ABC):
    """Abstract base class for all NavSim reward functions."""

    def __init__(
        self,
        goal_bonus: float = 0.0,
        step_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        **kwargs: Any,
    ):
        self.goal_bonus = float(goal_bonus)
        self.step_penalty = float(step_penalty)
        self.collision_penalty = float(collision_penalty)
        self.extra_kwargs = kwargs

    @abstractmethod
    def __call__(self, state: Any, info: Dict[str, Any]) -> float:
        """Compute the step reward given current state and step info.

        Args:
            state: RobotState object from env.unwrapped.robot_state()
            info: Step info dictionary from env.step(action)

        Returns:
            float: Scalar reward for the transition.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state if the reward is stateful across steps."""
        pass

    # -------------------------------------------------------------------------
    # Safe inspection helpers for GridNav / Continuous NavSim environments
    # -------------------------------------------------------------------------

    @staticmethod
    def is_goal_reached(state: Any, info: Dict[str, Any]) -> bool:
        """Check if the goal was reached in this step."""
        if getattr(state, "reached_goal", False):
            return True
        if info.get("success", False) or info.get("reached_goal", False):
            return True
        if info.get("termination") == "success":
            return True
        return False

    @staticmethod
    def is_collided(state: Any, info: Dict[str, Any]) -> bool:
        """Check if collision / blocked move occurred in this step."""
        if getattr(state, "collided", False):
            return True
        if info.get("collided", False):
            return True
        return False

    @staticmethod
    def get_current_distance(state: Any, info: Dict[str, Any]) -> float:
        """Extract the current distance to goal (BFS geodesic or Euclidean)."""
        # 1. Step info from GridNavEnv: 'goal_geodesic' is the current steps to goal
        if "goal_geodesic" in info and info["goal_geodesic"] is not None:
            return float(info["goal_geodesic"])

        # 2. state.geodesic_distance() from GridRobotState
        if hasattr(state, "geodesic_distance") and callable(state.geodesic_distance):
            try:
                dist = state.geodesic_distance()
                if np.isfinite(dist):
                    return float(dist)
            except Exception:
                pass

        # 3. state.goal_distance() from RobotState
        if hasattr(state, "goal_distance") and callable(state.goal_distance):
            try:
                dist = state.goal_distance()
                if np.isfinite(dist):
                    return float(dist)
            except Exception:
                pass

        # 4. info['d_bfs_curr'] if provided
        if "d_bfs_curr" in info and info["d_bfs_curr"] is not None:
            return float(info["d_bfs_curr"])

        # 5. Fallback to info['d_bfs']
        if "d_bfs" in info and info["d_bfs"] is not None:
            return float(info["d_bfs"])

        return 0.0

    @staticmethod
    def get_initial_distance(state: Any, info: Dict[str, Any]) -> float:
        """Extract the initial / task distance from start to goal."""
        # In GridNavEnv, info['d_bfs'] is the BFS distance from start to goal
        if "d_bfs" in info and info["d_bfs"] is not None:
            try:
                val = float(info["d_bfs"])
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass

        if hasattr(state, "start_geodesic"):
            try:
                val = float(state.start_geodesic)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass

        return 1.0
