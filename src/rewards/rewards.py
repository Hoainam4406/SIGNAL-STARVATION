"""Concrete implementations of navigation reward functions (R1 - R6).

Based on the Signal Starvation research taxonomy and reward design specifications:
- R1: PureSparseReward (+1 only on reached_goal)
- R2: CollisionPenalizedSparseReward (+1 on goal, -0.5 on collision)
- R3: DistanceBasedDenseReward (-d_bfs_normalized per step)
- R4: ProgressDenseReward (d_bfs(prev) - d_bfs(now) progress reward)
- R5: ProgressGoalReward (R4 + goal bonus + step penalty)
- R6: PotentialBasedShapedReward (gamma * Phi(s') - Phi(s) with Phi = -d_bfs)
"""

from typing import Any, Dict, Optional
from src.rewards.base_reward import BaseReward


class PureSparseReward(BaseReward):
    """R1: Pure Sparse Reward.

    Provides a reward only upon reaching the goal, representing maximum signal
    starvation along the reward shaping dimension.
    Formula:
        r_t = goal_bonus * [reached_goal] + step_penalty
    """

    def __init__(
        self,
        goal_bonus: float = 1.0,
        step_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            goal_bonus=goal_bonus,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            **kwargs,
        )

    def __call__(self, state: Any, info: Dict[str, Any]) -> float:
        reward = 0.0
        if self.is_goal_reached(state, info):
            reward += self.goal_bonus
        reward += self.step_penalty
        return float(reward)


class CollisionPenalizedSparseReward(BaseReward):
    """R2: Collision-Penalized Sparse Reward.

    Sparse reward with negative reinforcement for collisions / blocked moves.
    Formula:
        r_t = goal_bonus * [reached_goal] - collision_penalty * [collided] + step_penalty
    """

    def __init__(
        self,
        goal_bonus: float = 1.0,
        collision_penalty: float = 0.5,
        step_penalty: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            goal_bonus=goal_bonus,
            collision_penalty=collision_penalty,
            step_penalty=step_penalty,
            **kwargs,
        )

    def __call__(self, state: Any, info: Dict[str, Any]) -> float:
        reward = 0.0
        if self.is_goal_reached(state, info):
            reward += self.goal_bonus
        if self.is_collided(state, info):
            reward -= abs(self.collision_penalty)
        reward += self.step_penalty
        return float(reward)


class DistanceBasedDenseReward(BaseReward):
    """R3: Distance-Based Dense Reward.

    Provides a dense negative reward proportional to the normalized BFS distance
    to goal at every step.
    Formula:
        d_norm = d_bfs_current / d_bfs_initial
        r_t = - distance_scale * d_norm + goal_bonus * [reached_goal] - collision_penalty * [collided] + step_penalty
    """

    def __init__(
        self,
        distance_scale: float = 1.0,
        goal_bonus: float = 0.0,
        step_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            goal_bonus=goal_bonus,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            **kwargs,
        )
        self.distance_scale = float(distance_scale)

    def __call__(self, state: Any, info: Dict[str, Any]) -> float:
        # Initial task distance from info['d_bfs']
        initial_dist = self.get_initial_distance(state, info)

        # Current distance to goal
        if "goal_geodesic" in info and info["goal_geodesic"] is not None:
            curr_dist = float(info["goal_geodesic"])
        elif "d_bfs_curr" in info and info["d_bfs_curr"] is not None:
            curr_dist = float(info["d_bfs_curr"])
        elif hasattr(state, "geodesic_distance") and callable(state.geodesic_distance):
            curr_dist = float(state.geodesic_distance())
        elif hasattr(state, "goal_distance") and callable(state.goal_distance):
            curr_dist = float(state.goal_distance())
        else:
            curr_dist = float(info.get("d_bfs", 0.0))

        # Normalize distance
        if initial_dist > 0:
            d_norm = curr_dist / initial_dist
        else:
            d_norm = curr_dist

        reward = - self.distance_scale * d_norm

        if self.is_goal_reached(state, info):
            reward += self.goal_bonus
        if self.is_collided(state, info):
            reward -= abs(self.collision_penalty)
        reward += self.step_penalty

        return float(reward)


class ProgressDenseReward(BaseReward):
    """R4: Progress Dense Reward.

    Rewards moving closer to the goal cell-by-cell using BFS distance difference:
    d_bfs(prev) - d_bfs(now).
    Uses info['progress'] from NavSim (or tracks info['d_bfs'] / distance across steps).
    Formula:
        progress = d_bfs(t-1) - d_bfs(t)
        r_t = progress_scale * progress + goal_bonus * [reached_goal] - collision_penalty * [collided] + step_penalty
    """

    def __init__(
        self,
        progress_scale: float = 1.0,
        goal_bonus: float = 0.0,
        step_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            goal_bonus=goal_bonus,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            **kwargs,
        )
        self.progress_scale = float(progress_scale)
        self.prev_dist: Optional[float] = None

    def reset(self) -> None:
        self.prev_dist = None

    def __call__(self, state: Any, info: Dict[str, Any]) -> float:
        # Detect new episode if steps == 1
        steps = info.get("steps", getattr(state, "steps", None))
        if steps is not None and steps <= 1:
            self.prev_dist = None

        # 1. Use info['progress'] directly if available (NavSim computes d_bfs_prev - d_bfs_now)
        if "progress" in info and info["progress"] is not None:
            progress = float(info["progress"])
        else:
            curr_dist = self.get_current_distance(state, info)
            if self.prev_dist is None:
                # First step: start from initial distance or current
                init_dist = self.get_initial_distance(state, info)
                progress = float(init_dist - curr_dist)
            else:
                progress = float(self.prev_dist - curr_dist)
            self.prev_dist = curr_dist

        reward = self.progress_scale * progress

        if self.is_goal_reached(state, info):
            reward += self.goal_bonus
        if self.is_collided(state, info):
            reward -= abs(self.collision_penalty)
        reward += self.step_penalty

        return float(reward)


class ProgressGoalReward(ProgressDenseReward):
    """R5: Progress + Goal Bonus Reward.

    Combines cell-by-cell BFS progress with a goal achievement bonus and a step penalty
    to encourage rapid, efficient navigation.
    Formula:
        r_t = progress + 10.0 * [reached_goal] - 0.01
    """

    def __init__(
        self,
        progress_scale: float = 1.0,
        goal_bonus: float = 10.0,
        step_penalty: float = -0.01,
        collision_penalty: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            progress_scale=progress_scale,
            goal_bonus=goal_bonus,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            **kwargs,
        )


class PotentialBasedShapedReward(BaseReward):
    """R6: Potential-Based Shaped Reward.

    Theoretically guaranteed policy invariance under reward transformation (Ng et al., 1999).
    Potential function:
        Phi(s) = - d_bfs(s, goal)
        Phi(goal) = 0.0
    Shaping term:
        F(s, s') = gamma * Phi(s') - Phi(s)
    Formula:
        r_t = potential_scale * F(s, s') + goal_bonus * [reached_goal] - collision_penalty * [collided] + step_penalty
    """

    def __init__(
        self,
        gamma: float = 1.0,
        potential_scale: float = 1.0,
        goal_bonus: float = 0.0,
        step_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            goal_bonus=goal_bonus,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            **kwargs,
        )
        self.gamma = float(gamma)
        self.potential_scale = float(potential_scale)
        self.prev_potential: Optional[float] = None

    def reset(self) -> None:
        self.prev_potential = None

    def __call__(self, state: Any, info: Dict[str, Any]) -> float:
        # Detect new episode if steps == 1
        steps = info.get("steps", getattr(state, "steps", None))
        if steps is not None and steps <= 1:
            self.prev_potential = None

        reached = self.is_goal_reached(state, info)
        curr_dist = self.get_current_distance(state, info)

        # Potential at goal is 0.0; otherwise -d_bfs
        curr_potential = 0.0 if reached else - float(curr_dist)

        if self.prev_potential is None:
            init_dist = self.get_initial_distance(state, info)
            self.prev_potential = - float(init_dist)

        # F(s, s') = gamma * Phi(s') - Phi(s)
        f_shaping = self.gamma * curr_potential - self.prev_potential
        self.prev_potential = curr_potential

        reward = self.potential_scale * f_shaping

        if reached:
            reward += self.goal_bonus
        if self.is_collided(state, info):
            reward -= abs(self.collision_penalty)
        reward += self.step_penalty

        return float(reward)
