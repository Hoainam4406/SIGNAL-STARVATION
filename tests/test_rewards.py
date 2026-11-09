"""Unit and integration tests for NavSim reward functions (R1 - R6).

Tests satisfy the DoD criteria:
- pytest tests/test_rewards.py PASS
- Run 100 steps with each of the 6 reward types without crash
"""

import math
import sys
from pathlib import Path
import tempfile

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import gymnasium as gym
import numpy as np
import pytest
import yaml

import navsim
import navsim.wrappers
from src.rewards import (
    BaseReward,
    PureSparseReward,
    CollisionPenalizedSparseReward,
    DistanceBasedDenseReward,
    ProgressDenseReward,
    ProgressGoalReward,
    PotentialBasedShapedReward,
    make_reward,
    wrap_env_with_reward,
    register_reward,
    REWARD_REGISTRY,
)


# =============================================================================
# Mock State & Info for isolated unit tests
# =============================================================================

class MockRobotState:
    def __init__(
        self,
        reached_goal: bool = False,
        collided: bool = False,
        geodesic_dist: float = 10.0,
        start_geodesic: int = 20,
        steps: int = 1,
    ):
        self.reached_goal = reached_goal
        self.collided = collided
        self._geodesic_dist = geodesic_dist
        self.start_geodesic = start_geodesic
        self.steps = steps

    def geodesic_distance(self) -> float:
        return float(self._geodesic_dist)

    def goal_distance(self) -> float:
        return float(self._geodesic_dist)


# =============================================================================
# Unit Tests for R1 - R6 Classes
# =============================================================================

class TestR1PureSparseReward:
    def test_default_sparse_step(self):
        r_fn = PureSparseReward()
        state = MockRobotState(reached_goal=False)
        info = {"steps": 1, "d_bfs": 20}
        assert r_fn(state, info) == 0.0

    def test_goal_reached(self):
        r_fn = PureSparseReward(goal_bonus=1.0)
        state = MockRobotState(reached_goal=True)
        info = {"steps": 5, "d_bfs": 20, "success": True}
        assert r_fn(state, info) == 1.0

    def test_custom_parameters(self):
        r_fn = PureSparseReward(goal_bonus=10.0, step_penalty=-0.05)
        # Normal step
        state = MockRobotState(reached_goal=False)
        info = {"steps": 1, "d_bfs": 20}
        assert pytest.approx(r_fn(state, info)) == -0.05

        # Goal step
        state_goal = MockRobotState(reached_goal=True)
        assert pytest.approx(r_fn(state_goal, info)) == 9.95


class TestR2CollisionPenalizedSparseReward:
    def test_normal_step(self):
        r_fn = CollisionPenalizedSparseReward(goal_bonus=1.0, collision_penalty=0.5)
        state = MockRobotState(reached_goal=False, collided=False)
        info = {"steps": 1, "d_bfs": 20, "collided": False}
        assert r_fn(state, info) == 0.0

    def test_collision_step(self):
        r_fn = CollisionPenalizedSparseReward(goal_bonus=1.0, collision_penalty=0.5)
        state = MockRobotState(reached_goal=False, collided=True)
        info = {"steps": 1, "d_bfs": 20, "collided": True}
        assert pytest.approx(r_fn(state, info)) == -0.5

    def test_goal_reached(self):
        r_fn = CollisionPenalizedSparseReward(goal_bonus=1.0, collision_penalty=0.5)
        state = MockRobotState(reached_goal=True, collided=False)
        info = {"steps": 10, "d_bfs": 20, "collided": False}
        assert pytest.approx(r_fn(state, info)) == 1.0

    def test_goal_and_collision(self):
        r_fn = CollisionPenalizedSparseReward(goal_bonus=1.0, collision_penalty=0.5)
        state = MockRobotState(reached_goal=True, collided=True)
        info = {"steps": 10, "d_bfs": 20, "collided": True}
        assert pytest.approx(r_fn(state, info)) == 0.5


class TestR3DistanceBasedDenseReward:
    def test_normalized_distance(self):
        r_fn = DistanceBasedDenseReward(distance_scale=1.0)
        state = MockRobotState(reached_goal=False, geodesic_dist=10.0, start_geodesic=20)
        info = {"steps": 1, "d_bfs": 20, "goal_geodesic": 10.0}
        # d_norm = 10 / 20 = 0.5 -> reward = -0.5
        assert pytest.approx(r_fn(state, info)) == -0.5

    def test_at_goal(self):
        r_fn = DistanceBasedDenseReward(distance_scale=1.0, goal_bonus=10.0)
        state = MockRobotState(reached_goal=True, geodesic_dist=0.0, start_geodesic=20)
        info = {"steps": 15, "d_bfs": 20, "goal_geodesic": 0.0}
        # d_norm = 0 / 20 = 0.0 -> reward = -0.0 + 10.0 = 10.0
        assert pytest.approx(r_fn(state, info)) == 10.0


class TestR4ProgressDenseReward:
    def test_progress_from_info(self):
        r_fn = ProgressDenseReward(progress_scale=1.0)
        state = MockRobotState(reached_goal=False)
        # 1 step closer to goal
        info = {"steps": 2, "d_bfs": 20, "progress": 1.0}
        assert pytest.approx(r_fn(state, info)) == 1.0

        # 1 step away from goal
        info_away = {"steps": 3, "d_bfs": 20, "progress": -1.0}
        assert pytest.approx(r_fn(state, info_away)) == -1.0

    def test_progress_tracked_without_info_key(self):
        r_fn = ProgressDenseReward(progress_scale=1.0)
        state1 = MockRobotState(reached_goal=False, geodesic_dist=15.0, steps=1)
        info1 = {"steps": 1, "d_bfs": 20, "goal_geodesic": 15.0}
        # Step 1: initial is 20, curr is 15 -> progress = +5
        assert pytest.approx(r_fn(state1, info1)) == 5.0

        state2 = MockRobotState(reached_goal=False, geodesic_dist=14.0, steps=2)
        info2 = {"steps": 2, "d_bfs": 20, "goal_geodesic": 14.0}
        # Step 2: prev is 15, curr is 14 -> progress = +1
        assert pytest.approx(r_fn(state2, info2)) == 1.0

    def test_yaml_example_parameters(self):
        # Config as given in prompt: goal_bonus=10.0, step_penalty=-0.01
        r_fn = ProgressDenseReward(progress_scale=1.0, goal_bonus=10.0, step_penalty=-0.01)
        state = MockRobotState(reached_goal=False)
        info = {"steps": 2, "d_bfs": 20, "progress": 1.0}
        # progress=1.0, step_penalty=-0.01 -> 0.99
        assert pytest.approx(r_fn(state, info)) == 0.99

        state_goal = MockRobotState(reached_goal=True)
        info_goal = {"steps": 15, "d_bfs": 20, "progress": 1.0}
        # progress=1.0 + goal_bonus=10.0 - 0.01 = 10.99
        assert pytest.approx(r_fn(state_goal, info_goal)) == 10.99


class TestR5ProgressGoalReward:
    def test_default_values(self):
        r_fn = ProgressGoalReward()
        assert r_fn.goal_bonus == 10.0
        assert r_fn.step_penalty == -0.01

        state = MockRobotState(reached_goal=False)
        info = {"steps": 2, "d_bfs": 20, "progress": 1.0}
        assert pytest.approx(r_fn(state, info)) == 0.99

        state_goal = MockRobotState(reached_goal=True)
        assert pytest.approx(r_fn(state_goal, info)) == 10.99


class TestR6PotentialBasedShapedReward:
    def test_gamma_one_undiscounted(self):
        r_fn = PotentialBasedShapedReward(gamma=1.0)
        # Step 1: d_0 = 20, d_1 = 19
        state1 = MockRobotState(reached_goal=False, geodesic_dist=19.0, steps=1)
        info1 = {"steps": 1, "d_bfs": 20, "goal_geodesic": 19.0}
        # F = (-19) - (-20) = 1.0
        assert pytest.approx(r_fn(state1, info1)) == 1.0

        # Step 2: d_2 = 18
        state2 = MockRobotState(reached_goal=False, geodesic_dist=18.0, steps=2)
        info2 = {"steps": 2, "d_bfs": 20, "goal_geodesic": 18.0}
        # F = (-18) - (-19) = 1.0
        assert pytest.approx(r_fn(state2, info2)) == 1.0

    def test_terminal_potential(self):
        r_fn = PotentialBasedShapedReward(gamma=1.0)
        # Set prev potential at d=1
        r_fn.prev_potential = -1.0
        state = MockRobotState(reached_goal=True, geodesic_dist=0.0, steps=10)
        info = {"steps": 10, "d_bfs": 20, "goal_geodesic": 0.0}
        # Phi(goal) = 0.0 -> F = 0.0 - (-1.0) = 1.0
        assert pytest.approx(r_fn(state, info)) == 1.0

    def test_discounted_gamma(self):
        r_fn = PotentialBasedShapedReward(gamma=0.9)
        r_fn.prev_potential = -10.0
        state = MockRobotState(reached_goal=False, geodesic_dist=9.0, steps=2)
        info = {"steps": 2, "d_bfs": 20, "goal_geodesic": 9.0}
        # F = 0.9 * (-9.0) - (-10.0) = -8.1 + 10.0 = 1.9
        assert pytest.approx(r_fn(state, info)) == 1.9


# =============================================================================
# Factory Tests
# =============================================================================

class TestRewardFactory:
    def test_make_reward_from_string(self):
        r1 = make_reward("pure_sparse")
        assert isinstance(r1, PureSparseReward)
        r4 = make_reward("progress_dense")
        assert isinstance(r4, ProgressDenseReward)
        r6 = make_reward("potential_shaped")
        assert isinstance(r6, PotentialBasedShapedReward)

    def test_make_reward_from_dict(self):
        cfg = {
            "type": "progress_dense",
            "goal_bonus": 10.0,
            "step_penalty": -0.01,
        }
        r = make_reward(cfg)
        assert isinstance(r, ProgressDenseReward)
        assert r.goal_bonus == 10.0
        assert r.step_penalty == -0.01

    def test_make_reward_from_nested_dict(self):
        cfg = {
            "reward": {
                "type": "progress_dense",
                "goal_bonus": 10.0,
                "step_penalty": -0.01,
            }
        }
        r = make_reward(cfg)
        assert isinstance(r, ProgressDenseReward)
        assert r.goal_bonus == 10.0
        assert r.step_penalty == -0.01

    def test_make_reward_from_yaml_file(self):
        yaml_content = """
reward:
  type: "progress_dense"  # R4
  goal_bonus: 10.0
  step_penalty: -0.01
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            r = make_reward(temp_path)
            assert isinstance(r, ProgressDenseReward)
            assert r.goal_bonus == 10.0
            assert r.step_penalty == -0.01
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown reward type"):
            make_reward({"type": "non_existent_reward"})

    def test_custom_registration(self):
        class CustomReward(BaseReward):
            def __call__(self, state, info):
                return 42.0

        register_reward("my_custom_reward", CustomReward)
        r = make_reward("my_custom_reward")
        assert isinstance(r, CustomReward)
        assert r(None, {}) == 42.0


# =============================================================================
# Integration Tests: Running 100 steps on each reward type (DoD criteria)
# =============================================================================

ALL_REWARD_TYPES = [
    "pure_sparse",                   # R1
    "collision_penalized_sparse",    # R2
    "distance_dense",                # R3
    "progress_dense",                # R4
    "progress_goal",                 # R5
    "potential_shaped",              # R6
]


@pytest.mark.parametrize("reward_type", ALL_REWARD_TYPES)
def test_100_steps_navsim_reward_wrapper(reward_type: str):
    """Run 100 steps in NavSim GridNav-v0 using navsim.wrappers.Reward API without crashing."""
    raw_env = gym.make("NavSim/GridNav-v0", dataset="demo-grid")

    # Instantiate reward function
    reward_fn = make_reward({
        "type": reward_type,
        "goal_bonus": 10.0,
        "step_penalty": -0.01,
    })

    # Wrapper explicitly adhering to:
    # navsim.wrappers.Reward(env, lambda state, info: reward_fn(state, info))
    env = navsim.wrappers.Reward(raw_env, lambda state, info: reward_fn(state, info))

    obs, info = env.reset(seed=42)
    assert obs is not None
    assert isinstance(info, dict)

    steps_completed = 0
    episodes_completed = 0

    for step_i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)

        # Check return types and values
        assert isinstance(reward, float)
        assert math.isfinite(reward), f"Reward is not finite at step {step_i}: {reward}"
        assert obs is not None
        assert isinstance(step_info, dict)

        steps_completed += 1

        if terminated or truncated:
            episodes_completed += 1
            reward_fn.reset()
            obs, info = env.reset()

    env.close()
    assert steps_completed == 100
    print(f"\n[OK] Reward {reward_type} completed 100 steps across {episodes_completed} episode boundaries.")


def test_wrap_env_with_reward_helper():
    """Verify wrap_env_with_reward utility function end-to-end."""
    raw_env = gym.make("NavSim/GridNav-v0", dataset="demo-grid")
    env = wrap_env_with_reward(raw_env, {
        "reward": {
            "type": "progress_dense",
            "goal_bonus": 10.0,
            "step_penalty": -0.01,
        }
    })

    assert isinstance(env, navsim.wrappers.Reward)
    assert hasattr(env, "reward_fn")
    assert isinstance(env.reward_fn, ProgressDenseReward)

    obs, info = env.reset(seed=123)
    obs, r, term, trunc, step_info = env.step(env.action_space.sample())
    assert isinstance(r, float)
    assert math.isfinite(r)
    env.close()
