"""Unit and integration tests for Task Filter Wrappers (Axis A of Signal Starvation).

Verifies DoD criteria:
- After setting filter, 100 resets consistently return tasks matching the chosen criteria.
- LevelFilter, SAFilter, StarvedFilter operate correctly individually and when chained.
- Input validation and error handling behave as expected.
"""

from pathlib import Path
import sys
import pytest
import gymnasium as gym
import navsim

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.envs.task_filter_wrapper import (
    BaseTaskFilterWrapper,
    LevelFilter,
    SAFilter,
    StarvedFilter,
    make_task_filtered_env,
)


@pytest.fixture(scope="function")
def base_gridnav_env():
    """Create a fresh standard NavSim GridNav-v0 environment for each test."""
    env = gym.make("NavSim/GridNav-v0", dataset="grid-turtlebot4-v2", split="train")
    yield env
    env.close()


# =============================================================================
# 1. LevelFilter Tests (DoD: 100 resets return tasks of chosen level)
# =============================================================================

def test_level_filter_100_resets_easy(base_gridnav_env):
    """DoD verification: 100 consecutive resets on LevelFilter(levels='easy')

    must all return tasks with level == 'easy'.
    """
    env = LevelFilter(base_gridnav_env, levels="easy")
    stats = env.get_filter_stats()
    assert stats["tasks"] > 0
    assert stats["maps"] > 0

    seen_levels = set()
    for _ in range(100):
        _, info = env.reset()
        seen_levels.add(info["level"])
        assert info["level"] == "easy", f"Expected level 'easy', got {info['level']}"

    assert seen_levels == {"easy"}


def test_level_filter_multiple_levels(base_gridnav_env):
    """Test LevelFilter with a list of levels: ['easy', 'medium']."""
    env = LevelFilter(base_gridnav_env, levels=["easy", "medium"])
    stats = env.get_filter_stats()
    assert stats["tasks"] > 0

    seen_levels = set()
    for _ in range(50):
        _, info = env.reset()
        seen_levels.add(info["level"])
        assert info["level"] in {"easy", "medium"}

    # Should observe both levels across 50 resets
    assert "easy" in seen_levels
    assert "medium" in seen_levels


def test_level_filter_dynamic_update(base_gridnav_env):
    """Test dynamically changing levels on the wrapper."""
    env = LevelFilter(base_gridnav_env, levels="easy")
    _, info1 = env.reset()
    assert info1["level"] == "easy"

    env.set_levels("hard")
    for _ in range(20):
        _, info2 = env.reset()
        assert info2["level"] == "hard"


# =============================================================================
# 2. SAFilter Tests (Signal density / random-walk reachability range)
# =============================================================================

def test_sa_filter_range(base_gridnav_env):
    """Test SAFilter with sa_min=0.1, sa_max=1.0."""
    env = SAFilter(base_gridnav_env, sa_min=0.1, sa_max=1.0)
    stats = env.get_filter_stats()
    assert stats["tasks"] > 0

    for _ in range(50):
        _, info = env.reset()
        sa = info["sa_uniform"]
        assert sa >= 0.1 - 1e-6, f"sa_uniform {sa} is below sa_min 0.1"
        assert sa <= 1.0 + 1e-6, f"sa_uniform {sa} is above sa_max 1.0"


def test_sa_filter_low_signal_starved(base_gridnav_env):
    """Test SAFilter selecting only extreme starved tasks (sa_uniform < 0.0001)."""
    env = SAFilter(base_gridnav_env, sa_min=0.0, sa_max=0.0001)
    stats = env.get_filter_stats()
    assert stats["tasks"] > 0

    for _ in range(30):
        _, info = env.reset()
        assert info["sa_uniform"] <= 0.0001 + 1e-9


# =============================================================================
# 3. StarvedFilter Tests
# =============================================================================

def test_starved_filter_exclude_starved(base_gridnav_env):
    """Test StarvedFilter(include_starved=False) strictly excludes starved tasks."""
    env = StarvedFilter(base_gridnav_env, include_starved=False)
    stats = env.get_filter_stats()
    assert stats["tasks"] == 1521  # Verified in audit

    for _ in range(50):
        _, info = env.reset()
        assert info["starved"] is False
        assert info["level"] in {"easy", "medium", "hard"}


def test_starved_filter_only_starved(base_gridnav_env):
    """Test StarvedFilter(include_starved=True) strictly selects starved tasks."""
    env = StarvedFilter(base_gridnav_env, include_starved=True)
    stats = env.get_filter_stats()
    assert stats["tasks"] == 439  # Verified in audit

    for _ in range(50):
        _, info = env.reset()
        assert info["starved"] is True
        assert info["level"] == "starved"


# =============================================================================
# 4. Filter Chaining & Merging Tests
# =============================================================================

def test_chained_filters(base_gridnav_env):
    """Test chaining LevelFilter and StarvedFilter without clobbering criteria."""
    # First wrap with level=['easy', 'medium', 'hard']
    env1 = LevelFilter(base_gridnav_env, levels=["easy", "medium", "hard"], merge_criteria=True)
    # Then wrap with StarvedFilter(include_starved=False)
    env2 = StarvedFilter(env1, include_starved=False, merge_criteria=True)

    for _ in range(40):
        _, info = env2.reset()
        assert info["level"] in {"easy", "medium", "hard"}
        assert info["starved"] is False


def test_make_task_filtered_env_helper(base_gridnav_env):
    """Test convenience factory make_task_filtered_env."""
    filtered_env = make_task_filtered_env(
        base_gridnav_env,
        levels=["easy"],
        sa_min=0.1,
        include_starved=False,
    )
    for _ in range(30):
        _, info = filtered_env.reset()
        assert info["level"] == "easy"
        assert info["sa_uniform"] >= 0.1 - 1e-6
        assert info["starved"] is False


# =============================================================================
# 5. Validation and Error Handling Tests
# =============================================================================

def test_invalid_level_raises(base_gridnav_env):
    with pytest.raises(ValueError, match="Invalid level"):
        LevelFilter(base_gridnav_env, levels="non_existent_level")

    with pytest.raises(ValueError, match="must contain at least one"):
        LevelFilter(base_gridnav_env, levels=[])


def test_invalid_sa_range_raises(base_gridnav_env):
    with pytest.raises(ValueError, match="cannot be greater"):
        SAFilter(base_gridnav_env, sa_min=0.8, sa_max=0.2)


def test_clear_filter_restores_all_tasks(base_gridnav_env):
    env = LevelFilter(base_gridnav_env, levels="easy")
    assert env.get_filter_stats()["tasks"] == 673

    stats_restored = env.clear_filter()
    assert stats_restored["tasks"] == 1960  # All train tasks
