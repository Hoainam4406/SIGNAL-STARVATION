"""Unit tests for agent factory and algorithm YAML configurations (TASK-302).

Verifies DoD criteria:
- make_agent('configs/ppo_default.yaml', env) returns SB3 PPO model without error
- make_agent works for PPO, DQN, A2C
- 3 YAML files are complete with comments citing respective papers
- Runtime parameter overrides and error handling
"""

from pathlib import Path
import sys
import pytest
import gymnasium as gym
import navsim
from stable_baselines3 import A2C, DQN, PPO

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.agents.agent_factory import make_agent, load_agent_config, AGENT_REGISTRY


@pytest.fixture(scope="module")
def gridnav_env():
    """Create a standard NavSim GridNav-v0 environment for test suite."""
    env = gym.make("NavSim/GridNav-v0", dataset="grid-turtlebot4-v2", split="train")
    yield env
    env.close()


# =============================================================================
# 1. YAML Configuration & Paper Citations Verification
# =============================================================================

def test_yaml_files_exist_and_cite_papers():
    """Check that ppo, dqn, and a2c YAML files exist and contain proper paper citations."""
    config_dir = root_dir / "configs"

    ppo_path = config_dir / "ppo_default.yaml"
    dqn_path = config_dir / "dqn_default.yaml"
    a2c_path = config_dir / "a2c_default.yaml"

    assert ppo_path.exists()
    assert dqn_path.exists()
    assert a2c_path.exists()

    with open(ppo_path, "r", encoding="utf-8") as f:
        ppo_text = f.read()
    with open(dqn_path, "r", encoding="utf-8") as f:
        dqn_text = f.read()
    with open(a2c_path, "r", encoding="utf-8") as f:
        a2c_text = f.read()

    # Verify Schulman et al. (2017)
    assert "Schulman et al. (2017)" in ppo_text
    assert "1707.06347" in ppo_text

    # Verify Mnih et al. (2015)
    assert "Mnih et al. (2015)" in dqn_text
    assert "nature" in dqn_text.lower()

    # Verify Mnih et al. (2016)
    assert "Mnih et al. (2016)" in a2c_text
    assert "1602.01783" in a2c_text


# =============================================================================
# 2. make_agent Verification (DoD: PPO, DQN, A2C instantiation)
# =============================================================================

def test_make_agent_ppo_dod(gridnav_env):
    """DoD verification: make_agent('configs/ppo_default.yaml', env) returns SB3 PPO model."""
    ppo_config_path = root_dir / "configs" / "ppo_default.yaml"
    model = make_agent(ppo_config_path, gridnav_env, verbose=0)

    assert isinstance(model, PPO)
    assert model.learning_rate == pytest.approx(3.0e-4)
    assert model.n_steps == 2048
    assert model.batch_size == 64
    assert model.gamma == pytest.approx(0.99)
    assert model.ent_coef == pytest.approx(0.01)


def test_make_agent_dqn(gridnav_env):
    """Verify make_agent with dqn_default.yaml."""
    dqn_config_path = root_dir / "configs" / "dqn_default.yaml"
    model = make_agent(dqn_config_path, gridnav_env, verbose=0)

    assert isinstance(model, DQN)
    assert model.learning_rate == pytest.approx(1.0e-4)
    assert model.buffer_size == 100_000
    assert model.learning_starts == 10_000
    assert model.batch_size == 32
    assert model.gamma == pytest.approx(0.99)


def test_make_agent_a2c(gridnav_env):
    """Verify make_agent with a2c_default.yaml."""
    a2c_config_path = root_dir / "configs" / "a2c_default.yaml"
    model = make_agent(a2c_config_path, gridnav_env, verbose=0)

    assert isinstance(model, A2C)
    assert model.learning_rate == pytest.approx(7.0e-4)
    assert model.n_steps == 5
    assert model.gamma == pytest.approx(0.99)


# =============================================================================
# 3. Parameter Overrides & Dict Configuration
# =============================================================================

def test_make_agent_runtime_overrides(gridnav_env):
    """Test overriding parameters at runtime via make_agent kwargs."""
    ppo_config_path = root_dir / "configs" / "ppo_default.yaml"
    model = make_agent(
        ppo_config_path,
        gridnav_env,
        learning_rate=1.0e-4,
        seed=123,
        verbose=0,
    )
    assert model.learning_rate == pytest.approx(1.0e-4)
    assert model.seed == 123


def test_make_agent_from_dict(gridnav_env):
    """Test instantiating agent directly from python dictionary."""
    config_dict = {
        "algorithm": "ppo",
        "policy": "MlpPolicy",
        "learning_rate": 5e-4,
        "n_steps": 128,
        "batch_size": 32,
    }
    model = make_agent(config_dict, gridnav_env, verbose=0)
    assert isinstance(model, PPO)
    assert model.n_steps == 128
    assert model.batch_size == 32


# =============================================================================
# 4. Error Handling Tests
# =============================================================================

def test_nonexistent_config_file_raises(gridnav_env):
    with pytest.raises(FileNotFoundError):
        make_agent("configs/non_existent.yaml", gridnav_env)


def test_missing_algorithm_raises(gridnav_env):
    with pytest.raises(ValueError, match="must specify an 'algorithm'"):
        make_agent({"policy": "MlpPolicy"}, gridnav_env)


def test_unsupported_algorithm_raises(gridnav_env):
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        make_agent({"algorithm": "unsupported_algo"}, gridnav_env)
