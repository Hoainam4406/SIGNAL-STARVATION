"""Agent Factory for instantiating Stable-Baselines3 algorithms from YAML configurations.

Supports:
- PPO (Proximal Policy Optimization) — Schulman et al., 2017
- DQN (Deep Q-Network) — Mnih et al., 2015
- A2C (Advantage Actor-Critic) — Mnih et al., 2016
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
import gymnasium as gym
import yaml
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

AGENT_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}


def load_agent_config(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load and parse an agent configuration dictionary from a file or dict.

    Args:
        config: Path to a YAML configuration file or an existing dictionary.

    Returns:
        dict: Parsed agent configuration dictionary.
    """
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML file to contain a mapping/dictionary, got {type(data)}")
        return data
    elif isinstance(config, dict):
        return dict(config)
    else:
        raise TypeError(f"config must be a path (str, Path) or dict, got {type(config)}")


def make_agent(
    config: Union[str, Path, Dict[str, Any]],
    env: gym.Env,
    **kwargs: Any,
) -> BaseAlgorithm:
    """Instantiate a Stable-Baselines3 algorithm model from configuration.

    Args:
        config: Path to a YAML configuration file or a config dictionary.
        env: Gymnasium environment instance.
        **kwargs: Optional runtime overrides (e.g. seed, tensorboard_log, verbose).

    Returns:
        BaseAlgorithm: An initialized SB3 algorithm instance (PPO, DQN, or A2C).

    Raises:
        ValueError: If the specified algorithm is unknown or required parameters are missing.
    """
    cfg = load_agent_config(config)

    # Merge optional runtime overrides
    cfg.update(kwargs)

    if "algorithm" not in cfg:
        raise ValueError("Configuration must specify an 'algorithm' key (e.g., 'ppo', 'dqn', 'a2c').")

    algo_name = str(cfg.pop("algorithm")).lower().strip()
    if algo_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unsupported algorithm '{algo_name}'. Available algorithms: {sorted(AGENT_REGISTRY.keys())}"
        )

    algo_cls = AGENT_REGISTRY[algo_name]
    policy = cfg.pop("policy", "MlpPolicy")

    # Clean up parameters: convert float learning rates if written in scientific notation string
    if "learning_rate" in cfg and isinstance(cfg["learning_rate"], str):
        try:
            cfg["learning_rate"] = float(cfg["learning_rate"])
        except ValueError:
            pass

    return algo_cls(policy=policy, env=env, **cfg)
