"""Environment and DRL Dependency Verification Script for Signal Starvation Project.

Verifies:
1. Environment variables (NAVSIM_DATASETS)
2. Core dependencies and version inspection:
   - stable_baselines3, torch, gymnasium, navsim, tensorboard, sb3_contrib, pandas, seaborn, statsmodels, omegaconf
3. SB3 Algorithms: PPO, DQN, A2C import and compatibility
4. Quick functional test with NavSim GridNav-v0:
   - Environment creation
   - Observation and Action space compatibility
   - Reset, step execution, and model instantiation
"""

import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent.parent
    dotenv_file = project_root / ".env"
    if dotenv_file.exists():
        load_dotenv(dotenv_file)
    else:
        load_dotenv()
except ImportError:
    pass


def main():
    print("=" * 72)
    print("DRL ENVIRONMENT & DEPENDENCY VERIFICATION (SB3 + NAVSIM)")
    print("=" * 72)

    all_passed = True

    # -------------------------------------------------------------------------
    # 1. Environment Variable Check
    # -------------------------------------------------------------------------
    print("\n[1/4] Checking Environment Variables...")
    navsim_datasets = os.environ.get("NAVSIM_DATASETS")
    expected_path = r"F:\navsim\datasets"

    if navsim_datasets:
        if os.path.exists(navsim_datasets):
            print(f"  [OK] NAVSIM_DATASETS = {navsim_datasets} (Directory exists)")
        else:
            print(f"  [WARNING] NAVSIM_DATASETS = {navsim_datasets} (Path not found on disk)")
    else:
        print(f"  [INFO] NAVSIM_DATASETS not in process env. Setting fallback: {expected_path}")
        os.environ["NAVSIM_DATASETS"] = expected_path
        if os.path.exists(expected_path):
            print(f"  [OK] Fallback path verified: {expected_path}")

    # -------------------------------------------------------------------------
    # 2. Package Import and Version Check
    # -------------------------------------------------------------------------
    print("\n[2/4] Verifying Core Packages & Versions...")
    packages = [
        ("gymnasium", "Gymnasium"),
        ("navsim", "NavSim"),
        ("torch", "PyTorch"),
        ("tensorboard", "TensorBoard"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("sb3_contrib", "SB3-Contrib"),
        ("pandas", "Pandas"),
        ("seaborn", "Seaborn"),
        ("statsmodels", "Statsmodels"),
        ("omegaconf", "OmegaConf"),
    ]

    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {label:<22} : v{ver}")
        except Exception as e:
            print(f"  [FAILED] {label:<22} : {e}")
            all_passed = False

    # -------------------------------------------------------------------------
    # 3. Verify SB3 Algorithms (PPO, DQN, A2C)
    # -------------------------------------------------------------------------
    print("\n[3/4] Verifying Stable-Baselines3 Algorithms...")
    try:
        from stable_baselines3 import PPO, DQN, A2C
        print("  [OK] Successfully imported PPO, DQN, A2C from stable_baselines3")
        algos = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
        for name, cls in algos.items():
            print(f"  [OK] Algorithm class {name:<4} : {cls}")
    except Exception as e:
        print(f"  [FAILED] Could not import PPO, DQN, A2C: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # 4. Quick Functional Test with NavSim GridNav-v0
    # -------------------------------------------------------------------------
    print("\n[4/4] Quick Simulation & SB3 Model Instantiation Test...")
    try:
        import gymnasium as gym
        import navsim
        from stable_baselines3 import PPO, DQN, A2C

        dataset = "grid-turtlebot4-v2"
        env = gym.make("NavSim/GridNav-v0", dataset=dataset, split="train")

        obs_space = env.observation_space
        act_space = env.action_space
        print(f"  [OK] Environment instantiated: NavSim/GridNav-v0 ({dataset})")
        print(f"  [OK] Observation Space      : {obs_space}")
        print(f"  [OK] Action Space           : {act_space}")

        # Quick rollout of 10 random steps
        obs, info = env.reset(seed=42)
        for _ in range(10):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                obs, info = env.reset()
        print("  [OK] 10 random simulation steps executed successfully")

        # Test policy compatibility by instantiating PPO, DQN, A2C on env
        ppo_model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
        print("  [OK] PPO('MlpPolicy', env) initialized successfully")

        dqn_model = DQN("MlpPolicy", env, buffer_size=1000, learning_starts=10, verbose=0)
        print("  [OK] DQN('MlpPolicy', env) initialized successfully")

        a2c_model = A2C("MlpPolicy", env, n_steps=16, verbose=0)
        print("  [OK] A2C('MlpPolicy', env) initialized successfully")

        env.close()
    except Exception as e:
        print(f"  [FAILED] Functional test failed: {e}")
        all_passed = False

    print("\n" + "=" * 72)
    if all_passed:
        print("[SUCCESS] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! [OK]")
    else:
        print("[FAILURE] ONE OR MORE VERIFICATION CHECKS FAILED.")
        sys.exit(1)
    print("=" * 72)


if __name__ == "__main__":
    main()
