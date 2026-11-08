import os
import sys
import time
from pathlib import Path

# Thêm src vào sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from experiment import ConfigManager, ExperimentLogger
from aggregate_results import aggregate_results


def run_mock_experiment():
    print("=" * 70)
    print("STARTING MOCK DRL EXPERIMENT RUN (3 SEEDS)")
    print("=" * 70)

    config_path = project_root / "configs" / "default.yaml"
    cfg = ConfigManager(config_path)

    seeds = [42, 123, 999]
    experiment_name = "mock_signal_starvation"
    created_runs = []

    for seed in seeds:
        print(f"\n---> Running mock experiment with SEED = {seed} ...")
        logger = ExperimentLogger(
            experiment_name=experiment_name,
            seed=seed,
            algorithm=cfg.get("experiment.algorithm", "PPO"),
            dataset=cfg.get("experiment.dataset", "navsim_mini"),
            robot=cfg.get("experiment.robot", "differential_drive"),
            reward_type=cfg.get("experiment.reward_type", "sparse"),
            level_filter=cfg.get("experiment.level_filter", "all"),
            config=cfg,
            use_tensorboard=True,
        )
        created_runs.append(logger.run_dir)

        # 1. Giả lập quá trình huấn luyện (Training loop)
        for ep in range(1, 6):
            reward = -20.0 + (ep * 8.5) + (seed % 5)
            steps = 150 - (ep * 10)
            elapsed = ep * 0.15
            logger.log_train_metric(episode=ep, reward=reward, steps=steps, elapsed_time=elapsed)
            logger.log_scalar("train/learning_rate", 0.0003, step=ep)

        # 2. Giả lập quá trình đánh giá (Evaluation)
        for eval_idx, ep in enumerate([2, 5]):
            total_reward = 15.0 + (ep * 5.0) + (seed % 7)
            success = 1 if ep == 5 else 0
            steps_to_goal = 85 if success else 150
            collision_count = 0 if success else 2
            spl = 0.78 if success else 0.0
            mean_progress = 0.95 if success else 0.45
            sa_uniform_mean = 0.62

            logger.log_eval_result(
                episode=ep,
                total_reward=total_reward,
                success=success,
                steps_to_goal=steps_to_goal,
                collision_count=collision_count,
                spl=spl,
                mean_progress=mean_progress,
                sa_uniform_mean=sa_uniform_mean,
                training_time_s=ep * 0.25,
            )

        logger.close()
        print(f"  [OK] Saved run to: {logger.run_dir}")

    # 3. Gom kết quả nhiều seeds bằng script aggregate_results
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print("=" * 70)
    summary_df = aggregate_results(
        results_dir=project_root / "results",
        experiment_name=experiment_name,
    )

    # 4. Kiểm tra các tiêu chí Definition of Done (DoD Verification)
    print("\n" + "=" * 70)
    print("VERIFYING DEFINITION OF DONE")
    print("=" * 70)

    all_passed = True
    required_cols = [
        "experiment_id",
        "seed",
        "algorithm",
        "dataset",
        "robot",
        "reward_type",
        "level_filter",
        "episode",
        "total_reward",
        "success",
        "steps_to_goal",
        "collision_count",
        "spl",
        "mean_progress",
        "sa_uniform_mean",
        "training_time_s",
        "eval_timestamp",
    ]

    for run_dir in created_runs:
        eval_csv = run_dir / "eval_results.csv"
        train_csv = run_dir / "train_metrics.csv"
        meta_json = run_dir / "metadata.json"
        snap_yaml = run_dir / "config_snapshot.yaml"
        tb_dir = run_dir / "tensorboard"

        # Check CSV
        if not eval_csv.exists() or not train_csv.exists():
            print(f"  [FAILED] Missing CSV files in {run_dir}")
            all_passed = False
        else:
            import pandas as pd
            df = pd.read_csv(eval_csv)
            if list(df.columns) == required_cols:
                print(f"  [PASSED] eval_results.csv format matched exactly ({len(df)} rows)")
            else:
                print(f"  [FAILED] eval_results.csv columns mismatch: {list(df.columns)}")
                all_passed = False

        # Check metadata & config snapshot
        if meta_json.exists() and snap_yaml.exists():
            print(f"  [PASSED] metadata.json and config_snapshot.yaml created.")
        else:
            print(f"  [FAILED] Missing metadata or snapshot in {run_dir}")
            all_passed = False

        # Check TensorBoard event files
        tb_files = list(tb_dir.glob("events.out.tfevents.*")) if tb_dir.exists() else []
        if tb_files and all(f.stat().st_size > 0 for f in tb_files):
            print(f"  [PASSED] TensorBoard event log created: {tb_files[0].name} ({tb_files[0].stat().st_size} bytes)")
        else:
            print(f"  [FAILED] TensorBoard event log missing or empty in {tb_dir}")
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL CRITERIA PASSED! MOCK EXPERIMENT SYSTEM IS READY.")
    else:
        print("SOME CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_mock_experiment()
